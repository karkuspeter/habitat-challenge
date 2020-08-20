import argparse
import habitat
import random
import numpy as np
import scipy
import os
import cv2
import time
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from gibsonagents.expert import Expert
from gibsonagents.pathplanners import Dstar_planner, Astar3D, VI_planner
from gibsonagents.classic_mapping import rotate_2d, ClassicMapping
from utils.dotdict import dotdict
from utils.tfrecordfeatures import tf_bytes_feature, tf_bytelist_feature, tf_int64_feature
from habitat_utils import load_map_from_file
from vin import grid_actions_from_trajectory

from arguments import parse_args

import tensorflow as tf

from train import get_brain, get_tf_config
from common_net import load_from_file, count_number_trainable_params
from visualize_mapping import plot_viewpoints, plot_target_and_path, mapping_visualizer
from gen_habitat_data import actions_from_trajectory
from gen_planner_data import rotate_map_and_poses, Transform2D

import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import ipdb as pdb
except:
    import pdb


ACTION_SOURCE = "plan"  #"expert"  # "plan"  # "expert"  # expert
START_WITH_SPIN = True
SPIN_TARGET = np.deg2rad(370)  # np.deg2rad(270)  # np.deg2rad(360 - 70)
SPIN_DIRECTION = 1   # 1 for same direction as target, -1 for opposite direction. Opposite is better if target < 360
SUPRESS_EXCEPTIONS = False
# COST_SETTING = 0  # 2
# SOFT_COST_MAP = True

# # Give up settings - submission
# GIVE_UP_NO_PROGRESS_STEPS = 90  # 100
# NO_PROGRESS_THRESHOLD = 15
# GIVE_UP_NUM_COLLISIONS = 6  # 100 # TODO increase TODO increase later distances
# GIVE_UP_STEP_AND_DISTANCE = [[0, 340], [150, 220], [300, 150], [400, 100]]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = [[3.5, 100], [4., 120], [5., 300], [6., 400]]   # in minutes ! and  distance reduction from beginning

# # Give up settings - more agressive for submission2
# GIVE_UP_NO_PROGRESS_STEPS = 90  # 100
# NO_PROGRESS_THRESHOLD = 15
# GIVE_UP_NUM_COLLISIONS = 6
# GIVE_UP_STEP_AND_DISTANCE = [[0, 340], [150, 220], [300, 100], [400, 50]]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = [[3.5, 100], [4., 120], [5., 300], [6., 400]]   # in minutenum_wrong_frees ! and  distance reduction from beginning

# # Relaxed giveup settings for local evaluation
# GIVE_UP_NO_PROGRESS_STEPS = 100  # 100
# NO_PROGRESS_THRESHOLD = 12
# GIVE_UP_NUM_COLLISIONS = 8  # 100 # TODO increase TODO increase later distances
# GIVE_UP_STEP_AND_DISTANCE = [[0, 440], [150, 320], [300, 250], [400, 150]]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = [[10., 100], [15., 120], [20., 300], [30., 400]]   # in minutes ! and  distance reduction from beginning

# # Almost never give up -- august submission
# GIVE_UP_NO_PROGRESS_STEPS = 100
# NO_PROGRESS_THRESHOLD = 12
# GIVE_UP_NUM_COLLISIONS = 25
# GIVE_UP_STEP_AND_DISTANCE = [[0, 440], [150, 320], [300, 250], [400, 150]]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = [[10., 100], [15., 120], [20., 300], [30., 400]]   # in minutes ! and  distance reduction from beginning

# # no giveup but 300 limit for data generation
# GIVE_UP_NO_PROGRESS_STEPS = 1000
# NO_PROGRESS_THRESHOLD = 1
# GIVE_UP_NUM_COLLISIONS = 1000
# GIVE_UP_STEP_AND_DISTANCE = [[300, 1], ]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = []   # in minutes ! and  distance reduction from beginning

# No giveup
GIVE_UP_NO_PROGRESS_STEPS = 1000  # 100
NO_PROGRESS_THRESHOLD = 1
GIVE_UP_NUM_COLLISIONS = 1000
GIVE_UP_STEP_AND_DISTANCE = []   # NOTE if changing first threshold also change max map size.
GIVE_UP_TIME_AND_REDUCTION = []   # in minutes ! and  distance reduction from beginning

# # Very agressive for fast testing
# GIVE_UP_NO_PROGRESS_STEPS = 50  # 100
# NO_PROGRESS_THRESHOLD = 15
# GIVE_UP_NUM_COLLISIONS = 1
# GIVE_UP_STEP_AND_DISTANCE = [[0, 340], [100, 200], [200, 100], [300, 50]]   # NOTE if changing first threshold also change max map size.
# GIVE_UP_TIME_AND_REDUCTION = [[3.5, 100], [4., 120], [5., 300], [6., 400]]   # in minutes ! and  distance reduction from beginning


PLANNER2D_TIMEOUT = 200  # 200.  # 0.08
# PLANNER3D_TIMEOUT = 2.5  # 1.5  # 200.  # 0.08  - ------------------
MANUAL_STOP_WHEN_NEAR_TARGET = False
RECOVER_ON_COLLISION = True
COLLISION_DISTANCE_THRESHOLD = 0.6  # 0.8
MAX_SHORTCUT_TURNS = 2  # was 1 in submission
NEAR_TARGET_COLLISION_STOP_DISTANCE = 5.  # when colliding withing this radius to the goal, stop instead

# # Patch map with collisions and around target
CLEAR_TARGET_RADIUS = 4   # 5  # 0  # 5
TARGET_MAP_MARGIN = 2
ALLOW_SHRINK_MAP = True

OBSTACLE_DOWNWEIGHT_DISTANCE = 20  # from top, smaller the further
OBSTACLE_DOWNWEIGHT_SCALARS = (0.3, 0.8) # (0.3, 0.8)
EXTRA_STEPS_WHEN_EXPANDING_MAP = 30

# !!!!!!
PLOT_EVERY_N_STEP = -1
USE_ASSERTS = True
# 42 * 60 * 60 - 3 * 60 * 60   #  30 * 60 - 5 * 60  #
TOTAL_TIME_LIMIT =  42 * 60 * 60 - 90 * 60   # challenge gave up at 38h and finished at 39h so 90 minutes should be enough
# 42 hours = 2520 mins for 1000-2000 episodes.
# Average episode time should be < 75.6 sec
ERROR_ON_TIMEOUT = False   #  True
SKIP_FIRST_N_FOR_TEST = -1  # 10  # 10  # 10
VIDEO_FRAME_SKIP = 6
VIDEO_LARGE_PLOT = False
# !!!!!!!

REPLACE_WITH_RANDOM_ACTIONS = False
EXIT_AFTER_N_STEPS_FOR_SPEED_TEST = -1
FAKE_INPUT_FOR_SPEED_TEST = False
MAX_MAP_SIZE_FOR_SPEED_TEST = False

# DATA GENERATION
DATA_INCLUDE_NONPLANNED_ACTIONS = False



class DSLAMAgent(habitat.Agent):
    def __init__(self, task_config, params, env=None, logdir='./temp/', tfwriters=()):
        self.start_time = time.time()

        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.step_i = 0
        self.episode_i = -2
        self.env = env
        self.task_config = task_config
        self.tfwriters = tfwriters
        self.num_data_entries = 0
        if env is None:
            self.follower = None
            assert ACTION_SOURCE != "expert"
        else:
            self.follower = ShortestPathFollower(env._sim, 0.36/2., False)

        # if len(params.gpu) > 0 and int(params.gpu[0]) > 4:
        #     print ("Try to explicitly disable gpu")
        #     try:
        #         tf.config.experimental.set_visible_devices([], 'GPU')
        #     except Exception as e:
        #         print("Exception " + str(e))

        print (params)

        assert params.sim == 'habitat'
        self.params = params
        self.map_source = self.params.agent_map_source
        self.pose_source = self.params.agent_pose_source
        self.action_source = ACTION_SOURCE
        self.start_with_spin = START_WITH_SPIN
        self.max_confidence = 0.96   # 0.98
        self.confidence_threshold = None  # (0.2, 0.01)  # (0.35, 0.05)
        self.use_custom_visibility = (self.params.visibility_mask in [2, 20, 21])

        assert self.params.agent_map_source in ['true', 'true-saved', 'true-partial', 'pred']
        assert self.params.agent_pose_source in ['slam', 'slam-truestart', 'true']

        _, gpuname = get_tf_config(devices=params.gpu)  # sets CUDA_VISIBLE_DEVICES

        if params.skip_slam:
            print ("SKIP SLAM overwritting particles and removing noise.")
            assert self.pose_source == 'true'
            assert params.num_particles == 1
            assert params.odom_source == 'relmotion'

        self.accumulated_spin = 0.
        self.spin_direction = None

        self.map_ch = 2
        # slam_map_ch = 1
        self.max_map_size = (self.params.global_map_size, self.params.global_map_size)  # (360, 360)

        params.batchsize = 1
        params.trajlen = 1
        sensor_ch = (1 if params.mode == 'depth' else (3 if params.mode == 'rgb' else 4))
        batchsize = params.batchsize

        if params.seed is not None and params.seed > 0:
            print("Fix Numpy and TF seed to %d" % params.seed)
            tf.set_random_seed(params.seed)
            np.random.seed(params.seed)
            random.seed(params.seed)

        # Build graph for slam and planner
        with tf.Graph().as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # Choose planner
                if self.params.planner == 'astar3d':
                    self.max_map_size = (370, 370)  # also change giveup setting when changing this
                    self.fixed_map_size = True
                    # assert MAP_SOURCE != "true"
                    self.pathplanner = Astar3D(single_thread=False, max_map_size=self.max_map_size, timeout=self.params.planner_timeout)
                elif self.params.planner == 'dstar_track_fixsize':
                    self.fixed_map_size = True
                    self.pathplanner = Dstar_planner(single_thread=False, max_map_size=self.max_map_size)
                elif self.params.planner in ['dstar_track', 'dstar2d']:
                    self.max_map_size = (900, 900)
                    self.fixed_map_size = False
                    self.pathplanner = Dstar_planner(single_thread=False, max_map_size=self.max_map_size)
                elif self.params.planner in ['vi4', 'vi8', 'vi4-e1', 'vi8-e1']:
                    self.fixed_map_size = True
                    self.pathplanner = VI_planner(max_map_size=(None, None), brain="trueplanner", params=self.params, connect8=(self.params.planner in ['vi8', 'vi8-e1']))
                elif self.params.planner in ['vin', 'vin-e1']:
                    self.fixed_map_size = True
                    self.pathplanner = VI_planner(max_map_size=(None, None), brain="plannerbrain_v16", params=self.params, connect8=False)
                else:
                    raise ValueError("Unknown planner %s"%self.params.planner)

                self.top_down_map_resolution = self.task_config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION

                # Test data and network
                assert params.target in ["traj"]

                train_brain = get_brain(params.brain, params)
                req = train_brain.requirements()
                self.brain_requirements = req
                self.local_map_shape = req.local_map_size

                # Build slam brain with placeholder inputs
                # global_map_input = tf.placeholder(shape=(batchsize, None, None, slam_map_ch,), dtype=tf.float32)
                # self.images_input = tf.placeholder(shape=(batchsize, None) + req.sensor_shape + (sensor_ch,),
                #                               dtype=tf.float32)
                # self.visibility_input = (
                #     tf.placeholder(shape=(batchsize, None) + tuple(req.local_map_size) + (1,), dtype=tf.float32)
                #     if params.visibility_mask == 2
                #     else tf.zeros((batchsize, None, 0, 0, 1)))

                self.new_images_input = tf.placeholder(shape=(batchsize, 1) + req.sensor_shape + (sensor_ch,),
                                              dtype=tf.float32)
                self.last_images_input = tf.placeholder(shape=(batchsize, 1) + req.sensor_shape + (sensor_ch,),
                                                       dtype=tf.float32)
                self.past_visibility_input = (
                    tf.placeholder(shape=(batchsize, None) + tuple(req.local_map_size) + (1,), dtype=tf.float32)
                    if self.use_custom_visibility
                    else tf.zeros((batchsize, None, 0, 0, 1)))
                self.visibility_input = (
                    tf.placeholder(shape=(batchsize, 1) + tuple(req.local_map_size) + (1,), dtype=tf.float32)
                    if self.use_custom_visibility
                    else tf.zeros((batchsize, 1, 0, 0, 1)))
                self.past_local_maps_input = tf.placeholder(shape=(batchsize, None) + tuple(req.local_map_size) + (1,), dtype=tf.float32)
                self.past_needed_image_features_input = tf.placeholder(shape=(batchsize, None) + tuple(req.local_map_size) + (req.latent_map_ch,), dtype=tf.float32)

                self.particle_xy_input = tf.placeholder(shape=(batchsize, None, params.num_particles, 2,), dtype=tf.float32)
                self.particle_yaw_input = tf.placeholder(shape=(batchsize, None, params.num_particles, 1,), dtype=tf.float32)
                self.last_step_particle_logits_input = tf.placeholder(shape=(batchsize, params.num_particles),
                                                                 dtype=tf.float32)
                self.new_action_input = tf.placeholder(shape=(batchsize, 1, 1,), dtype=tf.int32)
                self.new_rel_xy_input = tf.placeholder(shape=(batchsize, 1, 2,), dtype=tf.float32)
                self.new_rel_yaw_input = tf.placeholder(shape=(batchsize, 1, 1,), dtype=tf.float32)

                self.true_xy_input = tf.placeholder(shape=(batchsize, None, 2,), dtype=tf.float32)
                self.true_yaw_input = tf.placeholder(shape=(batchsize, None, 1,), dtype=tf.float32)

                self.inference_timesteps_input = tf.placeholder(shape=(batchsize, None), dtype=tf.int32)  # indexes history to be used for slam update

                self.global_map_shape_input = tf.placeholder(shape=(2, ), dtype=tf.int32)

                if self.params.obstacle_downweight:
                    custom_obstacle_prediction_weight = Expert.get_obstacle_prediction_weight(OBSTACLE_DOWNWEIGHT_DISTANCE, OBSTACLE_DOWNWEIGHT_SCALARS, self.local_map_shape)
                else:
                    custom_obstacle_prediction_weight = None

                if FAKE_INPUT_FOR_SPEED_TEST:
                    self.inference_outputs = train_brain.sequential_localization_with_past_and_pred_maps(
                        tf.zeros_like(self.past_local_maps_input), tf.ones_like(self.past_visibility_input),
                        tf.zeros_like(self.past_needed_image_features_input),
                        tf.zeros_like(self.new_images_input), tf.zeros_like(self.true_xy_input), tf.zeros_like(self.true_yaw_input),
                        tf.zeros_like(self.visibility_input),
                        tf.zeros_like(self.particle_xy_input), tf.zeros_like(self.particle_yaw_input),
                        tf.zeros_like(self.new_action_input), tf.zeros_like(self.new_rel_xy_input), tf.zeros_like(self.new_rel_yaw_input),
                        particle_logits_acc=tf.zeros_like(self.last_step_particle_logits_input),
                        global_map_shape=self.global_map_shape_input,
                        max_confidence=self.max_confidence)
                else:
                    ###
                    # THIS IS USED NORMALLY
                    ###
                    self.inference_outputs = train_brain.sequential_localization_with_past_and_pred_maps(
                        self.past_local_maps_input, self.past_visibility_input, self.past_needed_image_features_input,
                        self.new_images_input, self.true_xy_input, self.true_yaw_input, self.visibility_input,
                        self.particle_xy_input, self.particle_yaw_input,
                        self.new_action_input, self.new_rel_xy_input, self.new_rel_yaw_input,
                        inference_timesteps=self.inference_timesteps_input,
                        particle_logits_acc=self.last_step_particle_logits_input,
                        global_map_shape=(tuple(self.max_map_size) if self.fixed_map_size else self.global_map_shape_input),  # self.global_map_shape_input,  tuple(self.max_map_size),
                        max_confidence=self.max_confidence,
                        custom_obstacle_prediction_weight=custom_obstacle_prediction_weight,
                        last_images=self.last_images_input,
                    )
                if PLOT_EVERY_N_STEP < 0:
                    self.inference_outputs = self.drop_output(self.inference_outputs, drop_names=['tiled_visibility_mask'])
                self.inference_outputs_without_map = self.drop_output(self.inference_outputs, drop_names=['global_map_logodds'])

                # self.inference_outputs = train_brain.sequential_localization_with_map_prediction(
                #     self.images_input, self.true_xy_input, self.true_yaw_input, self.visibility_input,
                #     self.particle_xy_input, self.particle_yaw_input,
                #     self.new_action_input, self.new_rel_xy_input, self.new_rel_yaw_input,
                #     particle_logits_acc=self.last_step_particle_logits_input)

                # self.inference_outputs = train_brain.sequential_localization_with_past_and_pred_maps(
                #     self.past_local_maps_input, self.past_visibility_input, NEED_IMAGES,
                #     self.new_images_input, self.true_xy_input, self.true_yaw_input, self.visibility_input,
                #     self.particle_xy_input, self.particle_yaw_input,
                #     self.new_action_input, self.new_rel_xy_input, self.new_rel_yaw_input,
                #     particle_logits_acc=self.last_step_particle_logits_input)
                #
                # TODO pass in map inference inputs. Could produce one processed and one unprocess map for slam.

                # self.true_map_input = tf.placeholder(shape=self.max_map_size + (1, ), dtype=tf.uint8)
                # self.images_input = tf.placeholder(shape=req.sensor_shape + (sensor_ch,), dtype=tf.float32)
                # self.xy_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                # self.yaw_input = tf.placeholder(shape=(1, ), dtype=tf.float32)
                # # self.action_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                # actions = tf.zeros((1, 1, 2), dtype=tf.float32)
                # self.global_map_input = tf.placeholder(shape=self.max_map_size + (self.map_ch, ), dtype=tf.float32)
                # self.visibility_input = tf.placeholder(shape=self.local_map_shape + (1, ), dtype=tf.uint8) if self.use_custom_visibility else None
                # local_obj_map_labels = tf.zeros((1, 1, ) + self.local_map_shape + (1, ), dtype=np.uint8)
                #
                # self.inference_outputs = train_brain.sequential_inference(
                #     self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                #     actions, prev_global_map_logodds=self.global_map_input[None],
                #     local_obj_maps=local_obj_map_labels,
                #     confidence_threshold=self.confidence_threshold,
                #     max_confidence=self.max_confidence,
                #     max_obj_confidence=0.8,
                #     custom_visibility_maps=None if self.visibility_input is None else self.visibility_input[None, None],
                #     is_training=True)

                # self.true_map_input = tf.zeros(shape=self.max_map_size + (1, ), dtype=tf.uint8)
                # self.images_input = tf.zeros(shape=req.sensor_shape + (sensor_ch,), dtype=tf.float32)
                # self.xy_input = tf.ones(shape=(2,), dtype=tf.float32)
                # self.yaw_input = tf.zeros(shape=(1, ), dtype=tf.float32)
                # # self.action_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                # actions = tf.ones((1, 1, 2), dtype=tf.float32)
                # self.global_map_input = tf.ones(shape=self.max_map_size + (self.map_ch, ), dtype=tf.float32)
                # self.visibility_input = tf.ones(shape=self.local_map_shape + (1, ), dtype=tf.uint8) if self.use_custom_visibility else None
                # local_obj_map_labels = tf.zeros((1, 1, ) + self.local_map_shape + (1, ), dtype=np.uint8)
                #
                # self.inference_outputs = train_brain.sequential_inference(
                #     self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                #     actions, prev_global_map_logodds=self.global_map_input[None],
                #     local_obj_maps=local_obj_map_labels,
                #     confidence_threshold=self.confidence_threshold,
                #     max_confidence=self.max_confidence,
                #     max_obj_confidence=0.8,
                #     custom_visibility_maps=None if self.visibility_input is None else self.visibility_input[None, None],
                #     is_training=True)

                # Add the variable initializer Op.
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            count_number_trainable_params(verbose=True)

            # training session
            gpuconfig, gpuname = get_tf_config(devices=params.gpu)
            self.sess = tf.Session(config=gpuconfig)

            # # Debug
            # self.sess.run(init)  # withouth init if a variable is not loaded we get an error
            # outputs = self.sess.run(self.inference_outputs)
            # print ("Success")
            # pdb.set_trace()

            # self.sess.run(init)  # withouth init if a variable is not loaded we get an error

            load_from_file(self.sess, params.load, partialload=params.partialload, loadcenter=[],
                           skip=params.loadskip, autorename=False)

        self.global_map_logodds = None
        self.xy = None
        self.yaw = None
        self.target_xy = None
        self.step_i = -1
        self.t = time.time()

        # video
        self.frame_traj_data = []
        self.num_videos_saved = 0
        self.summary_str = ""
        self.filename_addition = ""
        self.logdir = logdir

        self.reset()

    def get_scene_name(self):
        scene_path =  "unknown" if self.env is None else self.env._sim._current_scene
        scene_name = scene_path.split('/')[-1].split('.')[0]
        return scene_name

    def reset(self):
        self.step_i = -1
        self.episode_i += 1
        self.t = time.time()
        self.episode_t = self.t
        self.accumulated_spin = 0.
        self.spin_direction = None
        self.distance_history = []
        self.raw_xy_transform = lambda xys: xys
        self.raw_yaw_offset = 0.

        self.recover_step_i = 0
        self.num_collisions = 0
        self.num_shortcut_actions = 0
        self.num_wrong_obstacle = 0
        self.num_wrong_free = 0
        self.num_wrong_free_area = 0
        self.num_wrong_free_area2 = 0
        self.num_wrong_free_area3 = 0
        self.map_mismatch_count = 0
        self.plan_times = []

        assert self.params.recoverpolicy in ['back5', 'back1']
        num_recover_back_steps = (5 if self.params.recoverpolicy == 'back5' else 1)
        self.recover_policy = [3] * 6 + [1] * num_recover_back_steps

        self.global_map_logodds = None  # will initialize in act() np.zeros((1, 1) + (1, ), np.float32)
        self.collision_timesteps = []

        for tfwriter in self.tfwriters:
            tfwriter.flush()

        self.pathplanner.reset()

        self.reset_video_writer()

        print ("Resetting agent %d. Scene %s."%(self.episode_i, self.get_scene_name()))

    def drop_output(self, outputs, drop_names):
        return dotdict({key: val for key, val in outputs.items() if key not in drop_names})

    def plan_and_control(self, xy, yaw, target_xy, global_map_pred, ang_vel, target_fi, allow_shrink_map=False):

        if self.start_with_spin and np.abs(self.accumulated_spin) < SPIN_TARGET and self.step_i < 40:
            if self.spin_direction is None:
                self.spin_direction = SPIN_DIRECTION * np.sign(target_fi)  # spin opposite direction to the goal
            self.accumulated_spin += ang_vel
            # spin

            status_message = "%d: spin %f: %f"%(self.step_i, self.spin_direction, self.accumulated_spin)

            action = (2 if self.spin_direction > 0 else 3)
            planned_path = np.zeros((0, 2))
            return action, planned_path, status_message, (global_map_pred * 255.).astype(np.uint8)

        if not self.params.soft_cost_map:
            assert global_map_pred.dtype == np.float32
            global_map_pred = (global_map_pred * 255.).astype(np.uint8)

        if allow_shrink_map:
            xy, target_xy, global_map_pred, offset_xy = self.shrink_map(xy, target_xy, global_map_pred)
        else:
            offset_xy = None

        # Scan map and cost graph.
        scan_graph, eroded_scan_map, normal_scan_map, costmap = Expert.get_graph_and_eroded_map(
            raw_trav_map=global_map_pred[..., :1],
            trav_map_for_simulator=global_map_pred[..., :1],
            raw_scan_map=global_map_pred,
            rescale_scan_map=1.,
            erosion=self.params.map_erosion_for_planning,
            build_graph=False,
            interactive_channel=False,
            cost_setting=self.params.cost_setting,
            soft_cost_map=self.params.soft_cost_map,
        )

        # plt.figure()
        # plt.imshow(costmap)
        # plt.show()
        # pdb.set_trace()

        # scan_map = global_map_pred[..., :1]
        #
        # scan_map = cv2.erode(scan_map, kernel=np.ones((3, 3)))
        # scan_map[scan_map<255] = 0
        #
        # costmap = np.zeros_like(global_map_pred, dtype=np.float32)
        # costmap[global_map_pred == 0] = 1000.
        #
        # temp_map1 = scan_map
        # temp_map2 = cv2.erode(temp_map1, kernel=np.ones((3, 3)))
        # temp_filter = np.logical_and(temp_map2 < 255, temp_map1 == 255)
        # costmap[temp_filter] = 100.
        #
        # temp_map1 = scan_map
        # temp_map2 = cv2.erode(temp_map1, kernel=np.ones((7, 7)))
        # temp_filter = np.logical_and(temp_map2 < 255, temp_map1 == 255)
        # costmap[temp_filter] = 1.
        assert self.params.goalpolicy in ['twostep', 'none']
        start_time = time.time()

        if self.params.planner == 'astar3d':
            assert self.params.goalpolicy in ['twostep']  # need to add support below for removing twostep
            action, obstacle_distance, planned_path, status_message = Expert.discrete3d_policy(
                scan_map=eroded_scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=costmap,
                pathplanner=self.pathplanner)
        elif self.params.planner == 'dstar2d':
            assert self.params.goalpolicy in ['twostep']  # need to add support below for removing twostep
            action, obstacle_distance, planned_path, status_message = Expert.discrete_policy(
                scan_map=eroded_scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=costmap,
                use_asserts=True,
                shortest_path_fn=lambda _, source_tuple, target_tuple, cost_map, scan_map: self.pathplanner.dstar_path(
                    cost_map, source_tuple, target_tuple, timeout=PLANNER2D_TIMEOUT))
        elif self.params.planner in ['dstar_track', 'dstar_track_fixsize']:
            action, obstacle_distance, planned_path, status_message = Expert.discrete_tracking_policy(
                scan_map=eroded_scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=costmap,
                use_lookahead=True, use_twostep_approach=(self.params.goalpolicy == 'twostep'),
                shortest_path_fn=lambda _, source_tuple, target_tuple, cost_map, scan_map: self.pathplanner.dstar_path(
                    cost_map, source_tuple, target_tuple, timeout=PLANNER2D_TIMEOUT))
        elif self.params.planner in ['vi4', 'vi8', 'vin']:
            action, obstacle_distance, planned_path, status_message = Expert.discrete_tracking_policy(
                scan_map=eroded_scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=costmap,
                use_lookahead=True, use_twostep_approach=(self.params.goalpolicy == 'twostep'),
                shortest_path_fn=lambda _, source_tuple, target_tuple, cost_map, scan_map: self.pathplanner.vi_path(
                    scan_map, cost_map, source_tuple, target_tuple, sess=self.sess))
        elif self.params.planner in ['vi4-e1', 'vi8-e1', 'vin-e1']:
            action, obstacle_distance, planned_path, status_message = Expert.discrete_tracking_policy(
                scan_map=eroded_scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=costmap,
                use_lookahead=True, use_twostep_approach=(self.params.goalpolicy == 'twostep'),
                shortest_path_fn=lambda _, source_tuple, target_tuple, cost_map, scan_map: self.pathplanner.vi_path(
                    normal_scan_map, cost_map, source_tuple, target_tuple, sess=self.sess))
        else:
            raise ValueError("Unknown planner %s"%(self.params.planner))

        status_message = "%d/%d: %.2f %s"%(self.episode_i, self.step_i, time.time()-self.t, status_message)
        self.t = time.time()
        self.plan_times.append(time.time() - start_time)

        if allow_shrink_map:
            planned_path = planned_path + offset_xy[None]

        return action, planned_path, status_message, eroded_scan_map

    def act(self, observations):
        if SUPRESS_EXCEPTIONS:
            try:
                return self.wrapped_act(observations)
            except Exception as e:
                print ("Excpetion " + str(e))
                return {"action": 0, "xy_error": 0.}
        else:
            return self.wrapped_act(observations)

    def wrapped_act(self, observations):
        # import ipdb; ipdb.set_trace()
        if REPLACE_WITH_RANDOM_ACTIONS and self.episode_i > 2:
            self.step_i += 1
            if self.step_i > 100:
                action = 0
            else:
                action = np.random.choice([1, 2, 3])
            return {"action": action, "xy_error": 0.}

        time_now = time.time()
        time_since_beginning = time_now - self.start_time

        initial_target_r_meters, initial_target_fi = observations['pointgoal']
        if self.step_i == -1:
            self.step_i += 1
            # self.initial_xy = np.zeros((2, ), np.float32)
            # self.initial_yaw = np.zeros((1, ), np.float32)

            self.episode_t = time_now

            if self.env is not None:
                return {"action": 3}  # turn right, because first step does not provide the top down map
            # otherwise continue below

        if TOTAL_TIME_LIMIT > 0 and time_since_beginning > TOTAL_TIME_LIMIT:
            print ("Giving up because total time limit of %d sec reached."%TOTAL_TIME_LIMIT)
            if ERROR_ON_TIMEOUT:
                raise ValueError("Timeout.. only for minival!")
            return {"action": 0, "xy_error": 0.}

        if SKIP_FIRST_N_FOR_TEST > 0 and self.episode_i < SKIP_FIRST_N_FOR_TEST:
            print ("Skip")
            return {"action": 0, "xy_error": 0.}

        # if EXIT_AFTER_N_STEPS_FOR_SPEED_TEST > 0 and self.step_i > EXIT_AFTER_N_STEPS_FOR_SPEED_TEST:
        #     raise SystemExit

        # Check for possible shortcut
        shortcut_action = None
        need_map = True
        if self.params.skip_plan_when_turning and len(self.pathplanner.cached_action_path) >= 2 and self.num_shortcut_actions < MAX_SHORTCUT_TURNS:
            cached_next_action = self.pathplanner.cached_action_path[1]
            if cached_next_action in [2, 3]:  # turning
                print ("Shortcut turn action")
                shortcut_action = cached_next_action
                need_map = False
                self.num_shortcut_actions += 1
                self.pathplanner.num_timeouts += 1
                self.pathplanner.cached_action_path = self.pathplanner.cached_action_path[1:]
            else:
                self.num_shortcut_actions = 0
        else:
            self.num_shortcut_actions = 0

        if RECOVER_ON_COLLISION and self.recover_step_i > 0:
            need_map = False

        # True pose and map
        if self.env is not None:
            # When using slam, must run with --habitat_eval local not localtest.
            # Thats because with --localtest we skip the first step, but that ruins the goal observation.
            assert self.pose_source != 'slam'

            info = self.env.get_metrics()
            agent_pos = self.env.sim.get_agent_state().position
            goal_pos = self.env.current_episode.goals[0].position

            # First deal with observed map and optionally samplre random rotation
            true_global_map = info['top_down_map']['map']
            true_global_map = (true_global_map > 0).astype(np.uint8) * 255
            true_global_map = np.atleast_3d(true_global_map)

            if self.step_i > 0:
                # Count pixels that used to be free but not in the latest map
                if not self.params.random_rotations:
                    new_map_mismatch_count = np.count_nonzero(np.logical_and(true_global_map == 0, self.last_true_global_map))
                    if new_map_mismatch_count > 200:
                        print ("TOO MANY MAP MISMATCHES %d"%new_map_mismatch_count)
                        # assert False
                    self.map_mismatch_count += new_map_mismatch_count
                true_global_map = self.last_true_global_map  # keep the first
            elif self.map_source == 'true-saved':
                saved_global_map = load_map_from_file(scene_id=self.get_scene_name(), height=agent_pos[1])
                assert saved_global_map.dtype == np.uint8
                assert saved_global_map.shape == true_global_map.shape
                self.map_mismatch_count  = np.count_nonzero(np.logical_and(saved_global_map, true_global_map))
                assert not self.params.random_rotations
                true_global_map = saved_global_map
                self.last_true_global_map = saved_global_map
            else:
                # rotate true map once and keep this for the episode. Remember the transformation and apply it from all raw observations
                if self.params.random_rotations:
                    self.raw_yaw_offset = np.random.rand() * 2. * np.pi
                    true_global_map, new_poses, transform = rotate_map_and_poses(true_global_map,  self.raw_yaw_offset, poses=[np.zeros((1, 2), np.float32)], constant_value=0)
                    assert true_global_map.dtype == np.uint8
                    # Reapply threshold
                    true_global_map = (true_global_map > 128).astype(np.uint8) * 255
                    self.raw_xy_transform = transform
                self.map_mismatch_count  = 0
                self.last_true_global_map = true_global_map

            # Deal with observed pose
            true_xy = np.array(info['top_down_map']['agent_map_coord'])  # x: downwards; y: rightwars
            true_xy = self.raw_xy_transform(true_xy[None])[0]
            true_yaw = info['top_down_map']['agent_angle']  # 0 downwards, positive ccw. Forms stanard coord system with x and y.
            true_yaw = true_yaw + self.raw_yaw_offset
            true_yaw = np.array((true_yaw, ), np.float32)
            true_yaw = (true_yaw + np.pi) % (2 * np.pi) - np.pi  # normalize
            # Recover from simulator pos
            true_xy_from_pos = maps.to_grid(agent_pos[0], agent_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX,
                                   (self.top_down_map_resolution, self.top_down_map_resolution), keep_float=True)
            true_xy_from_pos = self.raw_xy_transform(np.array(true_xy_from_pos, np.float32)[None])[0]
            offset_xy = true_xy - true_xy_from_pos
            true_target_xy = maps.to_grid(goal_pos[0], goal_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX,
                                          (self.top_down_map_resolution, self.top_down_map_resolution), keep_float=True)
            true_target_xy = self.raw_xy_transform(np.array(true_target_xy, np.float32)[None])[0]
            true_target_xy += offset_xy
            del offset_xy

        else:
            true_xy = np.zeros((2,), np.float32)
            true_yaw = np.zeros((1,), np.float32)
            true_target_xy = np.zeros((2,), np.float32)

            true_global_map = np.zeros([self.max_map_size[0], self.max_map_size[1], 1], np.float32)

        # Initialize everything
        if self.step_i == 0:
            if self.pose_source in ['true', 'slam-truestart']:
                # Initialize with true things. Only makes sense if we access it
                assert self.env is not None
                self.true_xy_offset = -true_xy.astype(np.int32)
                # self.true_xy_transform = Transform2D
                # self.true_xy_transform.add_translation()
                if self.fixed_map_size:
                    self.global_map_logodds = np.zeros((self.max_map_size[0], self.max_map_size[1], 1), np.float32)   # np.zeros(true_global_map.shape, np.float32)
                else:
                    self.global_map_logodds = np.zeros((1, 1, 1), np.float32)   # np.zeros(true_global_map.shape, np.float32)
                self.prev_yaw = true_yaw
                self.xy = true_xy + self.true_xy_offset
                self.yaw = true_yaw
                particle_xy0 = np.tile((self.xy)[None], [self.params.num_particles, 1])
                particle_yaw0 = np.tile(self.yaw[None], [self.params.num_particles, 1])

                # Target from observed distance. Can only use it after reset
                initial_target_r_meters, initial_target_fi = observations['pointgoal']
                initial_target_r = initial_target_r_meters / 0.05  # meters to grid cells
                # assumes initial pose is 0.0
                initial_target_xy = rotate_2d(np.array([initial_target_r, 0.], np.float32), initial_target_fi + true_yaw + np.deg2rad(30)) + true_xy + self.true_xy_offset

                # Target from observed distance. Can only use it after reset
                initial_target_r_meters, initial_target_fi = observations['pointgoal_with_gps_compass']
                target_r = initial_target_r_meters / 0.05  # meters to grid cells
                # assumes initial pose is 0.0
                observed_target_xy = rotate_2d(np.array([target_r, 0.], np.float32), initial_target_fi + true_yaw) + true_xy + self.true_xy_offset

                print ("Target observed: (%d, %d)  true: (%d, %d) initial (%d, %d)"%(
                    observed_target_xy[0], observed_target_xy[1], true_target_xy[0] + self.true_xy_offset[0], true_target_xy[1] + self.true_xy_offset[1], initial_target_xy[0], initial_target_xy[1]))
                if np.linalg.norm(observed_target_xy - (true_target_xy + self.true_xy_offset)) > 0.001:
                    pdb.set_trace()
                self.target_xy = observed_target_xy

            elif self.pose_source == "slam":
                self.true_xy_offset = np.zeros((2,), np.int32)  # we dont know
                if self.fixed_map_size:
                    self.global_map_logodds = np.zeros((self.max_map_size[0], self.max_map_size[1], 1), np.float32)   # np.zeros(true_global_map.shape, np.float32)
                else:
                    self.global_map_logodds = np.zeros((1, 1, 1), np.float32)   # np.zeros(true_global_map.shape, np.float32)

                self.prev_yaw = 0.
                self.xy = np.zeros((2, ), np.float32)
                self.yaw = np.zeros((1, ), np.float32)
                particle_xy0 = np.zeros((self.params.num_particles, 2), np.float32)
                particle_yaw0 = np.zeros((self.params.num_particles, 1), np.float32)

                # Target from observed distance. Can only use it after reset
                initial_target_r_meters, initial_target_fi = observations['pointgoal']
                target_r = initial_target_r_meters / 0.05  # meters to grid cells
                # assumes initial pose is 0.0
                observed_target_xy = rotate_2d(np.array([target_r, 0.], np.float32), initial_target_fi)
                self.target_xy = observed_target_xy

            else:
                raise ValueError("Unknown pose estimation source.")

            self.particle_xy_list = [particle_xy0]
            self.particle_yaw_list = [particle_yaw0]
            self.particle_logit_acc_list = [np.zeros((self.params.num_particles,), np.float32)]
            self.xy_loss_list = [0.]
            self.yaw_loss_list = [0.]
            self.true_xy_traj = [true_xy]
            self.true_yaw_traj = [true_yaw]
            self.action_traj = []

        # Resize map and add offset
        map_shape = self.global_map_logodds.shape
        if self.fixed_map_size:
            xy_map_margin = 10  # this is before slam update. a single step can move 6 cells plus estimation may change

            # Keep a fixed map size. Dont even update it, only move the offset, such that center point is between current pose and goal
            assert map_shape[:2] == self.max_map_size
            assert self.max_map_size[0] == self.max_map_size[1]

            center_xy = (self.xy + self.target_xy) * 0.5
            desired_center_xy = np.array(self.max_map_size, np.float32) * 0.5
            offset_xy = (desired_center_xy - center_xy).astype(np.int)

            new_xy = self.xy + offset_xy

            # Handle the case when xy would fall out of the map area or would be too near the edge.
            # These will be only nonzero if xy is outside the allowed area
            # if np.any(new_xy < xy_map_margin):
            offset_xy += np.ceil(np.maximum(xy_map_margin - new_xy, 0.)).astype(np.int32)
            # if np.any(new_xy >= self.max_map_size[0] - xy_map_margin):
            offset_xy -= np.ceil(np.maximum(new_xy - (self.max_map_size[0] - xy_map_margin), 0.)).astype(np.int32)

            self.particle_xy_list = [xy + offset_xy for xy in self.particle_xy_list]
            self.target_xy += offset_xy
            self.true_xy_offset += offset_xy
            self.xy += offset_xy

            # Handle the case when target is outside of the map area
            if np.any(self.target_xy < TARGET_MAP_MARGIN) or np.any(self.target_xy >= self.max_map_size[0] - TARGET_MAP_MARGIN):
                # Find the free map cell closest to the target
                global_map_pred = ClassicMapping.inverse_logodds(self.global_map_logodds)
                free_x, free_y = np.nonzero(np.squeeze(global_map_pred[TARGET_MAP_MARGIN:-TARGET_MAP_MARGIN, TARGET_MAP_MARGIN:-TARGET_MAP_MARGIN], axis=-1) >= 0.5)
                free_xy = np.stack([free_x, free_y], axis=-1)
                free_xy = free_xy.astype(np.float32)
                free_xy += 0.5
                free_xy += TARGET_MAP_MARGIN
                dist = np.linalg.norm(free_xy - self.target_xy[None], axis=1)
                # pretend the closest free cell is the target
                self.target_xy_for_planning = free_xy[np.argmin(dist)]
                print ("Moving target within the map: %s --> %s"%(str(self.target_xy), str(self.target_xy_for_planning)))
            else:
                self.target_xy_for_planning = self.target_xy.copy()

        else:
            # Expand map and offset pose if needed, such that target and the surrounding of current pose are all in the map.
            if MAX_MAP_SIZE_FOR_SPEED_TEST:
                offset_xy = np.array(((self.max_map_size[0]-map_shape[0])//2, (self.max_map_size[1]-map_shape[1])//2), np.int32)
                expand_xy = offset_xy.copy()

            else:
                local_map_max_extent = 110  # TODO need to adjust to local map size and scaler
                local_map_max_extent += 10  # to account for how much the robot may move in one step, including max overshooting
                target_margin = 8
                min_particle_xy = self.particle_xy_list[-1].min(axis=0)  # last is step is enough because earliers could arleady fit on map
                max_particle_xy = self.particle_xy_list[-1].max(axis=0)

                min_x = int(min(self.target_xy[0] - target_margin, min_particle_xy[0] - local_map_max_extent) - 1)
                min_y = int(min(self.target_xy[1] - target_margin, min_particle_xy[1] - local_map_max_extent) - 1)
                max_x = int(max(self.target_xy[0] + target_margin, max_particle_xy[0] + local_map_max_extent) + 1)
                max_y = int(max(self.target_xy[1] + target_margin, max_particle_xy[1] + local_map_max_extent) + 1)
                offset_xy = np.array([max(0, -min_x), max(0, -min_y)])
                expand_xy = np.array([max(0, max_x+1-map_shape[0]), max(0, max_y+1-map_shape[1])])

            is_offset = np.any(offset_xy > 0)
            is_expand =  np.any(expand_xy > 0)
            if is_offset:
                offset_xy += 0 if MAX_MAP_SIZE_FOR_SPEED_TEST else EXTRA_STEPS_WHEN_EXPANDING_MAP
                self.particle_xy_list = [xy + offset_xy for xy in self.particle_xy_list]
                self.target_xy += offset_xy
                self.true_xy_offset += offset_xy

            if is_expand:
                expand_xy += 0 if MAX_MAP_SIZE_FOR_SPEED_TEST else EXTRA_STEPS_WHEN_EXPANDING_MAP

            if is_offset or is_expand:
                prev_shape = self.global_map_logodds.shape

                self.global_map_logodds = np.pad(
                    self.global_map_logodds, [[offset_xy[0], expand_xy[0]], [offset_xy[1], expand_xy[1]], [0, 0]],
                    mode='constant', constant_values=0.)

                print ("Increasing map size: (%d, %d) --> (%d, %d)  offset (%d, %d), expand (%d, %d)"%(
                    prev_shape[0], prev_shape[1], self.global_map_logodds.shape[0], self.global_map_logodds.shape[1],
                    offset_xy[0], offset_xy[1], expand_xy[0], expand_xy[1]))

            excess_xy = np.array(self.global_map_logodds.shape[:2], np.int32) -  np.array(self.max_map_size[:2], np.int32)
            excess_xy = np.maximum(excess_xy, np.zeros_like(excess_xy))
            if np.any(excess_xy > 0):
                print ("Reducing map to fit max size (%d, %d)"%(excess_xy[0], excess_xy[1]))
                if self.target_xy[0] > self.global_map_logodds.shape[0] // 2:
                    self.global_map_logodds = self.global_map_logodds[excess_xy[0]:]
                else:
                    self.global_map_logodds = self.global_map_logodds[:-excess_xy[0]]
                if self.target_xy[1] > self.global_map_logodds.shape[1] // 2:
                    self.global_map_logodds = self.global_map_logodds[:, excess_xy[1]:]
                else:
                    self.global_map_logodds = self.global_map_logodds[:, :-excess_xy[1]]

            self.target_xy_for_planning = self.target_xy.copy()
            map_shape = self.global_map_logodds.shape

        # Offset true map
        if self.env is not None:
            reduce_xy = np.maximum(-self.true_xy_offset, np.zeros((2,), np.int32)).astype(np.int32)
            extend_xy = np.maximum(self.true_xy_offset, np.zeros((2,), np.int32)).astype(np.int32)
            global_map_label = true_global_map * (1./255.)
            global_map_label = global_map_label[reduce_xy[0]:, reduce_xy[1]:]
            global_map_label = np.pad(global_map_label, [[extend_xy[0], 0], [extend_xy[1], 0], [0, 0]])
            global_map_label = np.pad(global_map_label, [[0, max(map_shape[0]-global_map_label.shape[0], 0)], [0, max(map_shape[1]-global_map_label.shape[1], 0)], [0, 0]])
            global_map_label = global_map_label[:map_shape[0], :map_shape[1]]
            assert global_map_label.shape == map_shape
        else:
            global_map_label = None

        # Get image observations
        rgb = observations['rgb']
        depth = observations['depth']
        rgb = cv2.resize(rgb, (160, 90), )
        depth = cv2.resize(depth, (160, 90), )  # interpolation=cv2.INTER_NEAREST)
        depth = np.atleast_3d(depth)
        if self.params.mode == 'both':
            images = np.concatenate([depth, rgb], axis=-1)  # these are 0..1  float format
        elif self.params.mode == 'depth':
            images = depth
        else:
            images = rgb
        images = (images * 255).astype(np.uint8)
        images = np.array(images, np.float32)
        # images = images * 255  # to unit8 0..255 format
        images = images * (2. / 255.) - 1.  # to network input -1..1 format

        # Get visibility map from depth if needed
        if self.visibility_input is not None:
            visibility_map = ClassicMapping.is_visible_from_depth(depth, self.local_map_shape, sim=self.params.sim, zoom_factor=self.brain_requirements.transform_window_scaler,
                                                                  fix_habitat_depth=self.params.fix_habitat_depth)
            visibility_map = visibility_map[:, :, None].astype(np.float32)
            assert np.all(visibility_map <= 1.)
        else:
            visibility_map = np.zeros((1, 1, 1), dtype=np.float32)

        # # Map prediction only, using known pose
        # last_global_map_input = np.zeros(self.max_map_size + (self.map_ch, ), np.float32)
        # last_global_map_input[:map_shape[0], :map_shape[1]] = self.global_map_logodds
        # true_map_input = np.zeros(self.max_map_size + (1, ), np.uint8)
        # true_map_input[:global_map_label.shape[0], :global_map_label.shape[1]] = global_map_label
        #
        # feed_dict = {
        #     self.images_input: images, self.xy_input: true_xy, self.yaw_input: np.array((true_yaw, )),
        #     self.global_map_input: last_global_map_input,
        #     self.true_map_input: true_map_input,
        # }
        # if self.visibility_input is not None:
        #     visibility_map = ClassicMapping.is_visible_from_depth(depth, self.local_map_shape, sim=self.params.sim, zoom_factor=self.brain_requirements.transform_window_scaler)
        #     visibility_map = visibility_map[:, :, None].astype(np.uint8)
        #     feed_dict[self.visibility_input] = visibility_map
        #
        # mapping_output = self.run_inference(feed_dict)
        # global_map_logodds = np.array(mapping_output.global_map_logodds[0, -1])  # squeeze batch and traj
        # global_map_logodds = global_map_logodds[:map_shape[0], :map_shape[1]]
        # self.global_map_logodds = global_map_logodds

        # SLAM prediction
        if self.step_i == 0:
            # For the first step we dont do pose update, but we need to obtain local maps and image features
            self.image_traj = [images.copy()]
            self.visibility_traj = [visibility_map.copy()]

            # Get local maps for first
            feed_dict = {
                self.new_images_input: images[None, None],
                self.visibility_input: visibility_map[None, None],
            }
            new_local_maps, new_image_features = self.sess.run([self.inference_outputs['new_local_maps'], self.inference_outputs['new_image_features']], feed_dict=feed_dict)
            self.local_map_traj = [new_local_maps[0, 0]]
            self.image_features_traj = [new_image_features[0, 0]]

            slam_outputs = None
            # Transform predictions
            global_map_true_partial = None
            assert self.global_map_logodds.shape[-1] == 1
            global_map_pred = ClassicMapping.inverse_logodds(self.global_map_logodds)

            slam_xy = np.mean(self.particle_xy_list[-1], axis=0)
            slam_yaw = np.mean(self.particle_yaw_list[-1], axis=0)

            slam_mean_xy = slam_xy
            slam_mean_yaw = slam_yaw
            slam_mean2_xy = slam_xy
            slam_mean2_yaw = slam_yaw
            slam_ml_xy = slam_xy
            slam_ml_yaw = slam_yaw

            slam_traj_xy = None
            slam_traj_yaw = None

            # TODO we should predict global map as well with a single local map added to it

        else:
            assert len(self.action_traj) > 0
            assert len(self.particle_xy_list) == len(self.action_traj)
            assert self.visibility_traj[-1].dtype == np.float32
            assert np.all(self.visibility_traj[-1] <= 1.)
            inference_trajlen = self.params.inference_trajlen

            self.image_traj.append(images.copy())
            self.visibility_traj.append(visibility_map.copy())
            self.true_xy_traj.append(true_xy)
            self.true_yaw_traj.append(true_yaw)

            new_action = np.array((self.action_traj[-1], ), np.int32)[None]
            new_rel_xy, new_rel_yaw = actions_from_trajectory(
                np.stack([self.true_xy_traj[-2], self.true_xy_traj[-1]], axis=0), np.stack([self.true_yaw_traj[-2], self.true_yaw_traj[-1]], axis=0))

            # Pick best segment of the trajectory based on how much viewing areas overlap
            current_trajlen = len(self.particle_xy_list) + 1
            assert len(self.true_xy_traj) == current_trajlen and len(self.image_features_traj) == current_trajlen - 1
            if self.params.slam_use_best_steps:
                mean_traj_xy, mean_traj_yaw = ClassicMapping.mean_particle_traj(
                    np.array(self.particle_xy_list), np.array(self.particle_yaw_list), self.particle_logit_acc_list[-1][None, :, None])
                mean_traj_xy, mean_traj_yaw = ClassicMapping.propage_trajectory_with_action(mean_traj_xy, mean_traj_yaw, self.action_traj[-1])

                segment_steps = ClassicMapping.get_steps_with_largest_overlapping_view(
                    mean_traj_xy, mean_traj_yaw, segment_len=inference_trajlen, view_distance=30*self.brain_requirements.transform_window_scaler)

                # print ("Segment: ", segment_steps)
            else:
                segment_steps = np.arange(max(current_trajlen-inference_trajlen, 0), current_trajlen)
            assert segment_steps.ndim == 1

            # particle_xy_seg = np.stack(self.particle_xy_list[-(inference_trajlen - 1):], axis=0)
            # particle_yaw_seg = np.stack(self.particle_yaw_list[-(inference_trajlen - 1):], axis=0)
            # images_seg = np.stack(self.image_traj[-(inference_trajlen):], axis=0)
            # visibility_seg = np.stack(self.visibility_traj[-(inference_trajlen):], axis=0)
            past_particle_xy = np.stack(self.particle_xy_list, axis=0)
            past_particle_yaw = np.stack(self.particle_yaw_list, axis=0)
            true_xy_seg = np.stack([self.true_xy_traj[i] for i in segment_steps], axis=0) + self.true_xy_offset[None]
            true_yaw_seg = np.stack([self.true_yaw_traj[i] for i in segment_steps], axis=0)
            past_image_features_seg = np.stack([self.image_features_traj[i] for i in segment_steps[:-1]], axis=0)

            past_local_maps = np.stack(self.local_map_traj, axis=0)
            past_visibility = np.stack(self.visibility_traj[:-1], axis=0)

            feed_dict = {
                self.inference_timesteps_input: segment_steps[None],
                self.new_images_input: images[None, None],
                self.last_images_input: self.image_traj[-2][None, None],
                self.visibility_input: visibility_map[None, None],
                self.past_local_maps_input: past_local_maps[None],
                self.past_visibility_input: past_visibility[None],
                self.past_needed_image_features_input: past_image_features_seg[None],
                self.global_map_shape_input: np.array(map_shape[:2], np.int32),
                # global_map_input: global_map,
                # self.images_input: images_seg[None],  # always input both images and global map, only one will be connected
                self.true_xy_input: true_xy_seg[None],  # used for global to local transition and loss
                self.true_yaw_input: true_yaw_seg[None],
                # self.visibility_input: visibility_seg[None],
                # self.particle_xy_input: particle_xy_seg[None],
                # self.particle_yaw_input: particle_yaw_seg[None],
                self.particle_xy_input: past_particle_xy[None],
                self.particle_yaw_input: past_particle_yaw[None],
                self.new_action_input: new_action[None],
                self.new_rel_xy_input: new_rel_xy[None],
                self.new_rel_yaw_input: new_rel_yaw[None],
                self.last_step_particle_logits_input: self.particle_logit_acc_list[-1][None],
            }

            slam_outputs = self.run_inference(feed_dict, need_map=need_map)

            # Deal with resampling
            self.particle_xy_list = [particle[slam_outputs.particle_indices[0]] for particle in self.particle_xy_list]
            self.particle_yaw_list = [particle[slam_outputs.particle_indices[0]] for particle in self.particle_yaw_list]
            self.particle_logit_acc_list = [particle[slam_outputs.particle_indices[0]] for particle in self.particle_logit_acc_list]

            # Store new particles
            self.particle_xy_list.append(slam_outputs.particle_xy_t[0])
            self.particle_yaw_list.append(slam_outputs.particle_yaw_t[0])
            self.particle_logit_acc_list.append(slam_outputs.particle_logits_acc[0])

            if FAKE_INPUT_FOR_SPEED_TEST:
                self.particle_xy_list[-1] = self.particle_xy_list[-1] * 0 + true_xy[None] + self.true_xy_offset[None]

            # Store local map prediction
            self.local_map_traj.append(slam_outputs.new_local_maps[0, 0])
            self.image_features_traj.append(slam_outputs.new_image_features[0, 0])
            print (self.image_features_traj[-1].shape)

            # Store losses. only meaningful if true state was input
            self.xy_loss_list.append(slam_outputs.loss_xy_all[0])
            self.yaw_loss_list.append(slam_outputs.loss_yaw_all[0])

            # Update map
            if need_map:
                global_map_logodds = np.array(slam_outputs.global_map_logodds[0])  # squeeze batch and traj
                # if global_map_logodds.shape != self.global_map_logodds.shape:
                    # raise ValueError("Unexpected global map shape output from slam net.")
                if not self.fixed_map_size:
                    global_map_logodds = global_map_logodds[:map_shape[0], :map_shape[1]]
                self.global_map_logodds = global_map_logodds

            # Transform predictions
            global_map_true_partial = None
            assert self.global_map_logodds.shape[-1] == 1
            global_map_pred = ClassicMapping.inverse_logodds(self.global_map_logodds)

            slam_mean_xy = slam_outputs.mean_xy[0, -1]
            slam_mean_yaw = slam_outputs.mean_yaw[0, -1]
            slam_mean2_xy = slam_outputs.mean2_xy[0, -1]
            slam_mean2_yaw = slam_outputs.mean2_yaw[0, -1]
            slam_ml_xy = slam_outputs.ml_xy[0, -1]
            slam_ml_yaw = slam_outputs.ml_yaw[0, -1]

            slam_traj_xy = slam_outputs.xy[0, :]  # the one used for mapping
            slam_traj_yaw = slam_outputs.yaw[0, :]  # the one used for mapping

            slam_xy = slam_outputs.xy[0, -1]  # the one used for mapping
            slam_yaw = slam_outputs.yaw[0, -1]


        # TODO should separate reassemble the map for the whole trajectory for the mean particle trajectory
        #  do NOT use most likely particle, its meaningless after resampling. Density is what matters.
        #  need to implement reasonable sequential averaging of yaws..

        # Compute mean separately here
        if self.params.brain == 'habslambrain_v1' and USE_ASSERTS:
            mean_xy_from_np, mean_yaw_from_np = ClassicMapping.mean_particle_traj(self.particle_xy_list[-1], self.particle_yaw_list[-1], self.particle_logit_acc_list[-1][:, None])

            xy_diff = np.abs(mean_xy_from_np - slam_mean_xy)
            yaw_diff = np.abs(mean_yaw_from_np - slam_mean_yaw)
            yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
            if not np.all(xy_diff < 1.) or not np.all(yaw_diff < np.deg2rad(10.)):
                raise ValueError("SLAM mean and numpy mean dont match. Mean difference: %s vs %s | %s vs. %s" % (
                    str(mean_xy_from_np), str(slam_mean_xy), str(mean_yaw_from_np), str(slam_mean_yaw)))

        # Pose source
        if self.pose_source == 'true':
            xy = true_xy + self.true_xy_offset
            yaw = true_yaw
            traj_xy = np.array(self.true_xy_traj) + self.true_xy_offset[None]
            traj_yaw = np.array(self.true_yaw_traj)
            assert slam_traj_xy is None or traj_xy.shape[0] == slam_traj_xy.shape[0]
        elif self.pose_source in ["slam-truestart", "slam"]:
            xy = slam_xy
            yaw = slam_yaw
            traj_xy = slam_traj_xy
            traj_yaw = slam_traj_yaw
            # TODO weighted mean of particles
        else:
            raise NotImplementedError
        self.xy = xy
        self.yaw = yaw

        local_map_label = None
        # local_map_label = slam_outputs.local_map_label[0, 0, :, :, 0]
        # local_map_pred = slam_outputs.combined_local_map_pred[0, 0, :, :, 0]

        ang_vel = yaw - self.prev_yaw
        ang_vel = (ang_vel + np.pi) % (2*np.pi) - np.pi

        target_dist = np.linalg.norm(self.target_xy - xy)
        true_target_dist = np.linalg.norm(true_target_xy - true_xy)
        xy_error, yaw_error = self.pose_error(slam_xy, slam_yaw, true_xy, true_yaw)
        mean_xy_error, mean_yaw_error = self.pose_error(slam_mean_xy, slam_mean_yaw, true_xy, true_yaw)
        mean2_xy_error, _ = self.pose_error(slam_mean2_xy, slam_mean2_yaw, true_xy, true_yaw)
        ml_xy_error, _ = self.pose_error(slam_ml_xy, slam_ml_yaw, true_xy, true_yaw)
        self.distance_history.append(target_dist)

        if self.pose_source != 'slam' and not FAKE_INPUT_FOR_SPEED_TEST:
            assert np.abs(np.sqrt(self.xy_loss_list[-1]) - xy_error) < 2.  # one is before resampling, other is after

        # Detect collision
        is_colliding = False
        if self.step_i > 2 and self.action_traj[-1] == 1 and self.recover_step_i == 0:  # moved forward
            last_step_len = np.linalg.norm(traj_xy[-2] - traj_xy[-1], axis=0)
            if last_step_len < COLLISION_DISTANCE_THRESHOLD:
                is_colliding = True
                self.collision_timesteps.append(self.step_i)
                self.num_collisions += 1
        if self.recover_step_i >= len(self.recover_policy):
            self.recover_step_i = 0  # done with recovery

        dist_hist = np.array(self.distance_history[-GIVE_UP_NO_PROGRESS_STEPS:])

        # Give up becuase too far. No longer needed as we move the
        should_give_up = (np.any(self.target_xy_for_planning < TARGET_MAP_MARGIN) or
                          np.any(self.target_xy_for_planning + TARGET_MAP_MARGIN >= np.array(self.max_map_size)) or
                          np.any(self.xy < 0) or
                          np.any(self.xy >= np.array(self.max_map_size)))
        if USE_ASSERTS and should_give_up and self.fixed_map_size:
            raise ValueError("Target or state is outside of map area -- this should not happen for fixed size map.")

        try:
            for time_thres, dist_thres in GIVE_UP_STEP_AND_DISTANCE:
                if self.step_i >= time_thres and target_dist >= dist_thres:
                    should_give_up = True
                    break
        except Exception as e:
            print ("Exception " + str(e))
        # Give up if no progress for too long wallclock time
        try:
            mins_since_ep_start = (time.time() - self.episode_t) / 60
            reduction_since_beginning = self.distance_history[0] - self.distance_history[-1]
            for time_thres, reduct_thres in GIVE_UP_TIME_AND_REDUCTION:
                if mins_since_ep_start >= time_thres and reduction_since_beginning < reduct_thres:
                    print ("Give up because of wallclock time and reduction t=%f reduct=%f"%(mins_since_ep_start, reduction_since_beginning))
                    should_give_up = True
                    break
        except Exception as e:
            print ("Exception " + str(e))

        giving_up_collision = False
        giving_up_distance = False
        giving_up_progress = False
        is_done = False

        # Plan
        planned_path = np.zeros([0, 2], dtype=np.float32)
        # Choose which map to use for planning
        global_map_for_planning = self.get_global_map_for_planning(global_map_pred, global_map_label, traj_xy, traj_yaw, map_shape, self.map_source, keep_soft=self.params.soft_cost_map)
        processed_map_for_planning = global_map_for_planning
        if MANUAL_STOP_WHEN_NEAR_TARGET and target_dist < 3.:
            # Close enough to target. Normal requirement is 0.36/0.05 = 7.2
            plan_status_msg = "Manual stop"
            is_done = True
            action = 0

        elif should_give_up:
            plan_status_msg = "Giving up because target is too far .."
            giving_up_distance = True
            action = 0

        elif shortcut_action is not None:
            # NOTE must be before recover on collision - because we already incremented recover policy
            plan_status_msg = "Shortcut action"
            action = shortcut_action

        elif RECOVER_ON_COLLISION and (is_colliding or self.recover_step_i > 0):
            plan_status_msg = ("Recover from collision %d / %d."%(self.recover_step_i, len(self.recover_policy)))
            action = self.recover_policy[self.recover_step_i]
            self.recover_step_i += 1
            self.pathplanner.reset()  # to clear out its cache
            if target_dist < NEAR_TARGET_COLLISION_STOP_DISTANCE:
                plan_status_msg += " --> Attempt to stop instead, near target"
                is_done = True
                action = 0

        elif GIVE_UP_NUM_COLLISIONS > 0 and self.num_collisions >= GIVE_UP_NUM_COLLISIONS:
            plan_status_msg = "Too many collisions (%d). Giving up.."%(self.num_collisions, )
            giving_up_collision = True
            action = 0

        elif GIVE_UP_NO_PROGRESS_STEPS > 0 and self.step_i > GIVE_UP_NO_PROGRESS_STEPS and self.step_i > 100 and np.max(dist_hist) - np.min(dist_hist) < NO_PROGRESS_THRESHOLD:
            plan_status_msg = "No progress for %d steps. Giving up.."%(GIVE_UP_NO_PROGRESS_STEPS, )
            giving_up_progress = True
            action = 0

        else:
            action, planned_path, plan_status_msg, processed_map_for_planning = self.plan_and_control(
                xy, yaw, self.target_xy_for_planning, global_map_for_planning, ang_vel, initial_target_fi,
                allow_shrink_map=self.fixed_map_size and ALLOW_SHRINK_MAP)
            is_done = (action == 0)

            # Visualize agent
            if self.step_i % PLOT_EVERY_N_STEP == 0 and PLOT_EVERY_N_STEP > 0 and slam_outputs is not None:
                local_map_pred = self.local_map_traj[-1][:, :, 0]
                self.visualize_agent(slam_outputs.tiled_visibility_mask[0, 0, :, :, 0], images, global_map_pred,
                                     global_map_for_planning,
                                     # processed_map_for_planning.astype(np.float32)/255.,
                                     global_map_label,
                                     global_map_true_partial, local_map_pred, local_map_label, planned_path,
                                     sim_rgb=observations['rgb'],
                                     xy=xy, yaw=yaw, true_xy=true_xy + self.true_xy_offset, true_yaw=true_yaw, target_xy=self.target_xy_for_planning)
            # pdb.set_trace()

        # Overwrite with expert
        if self.action_source == 'expert':
            best_action = self.follower.get_next_action(goal_pos)
            action = best_action

            if action == 0 and EXIT_AFTER_N_STEPS_FOR_SPEED_TEST > 0:
                print ("Sping instead of stopping.")
                action = 3
            is_done = (action == 0)

        # Save data
        if len(self.tfwriters) > 0:
            if planned_path.shape[0] == 0 or target_dist < 10.5:  # two step strategy for <= 10
                if DATA_INCLUDE_NONPLANNED_ACTIONS:
                    raise NotImplementedError
            else:
                assert self.map_source != "pred"
                pred_map_for_planning = self.get_global_map_for_planning(global_map_pred, global_map_label, traj_xy,
                                                                           traj_yaw, map_shape, "pred", keep_soft=True)

                self.write_datapoint(global_map_for_planning, pred_map_for_planning, self.target_xy_for_planning, planned_path.astype(np.int32), action)

        # pdb.set_trace()
        # if self.episode_i == 0:
        #     cv2.imwrite('./temp/ep%d-step%d.png'%(self.episode_i, self.step_i), observations['rgb'])
        # if self.step_i == 0:
        #     top_down_map = maps.get_topdown_map(
        #         self.env.sim, map_resolution=(5000, 5000)
        #     )
        #     plt.imshow(top_down_map)
        #     plt.show()

        self.prev_yaw = yaw
        self.action_traj.append(action)
        self.step_i += 1

        slam_status_msg = "Pose errors mean=%.1f mean2=%.1f ml=%.1f yaw=%.1f. Loss=%.1f "%(
            mean_xy_error, mean2_xy_error, ml_xy_error, np.rad2deg(mean_yaw_error), np.sqrt(self.xy_loss_list[-1]))
        act_status_msg = "Est dist=%.1f. True dist=%.1f Act=%d %s"%(
            target_dist, true_target_dist, action, "COL" if is_colliding else "")
        print (plan_status_msg)
        print (slam_status_msg + act_status_msg)

        # Get map statistics
        if global_map_label is not None:
            ij = xy.astype(np.int32)
            self.num_wrong_obstacle += 1. if not is_colliding and global_map_label[ij[0], ij[1]] < 0.5 else 0.
            self.num_wrong_free += 1. if is_colliding and global_map_label[ij[0], ij[1]] >= 0.5 else 0.
            self.num_wrong_free_area += 1. if is_colliding and np.all(global_map_label[max(ij[0]-1, 0):ij[0]+2, max(ij[1]-1, 0):ij[1]+2] >= 0.5) else 0.
            self.num_wrong_free_area2 += 1. if is_colliding and np.all(global_map_label[max(ij[0]-2, 0):ij[0]+3, max(ij[1]-2, 0):ij[1]+3] >= 0.5) else 0.
            self.num_wrong_free_area3 += 1. if is_colliding and np.all(global_map_label[max(ij[0]-3, 0):ij[0]+4, max(ij[1]-3, 0):ij[1]+4] >= 0.5) else 0.

        if self.params.save_video > self.num_videos_saved:
            if not isinstance(planned_path, np.ndarray) or planned_path.ndim != 2:
                print ("planned path has an unexpected format")
                pdb.set_trace()
            self.frame_traj_data.append(dict(
                rgb=observations['rgb'], global_map=global_map_pred.copy(),
                global_map_for_planning=global_map_for_planning.copy(),
                true_global_map=global_map_label.copy(),
                xy=self.xy.copy(), yaw=self.yaw.copy(),
                target_xy=self.target_xy_for_planning.copy(),
                path=planned_path.copy(), # subgoal=planned_subgoal.copy(),
                target_status=slam_status_msg, control_status=plan_status_msg, act_status=act_status_msg))

        return {"action": action, "has_collided": float(self.num_collisions > 0), "num_collisions": self.num_collisions,
                "xy_error": xy_error, "mean_xy_error": mean_xy_error, "mean2_xy_error": mean2_xy_error, "ml_xy_error": ml_xy_error,
                'mean_yaw_error': mean_yaw_error, 'target_dist': target_dist,
                'num_wrong_obstacle': self.num_wrong_obstacle, 'num_wrong_free': self.num_wrong_free,
                'num_wrong_free_area': self.num_wrong_free_area,
                'num_wrong_free_area2': self.num_wrong_free_area2, 'num_wrong_free_area3': self.num_wrong_free_area3,
                'time_plan': 0. if len(self.plan_times) == 0 else np.mean(self.plan_times),
                'map_mismatch_count': float(self.map_mismatch_count) / self.step_i,  # TODO remove
                'giveup_collision': float(giving_up_collision), 'giveup_progress': float(giving_up_progress),
                'giveup_distance': float(giving_up_distance), 'is_done: ': is_done}   # 0: stop, forward, left, right
        # return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}

    def write_datapoint(self, map_for_planning, pred_map, target_xy, planned_path, action):
        assert planned_path.shape[0] > 0
        assert planned_path.dtype == np.int32
        planned_actions = grid_actions_from_trajectory(planned_path, connect8=False)
        target_ij = target_xy.astype(np.int32)

        if planned_path.shape[0] < self.params.trainlen:
            return
        if planned_path[-1][0] != target_ij[0] or planned_path[-1][1] != target_ij[1]:
            print ("Skip because path does not reach goal")
            print (planned_path)
            print (target_ij)
            return

        # convert maps
        assert map_for_planning.dtype == np.float32 and pred_map.dtype == np.float32
        assert map_for_planning.shape == pred_map.shape
        map_for_planning = (map_for_planning * 255.).astype(np.uint8)
        pred_map = (pred_map * 255.).astype(np.uint8)  # encode predicted probability as uint8

        assert len(self.tfwriters) == len(self.params.data_map_sizes)
        for tfwriter, map_size in zip(self.tfwriters, self.params.data_map_sizes):

            self.write_data_for_map_size(map_for_planning, pred_map, planned_actions, planned_path, target_xy, tfwriter, map_size)

    def write_data_for_map_size(self, map_for_planning, pred_map, planned_actions, planned_path, target_xy, tfwriter, map_size):
        segment_len = self.params.trainlen

        if map_size < map_for_planning.shape[0]:
            # Find last trajectory segment that is still within the map size
            margin = 2
            for start_i in range(len(planned_path)):
                range_ij = np.max(planned_path[start_i:], axis=0) - np.min(planned_path[start_i:], axis=0)
                if np.all(range_ij < map_size - 2 * margin):
                    break
            planned_path = planned_path[start_i:]
            planned_actions = planned_actions[start_i:]

            # Crop map
            offset_ij = np.min(planned_path, axis=0)
            range_ij =  np.max(planned_path, axis=0) - offset_ij
            # add half of the remaining spacing to the beginning
            topleft_space = (map_size - range_ij) // 2
            offset_ij = offset_ij - topleft_space
            offset_ij = np.maximum(offset_ij, np.zeros((2, ), np.int32))  # cannot be less than zero
            offset_ij = np.minimum(offset_ij, np.array(map_for_planning.shape[:2], np.int32) - map_size)
            # crop the given size starting from offset_ij
            map_for_planning = map_for_planning[offset_ij[0]:offset_ij[0]+map_size, offset_ij[1]:offset_ij[1]+map_size]
            pred_map = pred_map[offset_ij[0]:offset_ij[0]+map_size, offset_ij[1]:offset_ij[1]+map_size]

            # Move poses to cropped frame
            planned_path = planned_path - offset_ij[None]
            target_xy = target_xy - offset_ij.astype(np.float32)

        if planned_path.shape[0] < self.params.trainlen:
            return
        assert map_for_planning.shape[0] == map_size and map_for_planning.shape[1] == map_size
        assert np.all(planned_path >= 0) and np.all(planned_path < map_size)

        planned_xy = planned_path.astype(np.float32) + 0.5

        true_map_png = cv2.imencode('.png', map_for_planning)[1].tobytes()
        pred_map_png = cv2.imencode('.png', pred_map)[1].tobytes()
        # segments
        traj_segments = []
        overlap = 1  # use extra step because last action will be dropped
        assert overlap < segment_len
        start_i = 0
        while start_i < planned_path.shape[0]:  # include all steps
            # for incomplete last segment, start earlier overlapping with previous segment
            if start_i + segment_len > planned_path.shape[0]:
                start_i = planned_path.shape[0] - segment_len
                overlap = -1  # this is to triger break at the end of this iteration
            segment = tuple(range(start_i, start_i + segment_len))
            traj_segments.append(segment)
            start_i += segment_len - overlap
        del overlap

        # store each segment
        goal_xy = target_xy.copy()
        for segment in traj_segments:  # repeat multiple times
            xy_segment = planned_xy[segment, :]
            grid_action_segment = planned_actions[segment[:-1],]

            # dummy yaw and action
            yaw_segment = np.ones((segment_len, 1), np.float32) * -1
            action_segment = np.ones((segment_len - 1, 1), np.int32) * -1

            # tfrecord features
            context_features = {
                'true_map': tf_bytes_feature(true_map_png),
                'pred_map': tf_bytes_feature(pred_map_png),
                'trajlen': tf_int64_feature(len(segment)),
                'goal_xy': tf_bytes_feature(goal_xy.astype(np.float32).tobytes()),
                'xy': tf_bytes_feature(xy_segment.astype(np.float32).tobytes()),
                'yaw': tf_bytes_feature(yaw_segment.astype(np.float32).tobytes()),
                'action': tf_bytes_feature(action_segment.astype(np.int32).tobytes()),
                # 'grid_q_values': tf_bytelist_feature(q_segment.tobytes()),
                'grid_action': tf_bytes_feature(grid_action_segment.astype(np.int32).tobytes()),
            }
            sequence_features = {
                # 'local_map': tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[local_map_pngs[i]])) for i in segment]),
                # 'visibility': tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[visibility_map_pngs[i]])) for i in segment]),
            }

            # store
            example = tf.train.SequenceExample(context=tf.train.Features(feature=context_features),
                                               feature_lists=tf.train.FeatureLists(feature_list=sequence_features))
            tfwriter.write(example.SerializeToString())
            self.num_data_entries += 1

    def get_global_map_for_planning(self, global_map_pred, global_map_label, traj_xy, traj_yaw, map_shape, map_source, keep_soft):
        if map_source in ['true', 'true-saved']:
            assert global_map_label.ndim == 3
            global_map_for_planning = global_map_label.copy()
            assert global_map_for_planning.shape == map_shape

        elif map_source in ['true-partial']:
            global_map_for_planning = global_map_label.copy()
            # Overwrite unseen areas with 0.5
            unseen_mask = np.isclose(global_map_pred, 0.5)
            global_map_for_planning[unseen_mask] = 0.5

        else:
            global_map_for_planning = global_map_pred.copy()

        if self.params.collision_patch_radius > 0 and self.step_i > 1:
            global_map_for_planning = self.patch_map_with_collisions(global_map_for_planning, traj_xy,
                                                                     traj_yaw, self.collision_timesteps,
                                                                     self.params.collision_patch_radius)

        if CLEAR_TARGET_RADIUS > 0:
            try:
                min_xy = self.target_xy_for_planning.astype(np.int32) - CLEAR_TARGET_RADIUS
                max_xy = self.target_xy_for_planning.astype(np.int32) + CLEAR_TARGET_RADIUS + 1

                global_map_for_planning[min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]] = 1.
            except Exception as e:
                print ("Exception clearing target. " + str(e))
                raise e
        #
        # if self.step_i == 1:
        #     print ("DEBUG !!!!!!! REMOVE !!!!!!!")
        #     self.collision_timesteps.append(1)
        # threshold
        if not keep_soft:
            traversable_threshold = self.params.traversable_threshold  # higher than this is traversable
            object_treshold = 0.  # treat everything as non-object
            threshold_const = np.array((traversable_threshold, object_treshold))[None, None, :self.map_ch - 1]
            global_map_for_planning = np.array(global_map_for_planning >= threshold_const, np.float32)
        return global_map_for_planning

    @staticmethod
    def shrink_map(xy, target_xy, global_map, margin=8):
        assert margin > 6  # one step in each direction requires at least 6 margin
        assert global_map.dtype == np.uint8

        free_i, free_j, _  = np.nonzero(global_map == 0)
        min_i = min(int(xy[0]), int(target_xy[0]), np.min(free_i)) - margin
        min_j = min(int(xy[1]), int(target_xy[1]), np.min(free_j)) - margin
        max_i = max(int(xy[0]), int(target_xy[0]), np.max(free_i)) + margin + 1
        max_j = max(int(xy[1]), int(target_xy[1]), np.max(free_j)) + margin + 1

        min_i = max(min_i, 0)
        min_j = max(min_j, 0)
        max_i = min(max_i, global_map.shape[0])
        max_j = min(max_j, global_map.shape[1])

        offset_xy = np.array([min_i, min_j], np.float32)

        if min_i > 0 or min_j > 0 or max_i < global_map.shape[0] or max_j < global_map.shape[1]:
            global_map = global_map[min_i:max_i, min_j:max_j]
            xy = xy - offset_xy
            target_xy = target_xy - offset_xy

        return xy, target_xy, global_map, offset_xy

    @staticmethod
    def patch_map_with_collisions(global_map_for_planning, traj_xy, traj_yaw, collision_timesteps, patch_radius):
        for timestep in collision_timesteps:
            xy = traj_xy[timestep]
            yaw = traj_yaw[timestep]

            if patch_radius > 0.5:
                num_samples = max(int(2 * patch_radius), 6)
                ego_x, ego_y = np.meshgrid(
                    np.linspace(0, 2 * patch_radius, num_samples) - 0.4,
                    np.linspace(-patch_radius, patch_radius, num_samples),
                    indexing='ij')
                ego_xy = np.stack((ego_x.flatten(), ego_y.flatten()), axis=-1)
                abs_xy = xy[None] + rotate_2d(ego_xy, yaw[None])
                abs_ij = abs_xy.astype(np.int32)
            else:
                abs_ij = xy[None].astype(np.int32)
            # Filter out of range
            abs_ij = abs_ij[np.logical_and.reduce([
                abs_ij[:, 0] >= 0, abs_ij[:, 1] >= 0, abs_ij[:, 0] < global_map_for_planning.shape[0],
                abs_ij[:, 1] < global_map_for_planning.shape[1] ])]
            # Set map not traversable (0.)
            global_map_for_planning[abs_ij[:, 0], abs_ij[:, 1]] = 0.
        return global_map_for_planning

    def pose_error(self, slam_xy, slam_yaw, true_xy, true_yaw):
        xy_error = np.linalg.norm(true_xy + self.true_xy_offset - slam_xy)
        yaw_error = true_yaw - slam_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        return xy_error, yaw_error

    def run_inference(self, feed_dict, need_map=True):
        outputs = self.sess.run((self.inference_outputs if need_map else self.inference_outputs_without_map), feed_dict=feed_dict)
        return outputs


    def video_update(self, frame_i):
        # frame skip of 3
        if frame_i % VIDEO_FRAME_SKIP == 0:
            ind = min(frame_i // VIDEO_FRAME_SKIP, len(self.frame_traj_data)-1)
            self.video_image_ax.set_data(self.frame_traj_data[ind]['rgb'])
            self.video_text_ax1.set_text(self.frame_traj_data[ind]['target_status'])
            split_str = self.frame_traj_data[ind]['control_status'] + " " + self.frame_traj_data[ind]['act_status']
            # Attempt to break lines
            segs = split_str.split("[")
            if len(segs) > 1:
                split_str = segs[0] + "\n["+"[".join(segs[1:])
            segs = split_str.split(" v=")
            if len(segs) > 1:
                split_str = segs[0] + "\nv=" + " v=".join(segs[1:])
            self.video_text_ax2.set_text(split_str)

            if self.video_global_map_ax is not None:
                xy = self.frame_traj_data[ind]['xy']
                target_xy = self.frame_traj_data[ind]['target_xy']
                #subgoal = self.frame_traj_data[ind]['subgoal']
                path = self.frame_traj_data[ind]['path'].copy()
                if len(path) == 0:
                    path = xy[None]
                path = np.array(path)[:, :2]

                global_map = np.atleast_3d(self.frame_traj_data[ind]['global_map'])
                global_map = np.tile(global_map[:, :, :1], [1, 1, 3])
                true_map = np.atleast_3d(self.frame_traj_data[ind]['true_global_map'])
                true_map = np.tile(true_map[:, :, :1], [1, 1, 3])
                map_for_planning = np.atleast_3d(self.frame_traj_data[ind]['global_map_for_planning'])
                map_for_planning = np.tile(map_for_planning[:, :, :1], [1, 1, 3])

                # Follow agent with a window
                window_size = 220
                map_for_planning_crop, path_crop, target_xy_crop, xy_crop = self.crop_experience_window(
                    window_size, map_for_planning, path, target_xy, xy)

                if self.fixed_map_size:
                    # Use a fixed global view
                    combined_map2 = global_map
                    combined_map3 = true_map
                else:
                    global_map_crop, _, _, temp_xy_crop = self.crop_experience_window(window_size, global_map, path, target_xy, xy)
                    assert np.all(temp_xy_crop == xy_crop)
                    combined_map2 = global_map_crop

                    true_map_crop,  _, _, temp_xy_crop  = self.crop_experience_window(window_size, true_map, path, target_xy, xy)
                    assert np.all(temp_xy_crop == xy_crop)
                    combined_map3 = true_map_crop

                    xy = xy_crop
                    target_xy = target_xy_crop
                    path = path_crop
                planned_path_skip = 4

                combined_map = map_for_planning_crop  # global_map_crop if MAP_SOURCE == 'pred' else true_map_crop

                # global_map = global_map[:map_size-map_offset_xy[0], :map_size-map_offset_xy[1]]
                # combined_map[map_offset_xy[0]:map_offset_xy[0]+global_map.shape[0], map_offset_xy[1]:map_offset_xy[1]+global_map.shape[1]] = global_map

                combined_map[int(xy_crop[0])-1:int(xy_crop[0])+2, int(xy_crop[1])-1:int(xy_crop[1]+2)] = (1., 0., 1.)
                combined_map2[int(xy[0])-1:int(xy[0])+2, int(xy[1])-1:int(xy[1]+2)] = (1., 0., 1.)
                combined_map3[int(xy[0])-1:int(xy[0])+2, int(xy[1])-1:int(xy[1]+2)] = (1., 0., 1.)

                # print (self.video_ax.get_xlim())
                self.video_ax.set_xlim(-0.5, combined_map.shape[1]-0.5)
                self.video_ax.set_ylim(combined_map.shape[0]-0.5, -0.5)
                self.video_global_map_ax.set_data(combined_map)
                self.video_global_map_ax.set_extent([-0.5, combined_map.shape[1]-0.5, combined_map.shape[0]-0.5, -0.5])
                self.video_path_scatter.set_offsets(np.flip(path_crop[planned_path_skip::planned_path_skip], axis=-1))
                self.video_target_scatter.set_offsets([np.flip(target_xy_crop, axis=-1)])

                if VIDEO_LARGE_PLOT:
                    self.video_ax2.set_xlim(-0.5, combined_map2.shape[1]-0.5)
                    self.video_ax2.set_ylim(combined_map2.shape[0]-0.5, -0.5)
                    self.video_global_map_ax2.set_data(combined_map2)
                    self.video_global_map_ax2.set_extent([-0.5, combined_map2.shape[1]-0.5, combined_map2.shape[0]-0.5, -0.5])
                    self.video_path_scatter2.set_offsets(np.flip(path[planned_path_skip::planned_path_skip], axis=-1))
                    self.video_target_scatter2.set_offsets([np.flip(target_xy, axis=-1)])

                    self.video_ax3.set_xlim(-0.5, combined_map3.shape[1]-0.5)
                    self.video_ax3.set_ylim(combined_map3.shape[0]-0.5, -0.5)
                    self.video_global_map_ax3.set_data(combined_map3)
                    self.video_global_map_ax3.set_extent([-0.5, combined_map3.shape[1]-0.5, combined_map3.shape[0]-0.5, -0.5])
                    self.video_path_scatter3.set_offsets(np.flip(path[planned_path_skip::planned_path_skip], axis=-1))
                    self.video_target_scatter3.set_offsets([np.flip(target_xy, axis=-1)])

                # # View angle
                # half_fov = 0.5 * np.deg2rad(70)
                # for ang_i, angle in enumerate([half_fov, -half_fov]):
                #     angle = angle - float(self.frame_traj_data[ind]['yaw'])
                #     # angle = angle + yaw[batch_i, traj_i, 0]
                #     v = np.array([np.cos(angle), np.sin(angle)]) * 10.
                #     x1 = np.array([xy[1], xy[0]])  # need to be flipped for display
                #     x2 = v + x1
                #
                #     self.video_view_angle_lines[ang_i].set_data([x1[0], x2[0]], [x1[1], x2[1]])
                #
                # # pdb.set_trace()

                # Path
                # # print(self.frame_traj_data[ind]['xy'], path[0])
                # for i in range(len(self.video_path_circles)-2):
                #     path_i = min(i * 4, len(path)-1)
                #     xy = path[path_i]
                #     self.video_path_circles[i].center = ([xy[1], xy[0]])
                # # Sub-goal
                # xy = self.frame_traj_data[ind]['subgoal']
                # self.video_path_circles[-2].center = ([xy[1], xy[0]])
                # # Target
                # xy = path[-1]
                # self.video_path_circles[-1].center = ([xy[1], xy[0]])

            # self.video_text_ax2.set_data(self.summary_str)
        return self.video_image_ax

    def crop_experience_window(self, map_size, global_map, path, target_xy, xy):
        # Cut it to fixed size 300 x 300
        center_xy = (xy + target_xy) * 0.5
        desired_center_xy = np.array(map_size, np.float32) * 0.5
        center_xy = center_xy.astype(np.int)
        desired_center_xy = desired_center_xy.astype(np.int)
        offset_xy = (desired_center_xy - center_xy).astype(np.int)
        xy = xy + offset_xy
        target_xy = target_xy + offset_xy
        # subgoal += offset_xy
        path = path + offset_xy[None]
        map_start_xy = np.maximum(center_xy - map_size // 2, 0)
        map_cutoff_xy = -np.minimum(center_xy - map_size // 2, 0)
        global_map = global_map[map_start_xy[0]:map_start_xy[0] + map_size - map_cutoff_xy[0], map_start_xy[1]:map_start_xy[1] + map_size - map_cutoff_xy[1]]
        global_map_crop = np.ones((map_size, map_size, 3), np.float32) * 0.5
        global_map_crop[map_cutoff_xy[0]:map_cutoff_xy[0] + global_map.shape[0], map_cutoff_xy[1]:global_map.shape[1] + map_cutoff_xy[1]] = global_map

        return global_map_crop, path, target_xy, xy

    def reset_video_writer(self):
        if len(self.frame_traj_data) > 0:
            # Save video
            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                self.video_image_ax = ax.imshow(np.zeros((90, 160, 3)))
                self.video_global_map_ax = None
                self.video_text_ax0 = fig.text(0.04, 0.9, self.summary_str, transform=fig.transFigure, fontsize=10, verticalalignment='top')  # bottom left

                self.video_text_ax1 = fig.text(0.96, 0.9, "Target", transform=fig.transFigure, fontsize=10, verticalalignment='top', horizontalalignment='right')
                self.video_text_ax2 = fig.text(0.04, 0.05, "Status2", transform=fig.transFigure, fontsize=10, verticalalignment='bottom', wrap=True)
            else:
                fig = plt.figure(figsize=(6, 9))  # aspect ratio
                ax = fig.add_subplot(221 if VIDEO_LARGE_PLOT else 121)
                # ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.video_image_ax = ax.imshow(np.zeros((90, 160, 3)))

                ax = fig.add_subplot(222 if VIDEO_LARGE_PLOT else 122)
                # ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.video_global_map_ax = ax.imshow(np.zeros((1200, 1200, 3)))
                self.video_ax = ax
                self.video_path_scatter = ax.scatter([0.], [1.], s=2., c='green', marker='o')
                self.video_target_scatter = ax.scatter([0.], [1.], s=2., c='red', marker='o')

                if VIDEO_LARGE_PLOT:
                    ax = fig.add_subplot(223)
                    # ax.set_aspect('equal')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    self.video_global_map_ax2 = ax.imshow(np.zeros((1200, 1200, 3)))
                    self.video_ax2 = ax
                    self.video_path_scatter2 = ax.scatter([0.], [1.], s=2., c='green', marker='o')
                    self.video_target_scatter2 = ax.scatter([0.], [1.], s=2., c='red', marker='o')

                    ax = fig.add_subplot(224)
                    # ax.set_aspect('equal')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    self.video_global_map_ax3 = ax.imshow(np.zeros((1200, 1200, 3)))
                    self.video_ax3 = ax
                    self.video_path_scatter3 = ax.scatter([0.], [1.], s=2., c='green', marker='o')
                    self.video_target_scatter3 = ax.scatter([0.], [1.], s=2., c='red', marker='o')

                # self.video_view_angle_lines = [mlines.Line2D([0., 0.], [10., 10.,], color='green') for _ in range(2)]
                # ax.add_line(self.video_view_angle_lines[0])
                # ax.add_line(self.video_view_angle_lines[1])
                # self.video_path_circles = []
                # for i in range(20):
                #     circle = plt.Circle((0., 0.), 2., color=('red' if i >= 18 else 'orange'), fill=False, transform='data')
                #     ax.add_artist(circle)
                #     self.video_path_circles.append(circle)
                # self.video_view_angle_lines = []
                # for _ in range(2):
                #     self.video_view_angle_lines.extend(ax.plot([0., 1.], [0., 1.], '-', color='blue'))  # plot returns a list of lines


                self.video_text_ax0 = fig.text(0.04, 0.9, self.summary_str, transform=fig.transFigure, fontsize=9,
                                               verticalalignment='top')  # bottom left

                self.video_text_ax1 = fig.text(0.96, 0.9, "Target", transform=fig.transFigure, fontsize=9,
                                               verticalalignment='top', horizontalalignment='right')
                self.video_text_ax2 = fig.text(0.04, 0.05, "Status2", transform=fig.transFigure, fontsize=9,
                                               verticalalignment='bottom', wrap=True)

            # im.set_clim([0, 1])
            fig.set_size_inches([5, 5])
            plt.tight_layout()

            ani = animation.FuncAnimation(fig, self.video_update, len(self.frame_traj_data) * VIDEO_FRAME_SKIP + 21, interval=100)  # time between frames in ms. overwritten by fps below
            writer = animation.writers['ffmpeg'](fps=30)
            video_filename = os.path.join(self.logdir, '%s_rgb_%d%s.mp4'%(self.get_scene_name(), self.episode_i, self.filename_addition))
            ani.save(video_filename, writer=writer, dpi=200)

            print ("Video saved to "+video_filename)
            self.num_videos_saved += 1

        self.frame_traj_data = []

    def visualize_agent(self, visibility_mask, images, global_map_pred, global_map_for_planning, global_map_label,
                        global_map_true_partial, local_map_pred, local_map_label, planned_path, sim_rgb=None,
                        local_obj_map_pred=None, xy=None, yaw=None, true_xy=None, true_yaw=None, target_xy=None):

        # Coordinate systems dont match the ones assumed in these plot functions, but all cancells out except for yaw
        yaw = yaw - np.pi/2
        if true_yaw is not None:
            true_yaw = true_yaw - np.pi/2

        status_msg = "step %d" % (self.step_i,)

        if global_map_label is not None:
            # assert global_map_label.shape[-1] == 3
            global_map_label = np.concatenate(
                [global_map_label, np.zeros_like(global_map_label), np.zeros_like(global_map_label)], axis=-1)
            plt.figure("Global map label")
            plt.imshow(global_map_label)
            plot_viewpoints(xy[0], xy[1], yaw)
            if true_xy is not None and true_yaw is not None:
                plot_viewpoints(true_xy[0], true_xy[1], true_yaw, color='green')
            plot_target_and_path(target_xy=target_xy, path=planned_path, every_n=1)
            plt.title(status_msg)
            plt.savefig('./temp/global-map-label.png')
            plt.figure("Global map (%d)" % self.step_i)

        map_to_plot = global_map_pred[..., :1]
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=target_xy, path=planned_path, every_n=1)
        # plot_target_and_path(target_xy=target_xy_vel, path=np.array(self.hist2)[:, :2])
        plt.title(status_msg)
        plt.savefig('./temp/global-map-pred.png')

        if global_map_pred.shape[-1] == 2:
            map_to_plot = global_map_pred[..., 1:2]
            map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
            plt.imshow(map_to_plot)
            plot_viewpoints(xy[0], xy[1], yaw)
            plot_target_and_path(target_xy=target_xy, path=planned_path, every_n=1)
            plt.title(status_msg)
            plt.savefig('./temp/global-obj-map-pred.png')

        # if global_map_true_partial is not None:
        #     plt.figure("Global map true (%d)" % self.step_i)
        #     map_to_plot = global_map_true_partial
        #     map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        #     plt.imshow(map_to_plot)
        #     plot_viewpoints(xy[0], xy[1], yaw)
        #     plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        #     # plot_target_and_path(target_xy=self.target_xy, path=np.array(self.hist1)[:, :2])
        #     # plot_target_and_path(target_xy=self.target_xy_vel, path=np.array(self.hist2)[:, :2])
        #     plt.title(status_msg)
        #     plt.savefig('./temp/global-map-true.png')
        #     plt.figure("Global map plan (%d)" % self.step_i)

        map_to_plot = global_map_for_planning
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=target_xy, path=planned_path, every_n=1)
        plt.title(status_msg)
        plt.savefig('./temp/global-map-plan.png')

        depth, rgb = mapping_visualizer.recover_depth_and_rgb(images)
        if self.params.mode == 'depth' and sim_rgb is not None:
            rgb = sim_rgb
            rgb[:5, :5, :] = 0  # indicate this is not observed

        images_fig, images_axarr = plt.subplots(2, 2, squeeze=True)
        plt.title(status_msg)
        plt.axes(images_axarr[0, 0])
        plt.imshow(depth)
        plt.axes(images_axarr[0, 1])
        plt.imshow(rgb)
        plt.axes(images_axarr[1, 0])
        if local_map_pred is not None:
            plt.imshow(local_map_pred * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0., vmax=1.)
        plt.axes(images_axarr[1, 1])
        if local_obj_map_pred is not None:
            plt.imshow(local_obj_map_pred * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0, vmax=1.)
        elif local_map_label is not None:
            plt.imshow(local_map_label * visibility_mask + (1 - visibility_mask) * 0.5, vmin=0., vmax=1.)
        plt.savefig('./temp/inputs.png')
        #
        #
        # pdb.set_trace()

        plt.close('all')


#
# def main():
#     params = parse_args(default_files=('./gibson_submission.conf', ))
#     is_submission = (params.gibson_mode == 'submission')
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
#     args = parser.parse_args()
#
#     config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
#     config = habitat.get_config(config_paths)
#
#     # agent = RandomAgent(task_config=config)
#
#     if args.evaluation == "local":
#         challenge = habitat.Challenge(eval_remote=False)
#     else:
#         challenge = habitat.Challenge(eval_remote=True)
#
#     env = challenge._env
#     agent = DSLAMAgent(task_config=config, env=env)
#
#     challenge.submit(agent)
#
#
# if __name__ == "__main__":
#     main()

