import argparse
import habitat
import random
import numpy as np
import os
import cv2
import time
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from gibsonagents.expert import Expert
from gibsonagents.classic_mapping import rotate_2d, ClassicMapping

from arguments import parse_args

import tensorflow as tf

from train import get_brain, get_tf_config
from common_net import load_from_file, count_number_trainable_params
from visualize_mapping import plot_viewpoints, plot_target_and_path, mapping_visualizer

import matplotlib.pyplot as plt
try:
    import ipdb as pdb
except:
    import pdb


POSE_ESTIMATION_SOURCE = "true"
MAP_SOURCE = "pred"
ACTION_SOURCE = "plan"  # expert
START_WITH_SPIN = True
PLOT_EVERY_N_STEP = 1


class DSLAMAgent(habitat.Agent):
    def __init__(self, task_config, params, env=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.step_i = 0
        self.episode_i = -2
        self.env = env
        self.task_config = task_config
        self.follower = ShortestPathFollower(env._sim, 0.36/2., False)

        self.top_down_map_resolution = self.task_config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION

        print (params)

        self.params = params
        self.pose_estimation_source = POSE_ESTIMATION_SOURCE
        self.map_source = MAP_SOURCE
        self.action_source = ACTION_SOURCE
        self.start_with_spin = START_WITH_SPIN
        self.max_confidence = 0.96   # 0.98
        self.confidence_threshold = None  # (0.2, 0.01)  # (0.35, 0.05)
        self.use_custom_visibility = (self.params.visibility_mask == 2)

        self.map_ch = 2
        self.accumulated_spin = 0.
        self.spin_direction = None

        params.batchdata = 1
        params.trajlen = 1
        sensor_ch = (1 if params.mode == 'depth' else (3 if params.mode == 'rgb' else 4))
        self.max_map_size = (800, 800)

        # Build brain
        with tf.Graph().as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # dataflow input
                train_brain = get_brain(params.brain, params)
                req = train_brain.requirements()
                self.brain_requirements = req
                self.local_map_shape = req.local_map_size

                # global_map = tf.zeros((1, ) + self.global_map_size + (1, ), dtype=tf.float32)
                self.true_map_input = tf.placeholder(shape=self.max_map_size + (1, ), dtype=tf.uint8)
                self.images_input = tf.placeholder(shape=req.sensor_shape + (sensor_ch,), dtype=tf.float32)
                self.xy_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                self.yaw_input = tf.placeholder(shape=(1, ), dtype=tf.float32)
                # self.action_input = tf.placeholder(shape=(2,), dtype=tf.float32)
                actions = tf.zeros((1, 1, 2), dtype=tf.float32)
                self.global_map_input = tf.placeholder(shape=self.max_map_size + (self.map_ch, ), dtype=tf.float32)
                self.visibility_input = tf.placeholder(shape=self.local_map_shape + (1, ), dtype=tf.uint8) if self.use_custom_visibility else None
                local_obj_map_labels = tf.zeros((1, 1, ) + self.local_map_shape + (1, ), dtype=np.uint8)

                self.inference_outputs = train_brain.sequential_inference(
                    self.true_map_input[None], self.images_input[None, None], self.xy_input[None, None], self.yaw_input[None, None],
                    actions, prev_global_map_logodds=self.global_map_input[None],
                    local_obj_maps=local_obj_map_labels,
                    confidence_threshold=self.confidence_threshold,
                    max_confidence=self.max_confidence,
                    max_obj_confidence=0.8,
                    custom_visibility_maps=None if self.visibility_input is None else self.visibility_input[None, None],
                    is_training=True)

                # Add the variable initializer Op.
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            count_number_trainable_params(verbose=True)

            # training session
            gpuconfig, gpuname = get_tf_config(devices=params.gpu)
            self.sess = tf.Session(config=gpuconfig)
            self.sess.run(init)

            load_from_file(self.sess, params.load, partialload=params.partialload, loadcenter=[],
                           skip=params.loadskip, autorename=False)

        self.scan_map_erosion = 5

        self.global_map_logodds = None
        self.xy = None
        self.yaw = None
        self.target_xy = None
        self.step_i = 0
        self.t = time.time()

        self.reset()


    def reset(self):
        self.step_i = 0
        self.episode_i += 1
        self.t = time.time()
        self.accumulated_spin = 0.
        self.spin_direction = None

        print ("Resetting agent %d. Scene %s."%(self.episode_i, self.env._sim._current_scene))

    def plan_and_control(self, xy, yaw, target_xy, global_map_pred, ang_vel, target_fi):


        if self.start_with_spin and np.abs(self.accumulated_spin) < np.deg2rad(360 - 70) and self.step_i < 40:
            if self.spin_direction is None:
                self.spin_direction = -np.sign(target_fi)  # spin opposite direction to the goal
            self.accumulated_spin += ang_vel
            # spin

            print ("%d: spin %f: %f"%(self.step_i, self.spin_direction, self.accumulated_spin))

            action = (2 if self.spin_direction > 0 else 3)
            planned_path = np.zeros((0, 2))
            return action, planned_path

        assert global_map_pred.dtype == np.float32
        global_map_pred = (global_map_pred * 255.).astype(np.uint8)

        # Scan map and cost graph.
        scan_graph, scan_map, resized_scan_map, cost_map = Expert.get_graph_and_eroded_map(
            raw_trav_map=global_map_pred[..., :1],
            trav_map_for_simulator=global_map_pred[..., :1],
            raw_scan_map=global_map_pred,
            rescale_scan_map=1.,
            erosion=self.scan_map_erosion,
            build_graph=False,
            interactive_channel=False,
            cost_setting=1,
        )

        # plt.figure()
        # plt.imshow(cost_map)
        # plt.show()
        # pdb.set_trace()

        # scan_map = global_map_pred[..., :1]
        #
        # scan_map = cv2.erode(scan_map, kernel=np.ones((3, 3)))
        # scan_map[scan_map<255] = 0
        #
        # cost_map = np.zeros_like(global_map_pred, dtype=np.float32)
        # cost_map[global_map_pred == 0] = 1000.
        #
        # temp_map1 = scan_map
        # temp_map2 = cv2.erode(temp_map1, kernel=np.ones((3, 3)))
        # temp_filter = np.logical_and(temp_map2 < 255, temp_map1 == 255)
        # cost_map[temp_filter] = 100.
        #
        # temp_map1 = scan_map
        # temp_map2 = cv2.erode(temp_map1, kernel=np.ones((7, 7)))
        # temp_filter = np.logical_and(temp_map2 < 255, temp_map1 == 255)
        # cost_map[temp_filter] = 1.

        action, obstacle_distance, planned_path, status_message = Expert.discrete_policy(
            scan_map=scan_map, pos_map_float=xy, yaw=yaw, target_map_float=target_xy, cost_map=cost_map)

        print ("%d: %f %s"%(self.step_i, time.time()-self.t, status_message))
        self.t = time.time()

        return action, planned_path

    def act(self, observations):
        # import ipdb; ipdb.set_trace()

        initial_target_r_meters, initial_target_fi = observations['pointgoal']

        if self.step_i == 0:
            self.step_i += 1

            self.initial_xy = np.zeros((2, ), np.float32)
            self.initial_yaw = np.zeros((1, ), np.float32)

            # Only relevant after reset
            initial_target_r_meters, initial_target_fi = observations['pointgoal']
            target_r = initial_target_r_meters / 0.05  # meters to grid cells

            target_xy = rotate_2d(np.array([target_r, 0.], np.float32), self.initial_yaw + initial_target_fi)
            target_xy += self.initial_xy

            return {"action": 3}  # turn right, because first step does not provide the top down map

        info = self.env.get_metrics()

        if self.pose_estimation_source == 'true':
            xy = np.array(info['top_down_map']['agent_map_coord'])  # x: downwards; y: rightwars
            yaw = info['top_down_map']['agent_angle']  # 0 downwards, positive ccw. Forms stanard coord system with x and y.
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # normalize

            # Recover target pos
            agent_pos = self.env.sim.get_agent_state().position
            goal_pos = self.env.current_episode.goals[0].position
            true_xy = maps.to_grid(agent_pos[0], agent_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX,
                                   (self.top_down_map_resolution, self.top_down_map_resolution), keep_float=True)
            offset_xy = xy - true_xy
            true_target_xy = maps.to_grid(goal_pos[0], goal_pos[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX,
                                          (self.top_down_map_resolution, self.top_down_map_resolution), keep_float=True)
            true_target_xy += offset_xy

            target_xy = true_target_xy
        else:
            raise NotImplementedError

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
        print (images.min(), images.max())
        images = (images * 255).astype(np.uint8)
        images = np.array(images, np.float32)
        # images = images * 255  # to unit8 0..255 format
        images = images * (2. / 255.) - 1.  # to network input -1..1 format

        true_global_map = info['top_down_map']['map']
        true_global_map = (true_global_map > 0).astype(np.uint8) * 255
        true_global_map = np.atleast_3d(true_global_map)


        # Initialize map. TODO adjust size when starting 0,0
        if self.step_i == 1:
            self.global_map_logodds = np.zeros(true_global_map.shape, np.float32)
            self.prev_yaw = yaw

        map_shape = self.global_map_logodds.shape

        # Map prediction
        global_map_label = true_global_map.copy()
        true_map_input = np.zeros(self.max_map_size + (1, ), np.uint8)
        global_map_label = global_map_label[:self.max_map_size[0], :self.max_map_size[1]]
        true_map_input[:global_map_label.shape[0], :global_map_label.shape[1]] = global_map_label

        last_global_map_input = np.zeros(self.max_map_size + (self.map_ch, ), np.float32)
        last_global_map_input[:map_shape[0], :map_shape[1]] = self.global_map_logodds

        feed_dict = {
            self.images_input: images, self.xy_input: xy, self.yaw_input: np.array((yaw, )),
            self.global_map_input: last_global_map_input,
            self.true_map_input: true_map_input,
        }
        if self.visibility_input is not None:
            visibility_map = ClassicMapping.is_visible_from_depth(depth, self.local_map_shape, zoom_factor=self.brain_requirements.transform_window_scaler)
            feed_dict[self.visibility_input] = visibility_map[:, :, None].astype(np.uint8)

        outputs = self.run_inference(feed_dict)

        global_map_logodds = np.array(outputs.global_map_logodds[0, -1])  # squeeze batch and traj
        global_map_logodds = global_map_logodds[:map_shape[0], :map_shape[1]]
        self.global_map_logodds = global_map_logodds

        local_map_label = outputs.local_map_label[0, 0, :, :, 0]
        local_map_pred = outputs.combined_local_map_pred[0, 0, :, :, 0]

        # global_map_true = self.inverse_logodds(self.global_map_logodds[:, :, 0])
        global_map_true_partial = ClassicMapping.inverse_logodds(self.global_map_logodds[:, :, 0:1])
        global_map_pred = ClassicMapping.inverse_logodds(self.global_map_logodds[:, :, 1:2])
        local_obj_map_pred = None

        if self.map_source == 'true':
            assert global_map_label.ndim == 3
            global_map_for_planning = true_global_map.astype(np.float32) * (1./255.)  # Use full true map

        else:
            global_map_for_planning = global_map_pred

        # threshold
        traversable_threshold = 0.499  # higher than this is traversable
        object_treshold = 0.  # treat everything as non-object
        threshold_const = np.array((traversable_threshold, object_treshold))[None, None, :self.map_ch-1]
        global_map_for_planning = np.array(global_map_for_planning >= threshold_const, np.float32)

        # # plan
        # action, planned_path = self.plan_and_control(
        #     self.xy, self.yaw, lin_vel, ang_vel, target_r, initial_target_fi, global_map_pred=global_map_for_planning,
        #     target_xy=self.target_xy)
        #

        # global_map_pred = true_global_map

        ang_vel = yaw - self.prev_yaw
        ang_vel = (ang_vel + np.pi) % (2*np.pi) - np.pi

        target_dist = np.linalg.norm(target_xy - xy)

        if target_dist < 3.:
            # Close enough to target. Normal requirement is 0.36/0.05 = 7.2
            action = 0
        else:
            action, planned_path = self.plan_and_control(xy, yaw, target_xy, global_map_for_planning, ang_vel, initial_target_fi)
            # Visualize agent
            if self.step_i % PLOT_EVERY_N_STEP == 0 and PLOT_EVERY_N_STEP > 0:
                self.visualize_agent(outputs, images, global_map_pred, global_map_for_planning, global_map_label,
                                     global_map_true_partial, local_map_pred, local_map_label, planned_path,
                                     sim_rgb=observations['rgb'], local_obj_map_pred=local_obj_map_pred,
                                     xy=xy, yaw=yaw, target_xy=target_xy)

        # Overwrite with expert
        if self.action_source == 'expert':
            best_action = self.follower.get_next_action(goal_pos)
            action = best_action

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
        self.step_i += 1

        print ("Taking action %d"%action)

        return {"action": action}   # 0: stop, forward, left, right
        # return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}

    def run_inference(self, feed_dict):
        outputs = self.sess.run(self.inference_outputs, feed_dict=feed_dict)
        return outputs

    def visualize_agent(self, outputs, images, global_map_pred, global_map_for_planning, global_map_label,
                        global_map_true_partial, local_map_pred, local_map_label, planned_path, sim_rgb=None,
                        local_obj_map_pred=None, xy=None, yaw=None, target_xy=None):

        # Coordinate systems dont match the ones assumed in these plot functions, but all cancells out except for yaw
        yaw = yaw - np.pi/2

        status_msg = "step %d" % (self.step_i,)
        visibility_mask = outputs.tiled_visibility_mask[0, 0, :, :, 0]
        # assert global_map_label.shape[-1] == 3
        global_map_label = np.concatenate(
            [global_map_label, np.zeros_like(global_map_label), np.zeros_like(global_map_label)], axis=-1)
        plt.figure("Global map label")
        plt.imshow(global_map_label)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=target_xy, path=planned_path)
        plt.title(status_msg)
        plt.savefig('./temp/global-map-label.png')
        plt.figure("Global map (%d)" % self.step_i)

        map_to_plot = global_map_pred[..., :1]
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=target_xy, path=planned_path)
        # plot_target_and_path(target_xy=target_xy_vel, path=np.array(self.hist2)[:, :2])
        plt.title(status_msg)
        plt.savefig('./temp/global-map-pred.png')

        if global_map_pred.shape[-1] == 2:
            map_to_plot = global_map_pred[..., 1:2]
            map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
            plt.imshow(map_to_plot)
            plot_viewpoints(xy[0], xy[1], yaw)
            plot_target_and_path(target_xy=target_xy, path=planned_path)
            plt.title(status_msg)
            plt.savefig('./temp/global-obj-map-pred.png')

        # plt.figure("Global map true (%d)" % self.step_i)
        # map_to_plot = global_map_true_partial
        # map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        # plt.imshow(map_to_plot)
        # plot_viewpoints(xy[0], xy[1], yaw)
        # plot_target_and_path(target_xy=self.target_xy, path=planned_path)
        # # plot_target_and_path(target_xy=self.target_xy, path=np.array(self.hist1)[:, :2])
        # # plot_target_and_path(target_xy=self.target_xy_vel, path=np.array(self.hist2)[:, :2])
        # plt.title(status_msg)
        # plt.savefig('./temp/global-map-true.png')
        # plt.figure("Global map plan (%d)" % self.step_i)

        map_to_plot = global_map_for_planning
        map_to_plot = np.pad(map_to_plot, [[0, 0], [0, 0], [0, 3-map_to_plot.shape[-1]]])
        plt.imshow(map_to_plot)
        plot_viewpoints(xy[0], xy[1], yaw)
        plot_target_and_path(target_xy=target_xy, path=planned_path)
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
        plt.imshow(local_map_pred * visibility_mask + (1 - visibility_mask) * 0.5)
        plt.axes(images_axarr[1, 1])
        if local_obj_map_pred is not None:
            plt.imshow(local_obj_map_pred * visibility_mask + (1 - visibility_mask) * 0.5)
        else:
            plt.imshow(local_map_label * visibility_mask + (1 - visibility_mask) * 0.5)
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

