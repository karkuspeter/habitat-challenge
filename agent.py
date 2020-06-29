import argparse
import habitat
import random
import numpy
import os
import cv2
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from myhabitatagent import DSLAMAgent
import numpy as np

from arguments import parse_args


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config, params=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        print (params)

    def reset(self):
        print ("reseting..")
        self.step_i = 0
        pass

    def act(self, observations):
        self.step_i += 1
        if self.step_i > 100:
            action = 0
        else:
            action = numpy.random.choice([1, 2, 3])
        print (self.step_i, action)
        return {"action": action}


class ShortestPathAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config, env=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.step_i = 0
        self.episode_i = -2
        self.env = env
        self.follower = ShortestPathFollower(env._sim, 0.36/2., False)
        self.reset()

    def reset(self):
        self.step_i = 0
        self.episode_i += 1
        print ("Resetting agent %d. Scene %s."%(self.episode_i, self.env._sim._current_scene))

    def act(self, observations):
        # print (observations.keys())
        # import ipdb; ipdb.set_trace()

        goal_pos = self.env.current_episode.goals[0].position
        best_action = self.follower.get_next_action(goal_pos)

        if self.episode_i == 0:
            cv2.imwrite('./temp/ep%d-step%d.png'%(self.episode_i, self.step_i), observations['rgb'])
        if self.step_i == 0:
            top_down_map = maps.get_topdown_map(
                self.env.sim, map_resolution=(5000, 5000)
            )
            import matplotlib.pyplot as plt
            plt.imshow(top_down_map)
            plt.show()

        #     global_map = observations['top_down_map']
        #     cv2.imwrite('./temp/map-%d.png'%self.episode_i, global_map)
        self.step_i += 1

        return {"action": best_action}   # 0: stop, forward, left, right
        # return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}



def main():
    params = parse_args(default_files=('./habitat_submission.conf', ))
    is_submission = (params.habitat_eval != 'localtest')

    if params.seed > 0:
        np.random.seed(params.seed)
        random.seed(params.seed)

    if not is_submission:
        os.environ["CHALLENGE_CONFIG_FILE"] = './configs/challenge_pointnav_supervised.yaml'

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    # parser.add_argument("--config", type=str, default='', required=False)
    # args = parser.parse_args()
    #
    # if args.config != '':
    #     os.environ["CHALLENGE_CONFIG_FILE"] = args.config

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)

    print ("Using config file(s): %s"%(str(config_paths)))
    # agent = RandomAgent(task_config=config)

    if params.habitat_eval == "localtest":
        grid_cell_size = 0.05  # 5cm
        map_size = (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / grid_cell_size
        assert config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION == int(map_size)

        challenge = habitat.Challenge(eval_remote=False)
        env = challenge._env

        if params.seed > 0:
            env._sim.seed(params.seed)
            env.seed(params.seed)

        # agent = ShortestPathAgent(task_config=config, env=env)
        agent = DSLAMAgent(task_config=config, params=params, env=env)
        challenge.submit(agent, num_episodes=params.num_episodes, skip_first_n=params.skip_first_n)

    elif params.habitat_eval == "local":
        challenge = habitat.Challenge(eval_remote=False)

        # if params.seed > 0:
        #     challenge._env._sim.seed(params.seed)
        #     challenge._env.seed(params.seed)

        agent = DSLAMAgent(task_config=config, params=params, env=None)
        # agent = RandomAgent(task_config=config, params=params)
        challenge.submit(agent)  # , num_episodes=params.num_episodes)

    else:
        challenge = habitat.Challenge(eval_remote=True)
        agent = DSLAMAgent(task_config=config, params=params, env=None)
        # agent = RandomAgent(task_config=config, params=params)
        challenge.submit(agent)


if __name__ == "__main__":
    main()

