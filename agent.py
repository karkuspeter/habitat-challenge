import argparse
import habitat
import random
import numpy
import os
import cv2
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)

    # agent = RandomAgent(task_config=config)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    env = challenge._env
    agent = ShortestPathAgent(task_config=config, env=env)

    challenge.submit(agent)


if __name__ == "__main__":
    main()

