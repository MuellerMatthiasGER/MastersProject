import json

import numpy as np
from deep_rl.component.task import BaseTask

import gymnasium as gym
import my_minigrid

import matplotlib.pyplot as plt


class MyMiniGrid(BaseTask):
    def __init__(self, config, env_config_path, log_dir=None, eval_mode=False):
        BaseTask.__init__(self)
        from minigrid.wrappers import OneHotPartialObsWrapper, ImgObsWrapper, ReseedWrapper
        from gym.wrappers import RecordVideo
        my_minigrid.register_custom_levels()

        self.name = 'MyMiniGrid'
        
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
        self.env_config = env_config
        env_names = env_config['tasks']

        # if there are env seeds given in the config file, they are set
        if 'seeds' in env_config.keys():
            seeds = env_config['seeds']
            if isinstance(seeds, int): seeds = [seeds,] * len(env_names)
            elif isinstance(seeds, list):
                assert len(seeds) == len(env_names), 'number of seeds in config file should match'\
                    ' the number of tasks.'
            else: raise ValueError('invalid seed specification in config file')
        else:
            seeds = None

        # create minigrid environments
        self.envs = {}
        for idx, env_name in enumerate(env_config['tasks']):
            name = '{0}_{1}'.format(idx, env_name)
            env = ImgObsWrapper(OneHotPartialObsWrapper(gym.make(env_name, render_mode='rgb_array')))
            
            if seeds:
                name += str(seeds[idx])
                env_seeds = [seeds[idx]] if isinstance(seeds[idx], int) else seeds[idx]
                env = ReseedWrapper(env, seeds=env_seeds)
            
            if eval_mode and config.record_evaluation:
                visual_log_path = f"{config.log_dir}/visual_log"
                env = RecordVideo(env, visual_log_path, video_length=15, name_prefix=env_name)

            self.envs[name] = env
            env_names[idx] = name
            
        # determine necessary network dimensions
        self.observation_space = self.envs[env_names[0]].observation_space
        self.action_space = self.envs[env_names[0]].action_space
        self.state_dim = self.observation_space.shape
        # note, action_dim of 3 will reduce agent action to left, right, and forward
        if 'action_dim' in env_config.keys():
            self.action_dim = env_config['action_dim']
        else:
            self.action_dim = self.envs[env_names[0]].action_space.n
        
        # env monitors
        for name in self.envs.keys():
            self.envs[name] = self.set_monitor(self.envs[name], log_dir)
        
        # task label config
        self.task_label_dim = env_config['label_dim']
        self.one_hot_labels = True if env_config['one_hot'] else False

        # all tasks
        self.tasks = [{'name': name, 'task': name, 'task_label': None} \
            for name in self.envs.keys()]
        
        # generate label for each task
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]

        # set default task
        self.current_task = self.tasks[0]
        self.env = self.envs[self.current_task['task']]

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # terminated is set if the agent dies or fulfills the goal
        # truncated is set if max steps are executed
        # for these tasks, this is not distinguished
        finished = terminated or truncated

        if finished:
            state = MyMiniGrid.reset(self)
        return state.ravel(), reward, finished, info

    def reset(self):
        state = self.env.reset()[0]
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.current_task = taskinfo
        self.env = self.envs[self.current_task['task']]
    
    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        raise NotImplementedError
   

class MyMiniGridFlatObs(MyMiniGrid):
    def __init__(self, config, env_config_path, log_dir=None, eval_mode=False):
        super(MyMiniGridFlatObs, self).__init__(config, env_config_path, log_dir, eval_mode)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

    def step(self, action):
        state, reward, finished, info = super().step(action)
        return state.ravel(), reward, finished, info

    def reset(self):
        state = super().reset()
        return state.ravel()