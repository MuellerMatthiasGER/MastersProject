import json

import numpy as np
from deep_rl.component.task import BaseTask

import gymnasium as gym
from gym import error

from deep_rl.utils.torch_utils import random_seed


class MyMiniGrid(BaseTask):
    def __init__(self, config, env_config_path, log_dir=None, eval_mode=False):
        BaseTask.__init__(self)
        from minigrid.wrappers import OneHotPartialObsWrapper, ImgObsWrapper, ReseedWrapper
        # imports important for registration of environments, even if imports are not used here
        import my_minigrid
        import my_minigrid_new
        import my_minigrid_green_blue

        self.name = config.env_name
        self.config = config
        self.is_recording = False
        self.eval_mode = eval_mode

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
            
            
            # it is also possible to set seeds only for some tasks
            # to indicate that a task shall not have a seed,
            # enter a symbol that is neither an integer nor a list 
            if seeds or eval_mode:
                if isinstance(seeds[idx], int):
                    name += "_" + str(seeds[idx])
                    env = ReseedWrapper(env, seeds=[seeds[idx]])
                elif isinstance(seeds[idx], list):
                    name += "_" + str(seeds[idx])
                    env = ReseedWrapper(env, seeds=seeds[idx])
                elif eval_mode:
                    # In this case eval mode is set, 
                    # but for the train tasks there are no seeds given.
                    # In this case a list of seeds is created (based on the rl_seed)
                    random_seed(config.seed)
                    fixed_rand_seeds = np.random.randint(0, 1000, size=config.evaluation_episodes)
                    env = ReseedWrapper(env, seeds=fixed_rand_seeds.tolist())
                else:
                    rand_generator = gym.utils.seeding.np_random(config.seed)[0]
                    env.env.np_random = rand_generator

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
        # generate a frame of the current state if the env is currently recorded
        if self.is_recording:
            frame = self.env.render()
            self.recorded_frames.append(frame)

        state, reward, terminated, truncated, info = self.env.step(action)

        # if the episode ended regular, i.e. not because of max_steps,
        # add a few more ending frames for better visability
        if self.is_recording and terminated:
            frame = self.env.render()
            for _ in range(4):
                self.recorded_frames.append(frame)

        # terminated is set if the agent dies or fulfills the goal
        # truncated is set if max steps are executed
        # for these tasks, this is not distinguished
        finished = terminated or truncated

        # eval mode resets on its own. 
        # To omit it here is only really necessary if the evaluation is done on
        # multiple seeds in one task to preserve integrity
        if finished and not self.eval_mode:
            state = MyMiniGrid.reset(self)
        return state, reward, finished, info

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
    
    def start_recording(self):
        if not self.config.record_evaluation:
            return

        self.config.logger.info("Video Recording Started")
        self.recorded_frames = []
        self.is_recording = True

    def finish_recording(self, iteration):
        if not self.config.record_evaluation:
            return

        self.config.logger.info("Video Recording Finished")

        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            clip = ImageSequenceClip(self.recorded_frames, fps=self.config.frames_per_sec)
            path = f"{self.config.video_log_path}/{iteration:05d}_{self.current_task['name']}.mp4"
            clip.write_videofile(path, verbose=False, logger=None)

        self.recorded_frames = []
        self.is_recording = False
   

class MyMiniGridFlatObs(MyMiniGrid):
    def __init__(self, config, env_config_path, log_dir=None, eval_mode=False):
        super(MyMiniGridFlatObs, self).__init__(config, env_config_path, log_dir, eval_mode)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        # TODO: make proper dependency
        self.action_memory = True
        # action memory
        if self.action_memory:
            self.state_dim += self.action_dim

    def action_memory_call(self, state, action=None):
        action_array = np.zeros(self.action_dim)
        if action:
            action_array[action] = 1

        # Concatenate 'action_array' to the state
        new_state = np.concatenate((state, action_array))
        return new_state

    def step(self, action):
        state, reward, finished, info = super().step(action)
        state = state.ravel()
        if self.action_memory:
            state = self.action_memory_call(state, action)
        return state, reward, finished, info

    def reset(self):
        state = super().reset()
        state = state.ravel()
        if self.action_memory:
            state = self.action_memory_call(state, None)
        return state