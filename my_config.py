#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *

import json

from my_tasks import MyMiniGrid, MyMiniGridFlatObs

def _build_config_base(env_name, env_config_path, log_dir=None):
    # load env_config from the log_dir if there is one given
    if log_dir:
        env_config_path = f"{log_dir}/env_config.json"

    config = Config()
    config.env_name = env_name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'supermask'
    config.num_workers = 4

    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    config.new_task_mask = env_config.get('new_task_mask', 'random')
    
    config.seed = env_config.get('config_seed', 42)
    random_seed(config.seed)

    if log_dir:
        config.log_dir = log_dir
        config.logger = get_logger(log_dir=config.log_dir, file_name='eval-log')
    else:
        experiment_name = env_config_path.split('/')[-1].split('.')[0]
        log_name = f"{experiment_name}-{config.seed}-mask-{config.new_task_mask}"
        config.log_dir = get_default_log_dir(log_name)
        config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    # setting log_dir_monitor_files creates monitor files for the tasks
    # not needed here, hence, it is set to None
    config.log_dir_monitor_files = None

    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.25
    config.iteration_log_interval = 1
    config.gradient_clip = 0.5
    config.max_steps = env_config.get('max_steps')
    config.cl_requires_task_label = True

    config.evaluation_episodes = 24
    config.eval_interval = 20
    config.record_evaluation = True
    config.frames_per_sec = 5

    config.cl_num_learn_blocks = 1

    return config, env_config_path

# `log_dir` can be set if an existing log dir shall be used.
# The value of `env_config_path` is ignored in that case.
def build_minigrid_config(env_config_path, log_dir=None):
    env_name = Config.ENV_MINIGRID
    config, env_config_path = _build_config_base(env_name, env_config_path, log_dir=log_dir)

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    num_tasks = len(env_config['tasks'])
    del env_config
    
    config.task_ids = np.arange(num_tasks).tolist()

    task_fn = lambda log_dir: MyMiniGridFlatObs(config, env_config_path, eval_mode=False)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir_monitor_files)
    eval_task_fn = lambda log_dir: MyMiniGridFlatObs(config, env_config_path, eval_mode=True)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200), num_tasks=num_tasks, new_task_mask=config.new_task_mask),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks,
        new_task_mask=config.new_task_mask)
    config.policy_fn = SamplePolicy
    
    # No state normalizer needed, therefore, scaling by the factor of 1
    config.state_normalizer = RescaleNormalizer(1)

    return config