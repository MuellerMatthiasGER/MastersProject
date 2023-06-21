import shutil
import os
import numpy as np
import pickle

def create_log_folder(config, name):
    path = f"{config.log_dir}/{name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_log_structure(config):
    # copy environment configuration file to log file
    shutil.copy(config.env_config_path, config.log_dir + '/env_config.json')

    # create task_stats directory
    config.log_path_tstats = create_log_folder(config, 'task_stats')

    # create video directory
    if config.record_evaluation:
        config.video_log_path = create_log_folder(config, 'video_log')

def pickle_run_parameters(config, **kwargs):
    data = {}
    for key, value in kwargs.items():
        data[key] = value
    
    with open(f"{config.log_dir}/run_parameters.bin", 'wb') as file:
        pickle.dump(data, file)

def log_new_task_starts(config, task_idx, task_info):
    config.logger.info('*****start training on task {0}'.format(task_idx))
    config.logger.info('name: {0}'.format(task_info['name']))
    config.logger.info('task: {0}'.format(task_info['task']))
    config.logger.info('task_label: {0}'.format(task_info['task_label']))

# returns true if the iteration is logged
def log_iteration(agent, iteration):
    config = agent.config
    if iteration % config.iteration_log_interval == 0:
        config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
            iteration, agent.total_steps,
            np.mean(agent.iteration_rewards),
            np.max(agent.iteration_rewards),
            np.min(agent.iteration_rewards)
        ))
        return True
    return False