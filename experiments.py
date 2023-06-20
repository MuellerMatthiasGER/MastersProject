import shutil
from my_config import build_minigrid_config
from deep_rl import *

def independent_masks():
    env_config_path = "./env_configs/minigrid_independent_masks.json"
    seed = 42
    new_task_mask = 'random'
    max_steps = 75000

    n_independent = 2

    config = build_minigrid_config(env_config_path, seed, new_task_mask, max_steps)
    agent = LLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks_info = agent.config.cl_tasks_info
    num_tasks = len(tasks_info)

    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks_info, f)
    # run_iterations_w_oracle(agent, tasks_info)

    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)

    iteration = 0
    steps = []
    rewards = []

    for task_idx, task_info in enumerate(tasks_info[:n_independent]):
        config.logger.info('*****start training on task {0}'.format(task_idx))
        config.logger.info('name: {0}'.format(task_info['name']))
        config.logger.info('task: {0}'.format(task_info['task']))
        config.logger.info('task_label: {0}'.format(task_info['task_label']))
        config.logger.info('new_task_mask: random')

        states = agent.task.reset_task(task_info)
        agent.states = config.state_normalizer(states)
        agent.data_buffer.clear()
        agent.task_train_start(task_info['task_label'])

        while True:
            # train step
            dict_logs = agent.iteration()

            iteration += 1
            steps.append(agent.total_steps)
            rewards.append(np.mean(agent.iteration_rewards))

            # logging
            if iteration % config.iteration_log_interval == 0:
                config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
                    iteration, agent.total_steps,
                    np.mean(agent.iteration_rewards),
                    np.max(agent.iteration_rewards),
                    np.min(agent.iteration_rewards)
                ))

            # check whether task training has been completed
            task_steps_limit = config.max_steps * (task_idx + 1)
            if config.max_steps and agent.total_steps >= task_steps_limit:
                agent.task_train_end()
                agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (config.agent_name, \
                        config.tag, agent.task.name, 1, task_idx+1))
                agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (config.agent_name, config.tag, \
                    agent.task.name))
                break
        # end of while True. current task training
    # end for each task

    network = agent.network.network
    network.fc_action.new_mask_type = mask_nets.NEW_MASK_LINEAR_COMB
    network.fc_action.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
    network.fc_action._forward_mask = network.fc_action._forward_mask_linear_comb
    network.fc_critic.new_mask_type = mask_nets.NEW_MASK_LINEAR_COMB
    network.fc_critic.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
    network.fc_critic._forward_mask = network.fc_critic._forward_mask_linear_comb
    for layer in network.phi_body.layers._modules.values():
        layer.new_mask_type = mask_nets.NEW_MASK_LINEAR_COMB
        layer.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
        layer._forward_mask = layer._forward_mask_linear_comb

    # train last task
    for task_idx, task_info in enumerate(tasks_info[n_independent:]):
        task_idx += n_independent

        config.logger.info('*****start training on task {0}'.format(task_idx))
        config.logger.info('name: {0}'.format(task_info['name']))
        config.logger.info('task: {0}'.format(task_info['task']))
        config.logger.info('task_label: {0}'.format(task_info['task_label']))
        config.logger.info('new_task_mask: linear_comb')

        states = agent.task.reset_task(task_info)
        agent.states = config.state_normalizer(states)
        agent.data_buffer.clear()
        agent.task_train_start(task_info['task_label'])

        while True:
            # train step
            dict_logs = agent.iteration()

            iteration += 1
            steps.append(agent.total_steps)
            rewards.append(np.mean(agent.iteration_rewards))

            # logging
            if iteration % config.iteration_log_interval == 0:
                config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
                    iteration, agent.total_steps,
                    np.mean(agent.iteration_rewards),
                    np.max(agent.iteration_rewards),
                    np.min(agent.iteration_rewards)
                ))

            # check whether task training has been completed
            task_steps_limit = config.max_steps * (task_idx + 1)
            if config.max_steps and agent.total_steps >= task_steps_limit:
                agent.task_train_end()
                agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (config.agent_name, \
                        config.tag, agent.task.name, 1, task_idx+1))
                agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (config.agent_name, config.tag, \
                    agent.task.name))
                break
        # end of while True. current task training
    # end for each task

    agent.close()


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    independent_masks()