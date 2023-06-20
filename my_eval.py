import numpy as np

# returns true if agent was evaluated
def eval_agent(agent, tasks_info, iteration):
    config = agent.config
    if (agent.config.eval_interval is not None and \
            iteration % agent.config.eval_interval == 0):
        config.logger.info('*****agent / evaluation block')
        eval_data = np.zeros(len(tasks_info),)

        for task_idx, task_info in enumerate(tasks_info):
            # prepare agent for evaluation on a given task
            agent.task_eval_start(task_info['task_label'])
            agent.evaluation_states = agent.evaluation_env.reset_task(task_info)

            # evaluate agent on task
            # performance can be success rate in (meta-)continualworld or rewards in other environments
            task_performance, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
            eval_data[task_idx] = np.mean(task_performance)
            config.logger.info("task: {0} - {1}".format(task_info['name'], task_performance))

        # save performances
        with open(f"{config.log_dir}/eval_data.csv", 'a') as eval_file:
            np.savetxt(eval_file, eval_data.reshape(1, -1), delimiter=',', fmt='%.4f')
        
        return True
    return False