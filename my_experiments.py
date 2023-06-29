from deep_rl import *
from my_agent import MyLLAgent

from my_config import build_minigrid_config
from my_analysis import analyse_agent
from my_utils import *
from my_logger import *
from my_eval import *

def independent_masks(env_config_path, seed, max_steps, n_independent):
    new_task_mask = 'linear_comb'
    config = build_minigrid_config(env_config_path, seed, new_task_mask, max_steps)
    pickle_run_parameters(config, env_config_path=env_config_path, seed=seed, new_task_mask=new_task_mask, max_steps=max_steps, n_independent=n_independent)

    agent = LLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks_info = agent.config.cl_tasks_info
    num_tasks = len(tasks_info)

    create_log_structure(config)

    iteration = 0

    for task_idx, task_info in enumerate(tasks_info):
        log_new_task_starts(config, task_idx, task_info)

        prepare_agent_for_task(agent, task_info)

        # experiment specific part: set beta values depending on if the task shall be learned independently or not
        if task_idx < n_independent:
            config.logger.info('new_task_mask: random')
            # change betas from equal distribution to using exclusively the new random mask
            new_betas = tensor = torch.zeros(num_tasks)
            new_betas[task_idx] = 100
            set_betas(agent, task_idx, new_betas)
            set_betas_trainable(agent, trainable=False)
        else:
            config.logger.info('new_task_mask: linear_comb')
            set_betas_trainable(agent, trainable=True)

        while True:
            # train step
            dict_logs = agent.iteration()
            iteration += 1

            # logging
            log_iteration(agent, iteration)
            print_betas(agent)

            # evaluate agent
            eval_agent(agent, tasks_info, iteration)

            # check whether task training has been completed
            if is_task_training_complete(agent, task_idx):
                break

        # end of while True. current task training
    # end for each task

    # Analyse
    analyse_agent(agent)

    agent.close()


def hyperparameter_search(env_config_path):
    entropy_weights = [0.01, 0.05, 0.1]
    ratio_clips = [0.1, 0.25, 0.5]
    gradient_clips = [0.5, 1, 2]

    def run(config):
        agent = MyLLAgent(config)
        config.agent_name = agent.__class__.__name__
        tasks_info = agent.config.cl_tasks_info

        create_log_structure(config)

        iteration = 0

        # evaluate agent before training for baseline of random init
        eval_agent(agent, tasks_info, iteration)

        for task_idx, task_info in enumerate(tasks_info):
            prepare_agent_for_task(agent, task_info)

            while True:
                # train step
                dict_logs = agent.iteration()
                iteration += 1

                # evaluate agent
                eval_agent(agent, tasks_info, iteration)

                # check whether task training has been completed
                if is_task_training_complete(agent, task_idx):
                    break

        #     end of while True / current task training
        # end for each task

        # Analysis
        analyse_agent(agent)

        agent.close()

    run_idx = 0
    for entropy_weight in entropy_weights:
        for ratio_clip in ratio_clips:
            for gradient_clip in gradient_clips:
                print("----------------------")
                print("----------------------")
                print("    Run Index: ", run_idx)
                print("----------------------")
                print("----------------------")
                run_idx += 1

                config = build_minigrid_config(env_config_path)
                config.entropy_weight = entropy_weight
                config.ppo_ratio_clip = ratio_clip
                config.gradient_clip = gradient_clip
                run(config)


def learn_color_shape(env_config_path):
    config = build_minigrid_config(env_config_path)

    agent = MyLLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks_info = agent.config.cl_tasks_info

    create_log_structure(config)

    iteration = 0

    # evaluate agent before training for baseline of random init
    eval_agent(agent, tasks_info, iteration)

    for task_idx, task_info in enumerate(tasks_info):
        log_new_task_starts(config, task_idx, task_info)

        prepare_agent_for_task(agent, task_info)

        while True:
            # train step
            dict_logs = agent.iteration()
            iteration += 1

            # logging
            log_iteration(agent, iteration)

            # evaluate agent
            eval_agent(agent, tasks_info, iteration)

            # check whether task training has been completed
            if is_task_training_complete(agent, task_idx):
                break

    #     end of while True / current task training
    # end for each task

    # Analysis
    analyse_agent(agent)

    agent.close()


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    env_config_path = "./env_configs/minigrid_color_shape.json"
    learn_color_shape(env_config_path)
