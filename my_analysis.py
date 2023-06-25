from deep_rl import *

from my_logger import *
from my_config import *

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

def _plot_hm_betas(data, title, fname):
    n_tasks = data.shape[0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    im = ax.imshow(data, cmap='YlGn', vmin=0.0, vmax=0.5)
    ax.set_xticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)
    ax.set_yticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_tasks):
        for j in range(n_tasks):
         text = ax.text(j, i, '{0:.2f}'.format(data[i, j]), ha='center', \
            va='center', fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.savefig(fname)
    plt.close(fig)

def _analyse_linear_coefficients(agent):
    config = agent.config
    lc_save_path = create_log_folder(config, 'linear_comb')
    
    for k, v in agent.network.named_parameters():
        if 'betas' in k:
            k = k.split('.')
            if len(k) == 3: # for network.fc_action.betas or network.fc_critic.betas
                k = '.'.join(k[1:])
            else: # for network.phi_body.layers.x.betas
                k = k[2] + k[3] + '.' + k[4]
            
            _data = copy.deepcopy(v.detach().cpu())
            _data[0, 0] = 1.0 # manually set as there is no linear combination for the first task
            _plot_hm_betas(_data.numpy(), k, '{0}/betas_before_softmax_{1}.pdf'.format(lc_save_path, k))
            # apply softmax to get probabilities of co-efficient parameters
            for _idx in range(_data.shape[0]):
                _data[_idx, 0:_idx+1] = torch.softmax(_data[_idx, 0:_idx+1] * 5, dim=0) # MYEDIT *5 is new
            _data = _data.numpy()
            _plot_hm_betas(_data, k, '{0}/betas_{1}.pdf'.format(lc_save_path, k))


def _plot_train_performance(agent):
    config = agent.config

    tasks = [task_info['task'] for task_info in config.cl_tasks_info]
    data = np.loadtxt(f"{config.log_dir}/eval_data.csv", delimiter=',')

    steps_between_evals = config.rollout_length * config.num_workers * config.eval_interval
    num_rows = data.shape[0]
    steps = np.arange(num_rows) * steps_between_evals

    for task_idx, task_name in enumerate(tasks):
        eval_data = data[:, task_idx]
        plt.plot(steps, eval_data, label=task_name)
        # draw vertical lines when new task is trained
        vertical_x = config.max_steps * task_idx
        plt.axvline(vertical_x, c='black')
        plt.text(vertical_x + (config.max_steps * 0.05), 0.1, f"Train Task {task_idx}", rotation=90)

    plt.xlabel("Steps")
    plt.ylabel("Avg. Reward")
    plt.ylim(top=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.savefig(f"{config.log_dir}/task_stats/plot.png")


def analyse_agent(agent):
    _plot_train_performance(agent)
    _analyse_linear_coefficients(agent)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(-1) # -1 is CPU, a positive integer is the index of GPU

    path = "./log/minigrid_color_shape-42-mask-linear_comb/230625-151518"
    config = build_minigrid_config(None, log_dir=path)

    # load agent
    agent = LLAgent(config)
    config.agent_name = agent.__class__.__name__
    # model_path = '{0}/{1}-{2}-model-{3}.bin'.format(path, config.agent_name, config.tag, config.env_name)
    # agent.load(model_path)

    # analyse_agent(agent)
    _plot_train_performance(agent)

    agent.close()
    