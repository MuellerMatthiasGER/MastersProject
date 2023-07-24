from deep_rl import *
from my_agent import MyLLAgent

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
                _data[_idx, 0:_idx+1] = torch.softmax(_data[_idx, 0:_idx+1], dim=0)
            _data = _data.numpy()
            _plot_hm_betas(_data, k, '{0}/betas_{1}.pdf'.format(lc_save_path, k))

def _plot_hm_layer_mask_diff(data, title, fname, vmin=None):
    n_tasks = data.shape[0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    im = ax.imshow(data, cmap='YlGn', vmin=vmin)
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

def _analyse_mask_diff(agent):
    config = agent.config
    diff_save_path = create_log_folder(config, 'mask_diff')
    num_tasks = len(config.task_ids)

    d = {}
    b = {}
    for k, v in agent.network.named_parameters():
        k_split = k.split('.')

        # remove every module that is not a mask (e.g., .weight, .betas)
        try: k_split[-1] = int(k_split[-1])
        except: continue

        if k_split[1] == 'phi_body': new_k = k_split[2] + k_split[3]
        elif k_split[1] == 'fc_action': new_k = k_split[1]
        elif k_split[1] == 'fc_critic': new_k = k_split[1]

        if new_k not in d.keys(): 
            d[new_k] = {}
            b[new_k] = {}
        d[new_k][k_split[-1]] = copy.deepcopy(v.detach().cpu().numpy())
        # binary of layers (apply >= 0)
        # CAUTION: MUST BE CHANGED IF MASKING ALGORITHM CHANGES
        b[new_k][k_split[-1]] = d[new_k][k_split[-1]] >= 0

    for k, v in d.items():
        diff_norm_data = np.zeros((num_tasks, num_tasks))
        diff_mean_data = np.zeros((num_tasks, num_tasks))

        same_score_sign = np.zeros((num_tasks, num_tasks))
        num_weights = np.float32(b[k][0].shape[0] * b[k][0].shape[1])

        for i in range(num_tasks):
            for j in range(num_tasks):
                diff_norm_data[i, j] = np.linalg.norm(d[k][i] - d[k][j])
                diff_mean_data[i, j] = np.mean(np.abs(d[k][i] - d[k][j]))
                
                same_score_sign[i, j] = (b[k][i] == b[k][j]).sum() / num_weights

        _plot_hm_layer_mask_diff(same_score_sign, \
            'Same score sign in percent for {0}'.format(k), \
            '{0}/layer_{1}_same_score_sign.pdf'.format(diff_save_path, k), vmin=0.5)
        print(k)
        # _plot_hm_layer_mask_diff(diff_norm_data, \
        #     'Mask correlation for across tasks for {0}'.format(k), \
        #     '{0}/layer_{1}_mask_diff_norm.pdf'.format(diff_save_path, k))
        # _plot_hm_layer_mask_diff(diff_mean_data, \
        #     'Mask correlation for across tasks for {0}'.format(k), \
        #     '{0}/layer_{1}_mask_diff_mean.pdf'.format(diff_save_path, k))

def _plot_eval_performance(agent, comparison_log_dir=None):
    config = agent.config

    tasks = [task_info['task'] for task_info in config.cl_tasks_info]
    data = np.loadtxt(f"{config.log_dir}/eval_data.csv", delimiter=',')

    steps_between_evals = config.rollout_length * config.num_workers * config.eval_interval
    num_rows = data.shape[0]
    steps = np.arange(num_rows) * steps_between_evals

    # comparison plot
    if comparison_log_dir:
        comp_env_config_path = f"{comparison_log_dir}/env_config.json"
        with open(comp_env_config_path, 'r') as f:
            comp_env_config = json.load(f)

        comp_tasks = comp_env_config['tasks']
        comp_data = np.loadtxt(f"{comparison_log_dir}/eval_data.csv", delimiter=',')

        num_rows = comp_data.shape[0]
        steps = np.arange(num_rows) * steps_between_evals

        gray_values = np.linspace(0.5, 0.7, len(comp_tasks))

        for task_idx, task_name in enumerate(comp_tasks):
            if len(data.shape) > 1:
                eval_data = comp_data[:, task_idx]
            else:
                # one dim data, i.e. only one task
                eval_data = comp_data
            plt.plot(steps, eval_data, label=task_name, color=plt.cm.gray(gray_values[task_idx]))

    # main plot
    for task_idx, task_name in enumerate(tasks):
        if len(data.shape) > 1:
            eval_data = data[:, task_idx]
        else:
            # one dim data, i.e. only one task
            eval_data = data
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
    plt.savefig(f"{config.log_dir}/task_stats/plot{'_comp' if comparison_log_dir else ''}.png")


def analyse_agent(agent):
    _plot_eval_performance(agent)
    _analyse_linear_coefficients(agent)
    _analyse_mask_diff(agent)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(-1) # -1 is CPU, a positive integer is the index of GPU

    path = "./log_safe/minigrid_green_blue-42-mask-linear_comb/230706-121833"
    config = build_minigrid_config(None, log_dir=path)

    # load agent
    agent = MyLLAgent(config)
    config.agent_name = agent.__class__.__name__
    model_path = '{0}/{1}-{2}-model-{3}.bin'.format(path, config.agent_name, config.tag, config.env_name)
    agent.load(model_path)

    # analyse_agent(agent)

    comp_log_dir = "./log_safe/minigrid_green_blue-42-mask-random/230706-151708"
    _plot_eval_performance(agent, comparison_log_dir=comp_log_dir)


    # _analyse_mask_diff(agent)


    agent.close()
    