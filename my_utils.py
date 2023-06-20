from deep_rl import *

def prepare_agent_for_task(agent, task_info):
    config = agent.config
    states = agent.task.reset_task(task_info)
    agent.states = config.state_normalizer(states)
    agent.data_buffer.clear()
    agent.task_train_start(task_info['task_label'])


def delete():
    pass 

# determines if training for the current task is completed
# if this is the case the agent is notified and saved
def is_task_training_complete(agent, task_idx):
    config = agent.config
    task_steps_limit = config.max_steps * (task_idx + 1)
    if config.max_steps and agent.total_steps >= task_steps_limit:
        agent.task_train_end()
        agent.save(config.log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (config.agent_name, \
                config.tag, agent.task.name, 1, task_idx+1))
        agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (config.agent_name, config.tag, \
            agent.task.name))
        return True
    return False

# =======================================
# =========== Mask Parameters ===========
# =======================================

def set_betas(agent, task_idx, new_betas: torch.tensor):
    network = agent.network.network
    network.fc_action.betas.data[task_idx] = new_betas
    network.fc_critic.betas.data[task_idx] = new_betas
    for layer in network.phi_body.layers._modules.values():
        layer.betas.data[task_idx] = new_betas

def set_betas_trainable(agent, trainable):
    network = agent.network.network
    network.fc_action.betas.requires_grad = trainable
    network.fc_critic.betas.requires_grad = trainable
    for layer in network.phi_body.layers._modules.values():
        layer.betas.requires_grad = trainable

def print_betas(agent):
    network = agent.network.network
    print(network.fc_action.betas)
    print(network.fc_critic.betas)
    for layer in network.phi_body.layers._modules.values():
        print(layer.betas)