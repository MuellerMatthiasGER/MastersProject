from deep_rl import *

def prepare_agent_for_task(agent, task_info):
    config = agent.config
    states = agent.task.reset_task(task_info)
    agent.states = config.state_normalizer(states)
    agent.data_buffer.clear()
    agent.task_train_start(task_info['task_label'])

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

# list of layer names can be provided, in which the betas shall be set, default is all layers
def set_betas(agent, task_idx, new_betas: torch.tensor, layer_names=['layers0', 'layers1', 'layers2', 'fc_action', 'fc_critic']):
    network = agent.network.network
    if 'fc_action' in layer_names:
        network.fc_action.betas.data[task_idx] = new_betas
    if 'fc_critic' in layer_names:
        network.fc_critic.betas.data[task_idx] = new_betas
    for i, layer in enumerate(network.phi_body.layers._modules.values()):
        if f'layers{i}' in layer_names:
            layer.betas.data[task_idx] = new_betas

def set_betas_trainable(agent, trainable, layer_names=['layers0', 'layers1', 'layers2', 'fc_action', 'fc_critic']):
    network = agent.network.network
    if 'fc_action' in layer_names:
        network.fc_action.betas.requires_grad = trainable
    if 'fc_critic' in layer_names:
        network.fc_critic.betas.requires_grad = trainable
    for i, layer in enumerate(network.phi_body.layers._modules.values()):
        if f'layers{i}' in layer_names:
            layer.betas.requires_grad = trainable

def print_betas(agent):
    network = agent.network.network
    print(network.fc_action.betas)
    print(network.fc_critic.betas)
    for layer in network.phi_body.layers._modules.values():
        print(layer.betas)