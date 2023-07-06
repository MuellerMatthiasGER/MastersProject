#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.mask_modules.mmn.mask_nets import MultitaskMaskLinearSparse
from deep_rl.network.network_heads import *

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
# with custom masks
class MyCategoricalActorCriticNet_SS(nn.Module, BaseNet):
    def __init__(self,
                 config,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(MyCategoricalActorCriticNet_SS, self).__init__()
        self.network = MyActorCriticNetSS(config, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output


class MyActorCriticNetSS(nn.Module):
    def __init__(self, config, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, \
        new_task_mask, discrete_mask=True):
        super(MyActorCriticNetSS, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        if config.mask_type == 'threshold_mask':
            self.fc_action = MultitaskMaskLinear(actor_body.feature_dim, action_dim, \
                discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
            self.fc_critic = MultitaskMaskLinear(critic_body.feature_dim, 1, \
                discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        elif config.mask_type == 'sparse_mask':
            self.fc_action = MultitaskMaskLinearSparse(actor_body.feature_dim, action_dim, \
                discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
            self.fc_critic = MultitaskMaskLinearSparse(critic_body.feature_dim, 1, \
                discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        

        ap = [p for p in self.actor_body.parameters() if p.requires_grad is True]
        ap += [p for p in self.fc_action.parameters() if p.requires_grad is True]
        self.actor_params = ap

        cp = [p for p in self.critic_body.parameters() if p.requires_grad is True]
        cp += [p for p in self.fc_critic.parameters() if p.requires_grad is True]
        self.critic_params = cp

        self.phi_params = [p for p in self.phi_body.parameters() if p.requires_grad is True]