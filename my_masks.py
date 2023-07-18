#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.mask_modules.mmn.mask_nets import GetSubnetContinuous, GetSubnetDiscrete, MultitaskMaskLinearSparse, mask_init, signed_constant
from deep_rl.network.network_heads import *
import torch.autograd as autograd

class MyGetSubnetDiscrete(autograd.Function):
    # only > instead of >=
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores > a).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g
    
class MyGetSubnetContinuous(autograd.Function):
    # only > instead of >=
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores > a).float() * scores

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

class MyMultitaskMaskLinear(MultitaskMaskLinear):
    def __init__(self, *args, discrete=True, num_tasks=1, 
                 new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, discrete=discrete, num_tasks=num_tasks, 
                         new_mask_type=new_mask_type, **kwargs)
        
        self.num_masks = 16
        self.num_masks_init = 0

        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(self.num_masks)
            ]
        )
        self.betas = nn.Parameter(torch.zeros(num_tasks, self.num_masks).type(torch.float32))

        # subnet class
        self._subnet_class = MyGetSubnetDiscrete

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.num_masks_init - 1]

        # check if this is the first task
        if self.num_tasks_learned == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet)

        # combine task mask with masks from other tasks.
        _subnets = [self.scores[idx].detach() for idx in range(self.num_masks_init - 1)]
        for net in _subnets:
            net[net < 0] = 0
        
        _subnet = self.scores[self.num_masks_init - 1]
        _subnets.append(_subnet)

        _betas = self.betas[self.task, 0:self.num_masks_init]
        _betas = MyGetSubnetContinuous.apply(_betas)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]

        # if the task is an old one, don't consider the new current mask
        if self.task < self.num_tasks_learned:
            _subnets = _subnets[:-1]

        # element wise sum of various masks
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb)

    @torch.no_grad()
    def consolidate_mask(self):
        # no task learned so far
        if self.task == 0:
            self.scores[0].data[self.scores[0] < 0] = 0
            return

        _subnets = [self.scores[idx] for idx in range(self.num_masks_init - 1)]
        existing_subnet_masks = [net.type(torch.bool) for net in _subnets]

        _subnet = self.scores[self.num_masks_init - 1]
        _subnets.append(_subnet)

        _betas = self.betas[self.task, 0:self.num_masks_init]
        _betas = MyGetSubnetContinuous.apply(_betas)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]

        # element wise sum of various masks
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        _subnet_linear_comb[_subnet_linear_comb < 0] = 0
        self.scores[self.num_masks_init - 1].data = _subnet_linear_comb.data
        final_mask = _subnet_linear_comb.type(torch.bool)

        # check if new mask is used at all
        if self.betas[self.task, self.num_masks_init - 1] <= 0:
            return

        # make new mask disjunct to existing masks
        for idx, mask in enumerate(existing_subnet_masks):
            # check if masks are already disjunct
            if not (mask & final_mask).any():
                continue

            # check if the new one is a subset of the old one
            if not (final_mask & ~mask).any():
                # then remove new mask from old mask
                self.scores[idx].data[final_mask] = 0

                # all older masks that use this mask must also use the new one to be complete
                self.betas[self.betas[:, idx] > 0, self.num_masks_init - 1] = self.betas[self.betas[:, idx] > 0, idx]
            
            # check if the old one is a subset of the new one
            elif not (mask & ~final_mask).any():
                # then remove old mask from new one
                self.scores[self.num_masks_init - 1].data[mask] = 0

                # ensure that old mask is used for this task
                if self.betas[self.task, idx] <= 0:
                   self.betas[self.task, idx] = self.betas[self.task, self.num_masks_init - 1]

            else:
                # remove both masks from each other and the intersection is the new one
                self.scores[self.num_masks_init].data = _subnet_linear_comb
                self.scores[self.num_masks_init].data[~final_mask | ~mask] = 0

                self.scores[idx].data[final_mask] = 0
                self.scores[self.num_masks_init - 1].data[mask] = 0

                # ensure right beta values
                # all masks that use the old mask must use the intersection
                self.betas[self.betas[:, idx] > 0, self.num_masks_init] = self.betas[self.betas[:, idx] > 0, idx]
                # current task must use the intersection
                self.betas[self.task, self.num_masks_init] = self.betas[self.task, self.num_masks_init - 1]
                # current task must use the old mask
                if self.betas[self.task, idx] <= 0:
                   self.betas[self.task, idx] = self.betas[self.task, self.num_masks_init - 1]
                
                self.num_masks_init += 1


        
    def __repr__(self):
        return f"MyMultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            self.num_masks_init += 1
            k = task + 1
            self.betas.data[task, 0:self.num_masks_init] = 1. / k

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
            self.fc_action = MyMultitaskMaskLinear(actor_body.feature_dim, action_dim, \
                discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
            self.fc_critic = MyMultitaskMaskLinear(critic_body.feature_dim, 1, \
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

class MyFCBody_SS(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM):
        super(MyFCBody_SS, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([MyMultitaskMaskLinear(dim_in, dim_out, discrete=discrete_mask, \
            num_tasks=num_tasks, new_mask_type=new_task_mask) \
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act