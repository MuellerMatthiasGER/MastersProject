import numpy as np

from deep_rl.agent.PPO_agent import LLAgent
from deep_rl.mask_modules.mmn.mask_nets import *

class MyLLAgent(LLAgent):
    def __init__(self, config):
        super().__init__(config)

    def task_eval_start(self, task_label):
        self.network.eval()
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            # agent has not been trained on current task
            # being evaluated. Therefore a linear combination 
            # of all previous masks is used
            task_idx = self._set_betas_for_unseen_eval_task()
        set_model_task(self.network, task_idx)
        self.curr_eval_task_label = task_label

    def _set_betas_for_unseen_eval_task(self):
        if len(self.seen_tasks) == 0 or len(self.seen_tasks) == 1:
            # Means that this is the initial eval run or
            # only one task has been trained so far.
            # In both cases, just use the first mask
            task_idx = 0
            return task_idx

        # generate a tmp task idx
        task_idx = len(self.seen_tasks)
        k = task_idx
        cur_trained_idx = task_idx - 1
        for n, m in self.network.named_modules():
            if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
                # set beta values for unseen tasks
                # Note that those beta values are before softmax.
                # Therefore, in order to only use past masks, the 
                # value of the mask for this next hypothetical task must be 
                # very small such that it will result in 0 in softmax.
                # The mask of the currently trained task has not been
                # consolidated yet. This must be accounted for.
                # Also to account for that the softmax function will be 
                # applied to that, log will be applied to the resulting 
                # prob vector will be
                cur_task_betas = m.betas.data[cur_trained_idx, 0:cur_trained_idx + 1]
                cur_task_betas_softmax = torch.softmax(cur_task_betas, dim=-1)
                
                new_betas = torch.zeros(task_idx + 1, device=cur_task_betas_softmax.device)
                new_betas[0:cur_trained_idx] = 1. / k
                new_betas[0:task_idx] += (1. / k) * cur_task_betas_softmax
                new_betas = torch.log(new_betas)
                m.betas.data[task_idx, 0:task_idx+1] = new_betas
                m.betas.data[task_idx] = torch.tensor(float('-inf'))
        return task_idx
