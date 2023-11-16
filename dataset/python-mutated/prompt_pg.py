from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
from ding.rl_utils import get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from ..model import model_wrap

@POLICY_REGISTRY.register('prompt_pg')
class PromptPGPolicy(Policy):
    """
    Overview:
        Policy class of Prompt Policy Gradient (PromptPG) algorithm.
        Link of the original paper: https://arxiv.org/abs/2209.14610
    """
    config = dict(type='prompt_pg', cuda=True, on_policy=True, deterministic_eval=True, learn=dict(batch_size=64, learning_rate=0.001, entropy_weight=0.01, grad_norm=5, ignore_done=False), collect=dict(unroll_len=1, discount_factor=0, collector=dict(get_train_sample=True)), eval=dict())

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            return 10
        return ('language_transformer', ['ding.model.template.language_transformer'])

    def _init_learn(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init the optimizer, algorithm config, main and target models.\n        '
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._grad_norm = self._cfg.learn.grad_norm
        self._learn_model = self._model

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward']\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.\n        "
        self._model.train()
        return_infos = []
        for i in range(0, len(data), self._cfg.learn.batch_size):
            batch = default_collate(data[i:i + self._cfg.learn.batch_size])
            if self._cuda:
                batch = to_device(batch, self._device)
            (train_samples, cand_samples) = (batch['obs']['train_sample'], batch['obs']['candidate_samples'])
            for ii in range(len(cand_samples)):
                cand_samples[ii] = cand_samples[ii][0]
            output = self._learn_model.forward(train_samples, cand_samples)
            return_ = batch['return']
            real_act = batch['action']
            (total_policy_loss, total_entropy_loss) = (0, 0)
            for ii in range(self._cfg.shot_number):
                log_prob = output['dist'].log_prob(real_act[:, ii])
                policy_loss = -(log_prob * return_).mean()
                total_policy_loss += policy_loss
            total_entropy_loss += -self._cfg.learn.entropy_weight * output['dist'].entropy().mean()
            total_loss = total_entropy_loss + total_policy_loss
            self._optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self._learn_model.parameters()), max_norm=self._grad_norm)
            self._optimizer.step()
            return_info = {'cur_lr': self._optimizer.param_groups[0]['lr'], 'total_loss': total_loss.item(), 'policy_loss': total_policy_loss.item(), 'entropy_loss': total_entropy_loss.item(), 'return_abs_max': return_.abs().max().item(), 'grad_norm': grad_norm}
            return_infos.append(return_info)
        return return_infos

    def _init_collect(self) -> None:
        if False:
            while True:
                i = 10
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.collect.discount_factor
        self._collect_model = model_wrap(self._model, wrapper_name='combination_multinomial_sample')

    def _forward_collect(self, data: dict) -> dict:
        if False:
            i = 10
            return i + 15
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            for ii in range(len(data['candidate_samples'])):
                data['candidate_samples'][ii] = data['candidate_samples'][ii][0]
            output = self._collect_model.forward(self._cfg.shot_number, data['train_sample'], data['candidate_samples'])
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Generate dict type transition data from inputs.\n        Arguments:\n            - obs (:obj:`Any`): Env observation\n            - model_output (:obj:`dict`): Output of collect model, including at least ['action']\n            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \\\n                (here 'obs' indicates obs after env step).\n        Returns:\n            - transition (:obj:`dict`): Dict type transition data.\n        "
        return {'obs': obs, 'action': model_output['action'], 'reward': timestep.reward, 'done': timestep.done}

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Get the trajectory and the n step return data, then sample from the n_step return data\n        Arguments:\n            - data (:obj:`list`): The trajectory's buffer list\n        Returns:\n            - samples (:obj:`dict`): The training samples generated\n        "
        if self._cfg.learn.ignore_done:
            raise NotImplementedError
        R = 0.0
        for i in reversed(range(len(data))):
            R = self._gamma * R + data[i]['reward']
            data[i]['return'] = R
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        if False:
            while True:
                i = 10
        self._eval_model = model_wrap(self._model, wrapper_name='combination_argmax_sample')

    def _forward_eval(self, data: dict) -> dict:
        if False:
            print('Hello World!')
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            for ii in range(len(data['candidate_samples'])):
                data['candidate_samples'][ii] = data['candidate_samples'][ii][0]
            output = self._eval_model.forward(self._cfg.shot_number, data['train_sample'], data['candidate_samples'])
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            print('Hello World!')
        return super()._monitor_vars_learn() + ['policy_loss', 'entropy_loss', 'return_abs_max', 'grad_norm']