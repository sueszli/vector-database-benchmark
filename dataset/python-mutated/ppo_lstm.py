from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
from ding.torch_utils import Adam, to_device
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
from ding.policy.common_utils import default_preprocess_learn
from ding.policy.command_mode_policy_instance import DummyCommandModePolicy

@POLICY_REGISTRY.register('ppo_lstm')
class PPOPolicy(Policy):
    """
    Overview:
        Policy class of PPO algorithm.
    """
    config = dict(type='ppo_lstm', cuda=False, on_policy=True, priority=False, priority_IS_weight=False, nstep_return=False, nstep=3, learn=dict(update_per_collect=5, batch_size=64, learning_rate=0.001, value_weight=0.5, entropy_weight=0.01, clip_ratio=0.2, adv_norm=False, ignore_done=False), collect=dict(unroll_len=1, discount_factor=0.99, gae_lambda=0.95), eval=dict(), other=dict(replay_buffer=dict(replay_buffer_size=1000)))

    def _init_learn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init the optimizer, algorithm config and the main model.\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and (not self._priority_IS_weight), 'Priority is not implemented in PPO'
        for m in self._model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight)
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return
        self._learn_model.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`dict`): Dict type data\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`):\n              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \\\n                        adv_abs_max, approx_kl, clipfrac\n        '
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=self._nstep_return)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        if not self._nstep_return:
            output = self._learn_model.forward(data['obs'])
            adv = data['adv']
            if self._adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-08)
            return_ = data['value'] + adv
            ppodata = ppo_data(output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_, data['weight'])
            (ppo_loss, ppo_info) = ppo_error(ppodata, self._clip_ratio)
            (wv, we) = (self._value_weight, self._entropy_weight)
            total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss
        else:
            output = self._learn_model.forward(data['obs'])
            adv = data['adv']
            if self._adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-08)
            ppodata = ppo_policy_data(output['logit'], data['logit'], data['action'], adv, data['weight'])
            (ppo_policy_loss, ppo_info) = ppo_policy_error(ppodata, self._clip_ratio)
            (wv, we) = (self._value_weight, self._entropy_weight)
            next_obs = data.get('next_obs')
            value_gamma = data.get('value_gamma')
            reward = data.get('reward')
            value = self._learn_model.forward(data['obs'])
            next_data = {'obs': next_obs}
            target_value = self._learn_model.forward(next_data['obs'])
            assert self._nstep > 1
            td_data = v_nstep_td_data(value['value'], target_value['value'], reward.t(), data['done'], data['weight'], value_gamma)
            (critic_loss, td_error_per_sample) = v_nstep_td_error(td_data, self._gamma, self._nstep)
            ppo_loss_data = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
            ppo_loss = ppo_loss_data(ppo_policy_loss.policy_loss, critic_loss, ppo_policy_loss.entropy_loss)
            total_loss = ppo_policy_loss.policy_loss + wv * critic_loss - we * ppo_policy_loss.entropy_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {'cur_lr': self._optimizer.defaults['lr'], 'total_loss': total_loss.item(), 'policy_loss': ppo_loss.policy_loss.item(), 'value_loss': ppo_loss.value_loss.item(), 'entropy_loss': ppo_loss.entropy_loss.item(), 'adv_abs_max': adv.abs().max().item(), 'approx_kl': ppo_info.approx_kl, 'clipfrac': ppo_info.clipfrac}

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'model': self._learn_model.state_dict(), 'optimizer': self._optimizer.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Collect mode init method. Called by ``self.__init__``.\n            Init traj and unroll length, collect model.\n        '
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return

    def _forward_collect(self, data: dict) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Forward function of collect mode.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            while True:
                i = 10
        "\n        Overview:\n               Generate dict type transition data from inputs.\n        Arguments:\n                - obs (:obj:`Any`): Env observation\n                - model_output (:obj:`dict`): Output of collect model, including at least ['action']\n                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']                       (here 'obs' indicates obs after env step).\n        Returns:\n               - transition (:obj:`dict`): Dict type transition data.\n        "
        if not self._nstep_return:
            transition = {'obs': obs, 'logit': model_output['logit'], 'action': model_output['action'], 'value': model_output['value'], 'prev_state': model_output['prev_state'], 'reward': timestep.reward, 'done': timestep.done}
        else:
            transition = {'obs': obs, 'next_obs': timestep.obs, 'logit': model_output['logit'], 'action': model_output['action'], 'prev_state': model_output['prev_state'], 'value': model_output['value'], 'reward': timestep.reward, 'done': timestep.done}
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Get the trajectory and calculate GAE, return one data to cache for next time calculation\n        Arguments:\n            - data (:obj:`deque`): The trajectory's cache\n        Returns:\n            - samples (:obj:`dict`): The training samples generated\n        "
        data = get_gae_with_default_last_value(data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda, cuda=self._cuda)
        if not self._nstep_return:
            return get_train_sample(data, self._unroll_len)
        else:
            return get_nstep_return_data(data, self._nstep)

    def _init_eval(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Evaluate mode init method. Called by ``self.__init__``.\n            Init eval model with argmax strategy.\n        '
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Forward function of eval mode, similar to ``self._forward_collect``.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data[0])
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._eval_model.reset(data_id=data_id)

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        return ('vac', ['ding.model.template.vac'])

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            while True:
                i = 10
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac']

@POLICY_REGISTRY.register('ppo_lstm_command')
class PPOCommandModePolicy(PPOPolicy, DummyCommandModePolicy):
    pass