from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy
from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, v_nstep_td_data, v_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .qmix import QMIXPolicy

@POLICY_REGISTRY.register('madqn')
class MADQNPolicy(QMIXPolicy):
    config = dict(type='madqn', cuda=True, on_policy=False, priority=False, priority_IS_weight=False, nstep=3, learn=dict(update_per_collect=20, batch_size=32, learning_rate=0.0005, clip_value=100, target_update_theta=0.008, discount_factor=0.99, double_q=False, weight_decay=1e-05), collect=dict(n_episode=32, unroll_len=10), eval=dict(), other=dict(eps=dict(type='exp', start=1, end=0.05, decay=50000), replay_buffer=dict(replay_buffer_size=5000, max_reuse=1000000000.0, max_staleness=1000000000.0)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return this algorithm default model setting for demonstration.\n        Returns:\n            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names\n        '
        return ('madqn', ['ding.model.template.madqn'])

    def _init_learn(self) -> None:
        if False:
            return 10
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and (not self._priority_IS_weight), 'Priority is not implemented in QMIX'
        self._optimizer_current = RMSprop(params=self._model.current.parameters(), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=1e-05, weight_decay=self._cfg.learn.weight_decay)
        self._optimizer_cooperation = RMSprop(params=self._model.cooperation.parameters(), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=1e-05, weight_decay=self._cfg.learn.weight_decay)
        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.nstep
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_update_theta})
        self._target_model = model_wrap(self._target_model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Preprocess the data to fit the required data format for learning\n        Arguments:\n            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function\n        Returns:\n            - data (:obj:`Dict[str, Any]`): the processed data, from \\\n                [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}\n        '
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \\\n                np.ndarray or dict/list combinations.\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \\\n                recorded in text log and tensorboard, values are python scalar or a list of scalars.\n        ArgumentsKeys:\n            - necessary: ``obs``, ``next_obs``, ``action``, ``reward``, ``weight``, ``prev_state``, ``done``\n        ReturnsKeys:\n            - necessary: ``cur_lr``, ``total_loss``\n                - cur_lr (:obj:`float`): Current learning rate\n                - total_loss (:obj:`float`): The calculated loss\n        '
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        total_q = self._learn_model.forward(inputs, single_step=False)['total_q']
        if self._cfg.learn.double_q:
            next_inputs = {'obs': data['next_obs']}
            self._learn_model.reset(state=data['prev_state'][1])
            logit_detach = self._learn_model.forward(next_inputs, single_step=False)['logit'].clone().detach()
            next_inputs = {'obs': data['next_obs'], 'action': logit_detach.argmax(dim=-1)}
        else:
            next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, cooperation=True, single_step=False)['total_q']
        if self._nstep == 1:
            v_data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
            (loss, td_error_per_sample) = v_1step_td_error(v_data, self._gamma)
            with torch.no_grad():
                if data['done'] is not None:
                    target_v = self._gamma * (1 - data['done']) * target_total_q + data['reward']
                else:
                    target_v = self._gamma * target_total_q + data['reward']
        else:
            data['reward'] = data['reward'].permute(0, 2, 1).contiguous()
            loss = []
            td_error_per_sample = []
            for t in range(self._cfg.collect.unroll_len):
                v_data = v_nstep_td_data(total_q[t], target_total_q[t], data['reward'][t], data['done'][t], data['weight'], self._gamma)
                (loss_i, td_error_per_sample_i) = v_nstep_td_error(v_data, self._gamma, self._nstep)
                loss.append(loss_i)
                td_error_per_sample.append(td_error_per_sample_i)
            loss = sum(loss) / (len(loss) + 1e-08)
            td_error_per_sample = sum(td_error_per_sample) / (len(td_error_per_sample) + 1e-08)
        self._optimizer_current.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.current.parameters(), self._cfg.learn.clip_value)
        self._optimizer_current.step()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        cooperation_total_q = self._learn_model.forward(inputs, cooperation=True, single_step=False)['total_q']
        next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            cooperation_target_total_q = self._target_model.forward(next_inputs, cooperation=True, single_step=False)['total_q']
        if self._nstep == 1:
            v_data = v_1step_td_data(cooperation_total_q, cooperation_target_total_q, data['reward'], data['done'], data['weight'])
            (cooperation_loss, _) = v_1step_td_error(v_data, self._gamma)
        else:
            cooperation_loss_all = []
            for t in range(self._cfg.collect.unroll_len):
                v_data = v_nstep_td_data(cooperation_total_q[t], cooperation_target_total_q[t], data['reward'][t], data['done'][t], data['weight'], self._gamma)
                (cooperation_loss, _) = v_nstep_td_error(v_data, self._gamma, self._nstep)
                cooperation_loss_all.append(cooperation_loss)
            cooperation_loss = sum(cooperation_loss_all) / (len(cooperation_loss_all) + 1e-08)
        self._optimizer_cooperation.zero_grad()
        cooperation_loss.backward()
        cooperation_grad_norm = torch.nn.utils.clip_grad_norm_(self._model.cooperation.parameters(), self._cfg.learn.clip_value)
        self._optimizer_cooperation.step()
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr': self._optimizer_current.defaults['lr'], 'total_loss': loss.item(), 'total_q': total_q.mean().item() / self._cfg.model.agent_num, 'target_total_q': target_total_q.mean().item() / self._cfg.model.agent_num, 'grad_norm': grad_norm, 'cooperation_grad_norm': cooperation_grad_norm, 'cooperation_loss': cooperation_loss.item()}

    def _reset_learn(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Reset learn model to the state indicated by data_id\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\\\n                the model state to the state indicated by data_id\n        '
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Return the state_dict of learn mode, usually including model and optimizer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.\n        '
        return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'optimizer_current': self._optimizer_current.state_dict(), 'optimizer_cooperation': self._optimizer_cooperation.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Load the state_dict variable into policy learn mode.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.\n\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in             load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more             complicated operation.\n        '
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_current.load_state_dict(state_dict['optimizer_current'])
        self._optimizer_cooperation.load_state_dict(state_dict['optimizer_cooperation'])

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Generate dict type transition data from inputs.\n        Arguments:\n            - obs (:obj:`Any`): Env observation\n            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']\n            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\\\n                (here 'obs' indicates obs after env step).\n        Returns:\n            - transition (:obj:`dict`): Dict type transition data, including 'obs', 'next_obs', 'prev_state',\\\n                'action', 'reward', 'done'\n        "
        transition = {'obs': obs, 'next_obs': timestep.obs, 'prev_state': model_output['prev_state'], 'action': model_output['action'], 'reward': timestep.reward, 'done': timestep.done}
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Get the train sample from trajectory.\n        Arguments:\n            - data (:obj:`list`): The trajectory's cache\n        Returns:\n            - samples (:obj:`dict`): The training samples generated\n        "
        if self._cfg.nstep == 1:
            return get_train_sample(data, self._unroll_len)
        else:
            data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
            return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Return variables' name if variables are to used in monitor.\n        Returns:\n            - vars (:obj:`List[str]`): Variables' name list.\n        "
        return ['cur_lr', 'total_loss', 'total_q', 'target_total_q', 'grad_norm', 'target_reward_total_q', 'cooperation_grad_norm', 'cooperation_loss']