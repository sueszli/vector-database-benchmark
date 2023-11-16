from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy
from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy
from ding.policy.qmix import QMIXPolicy

@POLICY_REGISTRY.register('wqmix')
class WQMIXPolicy(QMIXPolicy):
    """
    Overview:
        Policy class of WQMIX algorithm. WQMIX is a reinforcement learning algorithm modified from Qmix, \\
            you can view the paper in the following link https://arxiv.org/abs/2006.10800
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn\\
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval\\
            _reset_eval, _get_train_sample, default_model
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qmix           | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     True           | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update_``  int      20             | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``   float    0.001         | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(type='wqmix', cuda=True, on_policy=False, priority=False, priority_IS_weight=False, learn=dict(update_per_collect=20, batch_size=32, learning_rate=0.0005, clip_value=100, target_update_theta=0.008, discount_factor=0.99, w=0.5, wqmix_ow=True), collect=dict(unroll_len=10), eval=dict(), other=dict(eps=dict(type='exp', start=1, end=0.05, decay=50000), replay_buffer=dict(replay_buffer_size=5000, max_reuse=1000000000.0, max_staleness=1000000000.0)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return this algorithm default model setting for demonstration.\n        Returns:\n            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names\n        .. note::\n            The user can define and use customized network model but must obey the same inferface definition indicated             by import_names path. For WQMIX, ``ding.model.template.wqmix``\n        '
        return ('wqmix', ['ding.model.template.wqmix'])

    def _init_learn(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init the learner model of WQMIXPolicy\n        Arguments:\n            .. note::\n\n                The _init_learn method takes the argument from the self._cfg.learn in the config file\n\n            - learning_rate (:obj:`float`): The learning rate fo the optimizer\n            - gamma (:obj:`float`): The discount factor\n            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.\n            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and (not self._priority_IS_weight), 'Priority is not implemented in WQMIX'
        self._optimizer = RMSprop(params=list(self._model._q_network.parameters()) + list(self._model._mixer.parameters()), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=1e-05)
        self._gamma = self._cfg.learn.discount_factor
        self._optimizer_star = RMSprop(params=list(self._model._q_network_star.parameters()) + list(self._model._mixer_star.parameters()), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=1e-05)
        self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        if False:
            return 10
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
        inputs = {'obs': data['obs'], 'action': data['action']}
        self._learn_model.reset(state=data['prev_state'][0])
        total_q = self._learn_model.forward(inputs, single_step=False, q_star=False)['total_q']
        self._learn_model.reset(state=data['prev_state'][0])
        total_q_star = self._learn_model.forward(inputs, single_step=False, q_star=True)['total_q']
        next_inputs = {'obs': data['next_obs']}
        self._learn_model.reset(state=data['prev_state'][1])
        next_logit_detach = self._learn_model.forward(next_inputs, single_step=False, q_star=False)['logit'].clone().detach()
        next_inputs = {'obs': data['next_obs'], 'action': next_logit_detach.argmax(dim=-1)}
        with torch.no_grad():
            self._learn_model.reset(state=data['prev_state'][1])
            target_total_q = self._learn_model.forward(next_inputs, single_step=False, q_star=True)['total_q']
        with torch.no_grad():
            if data['done'] is not None:
                target_v = self._gamma * (1 - data['done']) * target_total_q + data['reward']
            else:
                target_v = self._gamma * target_total_q + data['reward']
        td_error = (total_q - target_v).clone().detach()
        data_ = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        (_, td_error_per_sample) = v_1step_td_error(data_, self._gamma)
        data_star = v_1step_td_data(total_q_star, target_total_q, data['reward'], data['done'], data['weight'])
        (loss_star, td_error_per_sample_star_) = v_1step_td_error(data_star, self._gamma)
        alpha_to_use = self._cfg.learn.alpha
        if self._cfg.learn.wqmix_ow:
            ws = torch.full_like(td_error, alpha_to_use)
            ws = torch.where(td_error < 0, torch.ones_like(td_error), ws)
        else:
            inputs = {'obs': data['obs']}
            self._learn_model.reset(state=data['prev_state'][0])
            logit_detach = self._learn_model.forward(inputs, single_step=False, q_star=False)['logit'].clone().detach()
            cur_max_actions = logit_detach.argmax(dim=-1)
            inputs = {'obs': data['obs'], 'action': cur_max_actions}
            self._learn_model.reset(state=data['prev_state'][0])
            max_action_qtot = self._learn_model.forward(inputs, single_step=False, q_star=True)['total_q']
            is_max_action = (data['action'] == cur_max_actions).min(dim=2)[0]
            qtot_larger = target_v > max_action_qtot
            ws = torch.full_like(td_error, alpha_to_use)
            ws = torch.where(is_max_action | qtot_larger, torch.ones_like(td_error), ws)
        if data['weight'] is None:
            data['weight'] = torch.ones_like(data['reward'])
        loss_weighted = (ws.detach() * td_error_per_sample * data['weight']).mean()
        self._optimizer.zero_grad()
        self._optimizer_star.zero_grad()
        loss_weighted.backward(retain_graph=True)
        loss_star.backward()
        grad_norm_q = torch.nn.utils.clip_grad_norm_(list(self._model._q_network.parameters()) + list(self._model._mixer.parameters()), self._cfg.learn.clip_value)
        grad_norm_q_star = torch.nn.utils.clip_grad_norm_(list(self._model._q_network_star.parameters()) + list(self._model._mixer_star.parameters()), self._cfg.learn.clip_value)
        self._optimizer.step()
        self._optimizer_star.step()
        return {'cur_lr': self._optimizer.defaults['lr'], 'total_loss': loss_weighted.item(), 'total_q': total_q.mean().item() / self._cfg.model.agent_num, 'target_reward_total_q': target_v.mean().item() / self._cfg.model.agent_num, 'target_total_q': target_total_q.mean().item() / self._cfg.model.agent_num, 'grad_norm_q': grad_norm_q, 'grad_norm_q_star': grad_norm_q_star}

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return the state_dict of learn mode, usually including model and optimizer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.\n        '
        return {'model': self._learn_model.state_dict(), 'optimizer': self._optimizer.state_dict(), 'optimizer_star': self._optimizer_star.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Load the state_dict variable into policy learn mode.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in \\\n            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \\\n            complicated operation.\n        '
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._optimizer_star.load_state_dict(state_dict['optimizer_star'])