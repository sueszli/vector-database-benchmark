from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import torch.nn.functional as F
import copy
from easydict import EasyDict
from ding.torch_utils import Adam, RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_epsilon_greedy_fn, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy

@POLICY_REGISTRY.register('qtran')
class QTRANPolicy(Policy):
    """
    Overview:
        Policy class of QTRAN algorithm. QTRAN is a multi model reinforcement learning algorithm,         you can view the paper in the following link https://arxiv.org/abs/1803.11485
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qtran          | RL policy register name, refer to      | this arg is optional,
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
        7  | ``learn.target_``  float    0.001          | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(type='qtran', cuda=True, on_policy=False, priority=False, priority_IS_weight=False, learn=dict(update_per_collect=20, batch_size=32, learning_rate=0.0005, clip_value=1.5, target_update_theta=0.008, discount_factor=0.99, td_weight=1, opt_weight=0.01, nopt_min_weight=0.0001, double_q=True), collect=dict(unroll_len=10), eval=dict(), other=dict(eps=dict(type='exp', start=1, end=0.05, decay=50000), replay_buffer=dict(replay_buffer_size=5000, max_reuse=1000000000.0, max_staleness=1000000000.0)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            return 10
        '\n        Overview:\n            Return this algorithm default model setting for demonstration.\n        Returns:\n            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names\n        .. note::\n            The user can define and use customized network model but must obey the same inferface definition indicated             by import_names path. For QTRAN, ``ding.model.qtran.qtran``\n        '
        return ('qtran', ['ding.model.template.qtran'])

    def _init_learn(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init the learner model of QTRANPolicy\n        Arguments:\n            .. note::\n\n                The _init_learn method takes the argument from the self._cfg.learn in the config file\n\n            - learning_rate (:obj:`float`): The learning rate fo the optimizer\n            - gamma (:obj:`float`): The discount factor\n            - agent_num (:obj:`int`): This is a multi-agent algorithm, we need to input agent num.\n            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and (not self._priority_IS_weight), 'Priority is not implemented in QTRAN'
        self._optimizer = RMSprop(params=self._model.parameters(), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=1e-05)
        self._gamma = self._cfg.learn.discount_factor
        self._td_weight = self._cfg.learn.td_weight
        self._opt_weight = self._cfg.learn.opt_weight
        self._nopt_min_weight = self._cfg.learn.nopt_min_weight
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_update_theta})
        self._target_model = model_wrap(self._target_model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Preprocess the data to fit the required data format for learning\n        Arguments:\n            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function\n        Returns:\n            - data (:obj:`Dict[str, Any]`): the processed data, from \\\n                [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}\n        '
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \\\n                np.ndarray or dict/list combinations.\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \\\n                recorded in text log and tensorboard, values are python scalar or a list of scalars.\n        ArgumentsKeys:\n            - necessary: ``obs``, ``next_obs``, ``action``, ``reward``, ``weight``, ``prev_state``, ``done``\n        ReturnsKeys:\n            - necessary: ``cur_lr``, ``total_loss``\n                - cur_lr (:obj:`float`): Current learning rate\n                - total_loss (:obj:`float`): The calculated loss\n        '
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        learn_ret = self._learn_model.forward(inputs, single_step=False)
        total_q = learn_ret['total_q']
        vs = learn_ret['vs']
        agent_q_act = learn_ret['agent_q_act']
        logit_detach = learn_ret['logit'].clone()
        logit_detach[data['obs']['action_mask'] == 0.0] = -9999999
        (logit_q, logit_action) = logit_detach.max(dim=-1, keepdim=False)
        if self._cfg.learn.double_q:
            next_inputs = {'obs': data['next_obs']}
            double_q_detach = self._learn_model.forward(next_inputs, single_step=False)['logit'].clone().detach()
            (_, double_q_action) = double_q_detach.max(dim=-1, keepdim=False)
            next_inputs = {'obs': data['next_obs'], 'action': double_q_action}
        else:
            next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, single_step=False)['total_q']
        td_data = v_1step_td_data(total_q, target_total_q.detach(), data['reward'], data['done'], data['weight'])
        (td_loss, td_error_per_sample) = v_1step_td_error(td_data, self._gamma)
        if data['weight'] is None:
            weight = torch.ones_like(data['reward'])
        opt_inputs = {'obs': data['obs'], 'action': logit_action}
        max_q = self._learn_model.forward(opt_inputs, single_step=False)['total_q']
        opt_error = logit_q.sum(dim=2) - max_q.detach() + vs
        opt_loss = (opt_error ** 2 * weight).mean()
        nopt_values = agent_q_act.sum(dim=2) - total_q.detach() + vs
        nopt_error = nopt_values.clamp(max=0)
        nopt_min_loss = (nopt_error ** 2 * weight).mean()
        total_loss = self._td_weight * td_loss + self._opt_weight * opt_loss + self._nopt_min_weight * nopt_min_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), 10000000)
        self._optimizer.step()
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr': self._optimizer.defaults['lr'], 'total_loss': total_loss.item(), 'td_loss': td_loss.item(), 'opt_loss': opt_loss.item(), 'nopt_loss': nopt_min_loss.item(), 'grad_norm': grad_norm}

    def _reset_learn(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset learn model to the state indicated by data_id\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\\\n                the model state to the state indicated by data_id\n        '
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return the state_dict of learn mode, usually including model and optimizer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.\n        '
        return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'optimizer': self._optimizer.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Load the state_dict variable into policy learn mode.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in \\\n            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \\\n            complicated operation.\n        '
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Collect mode init method. Called by ``self.__init__``.\n            Init traj and unroll length, collect model.\n            Enable the eps_greedy_sample and the hidden_state plugin.\n        '
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Forward function for collect mode with eps_greedy\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset collect model to the state indicated by data_id\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\\\n                the model state to the state indicated by data_id\n        '
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Generate dict type transition data from inputs.\n        Arguments:\n            - obs (:obj:`Any`): Env observation\n            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']\n            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\\\n                (here 'obs' indicates obs after env step).\n        Returns:\n            - transition (:obj:`dict`): Dict type transition data, including 'obs', 'next_obs', 'prev_state',\\\n                'action', 'reward', 'done'\n        "
        transition = {'obs': obs, 'next_obs': timestep.obs, 'prev_state': model_output['prev_state'], 'action': model_output['action'], 'reward': timestep.reward, 'done': timestep.done}
        return transition

    def _init_eval(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Evaluate mode init method. Called by ``self.__init__``.\n            Init eval model with argmax strategy and the hidden_state plugin.\n        '
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num, save_prev_state=True, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        if False:
            return 10
        '\n        Overview:\n            Forward function of eval mode, similar to ``self._forward_collect``.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Reset eval model to the state indicated by data_id\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\\\n                the model state to the state indicated by data_id\n        '
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Get the train sample from trajectory.\n        Arguments:\n            - data (:obj:`list`): The trajectory's cache\n        Returns:\n            - samples (:obj:`dict`): The training samples generated\n        "
        return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Return variables' name if variables are to used in monitor.\n        Returns:\n            - vars (:obj:`List[str]`): Variables' name list.\n        "
        return ['cur_lr', 'total_loss', 'td_loss', 'opt_loss', 'nopt_loss', 'grad_norm']