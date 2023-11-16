from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy
from ding.torch_utils import Adam, to_device
from ding.rl_utils import coma_data, coma_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate, timestep_collate
from .base_policy import Policy

@POLICY_REGISTRY.register('coma')
class COMAPolicy(Policy):
    """
    Overview:
        Policy class of COMA algorithm. COMA is a multi model reinforcement learning algorithm
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn\\
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval\\
            _reset_eval, _get_train_sample, default_model, _monitor_vars_learn
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      coma           | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     True           | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update``   int      1              | How many updates(iterations) to train  | this args can be vary
           | ``_per_collect``                           | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``  float    0.001          | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        9  | ``learn.td_``      float    0.8            | The trade-off factor of td-lambda,
           | ``lambda``                                 | which balances 1step td and mc
        10 | ``learn.value_``   float    1.0            | The loss weight of value network       | policy network weight
           | ``weight``                                                                          | is set to 1
        11 | ``learn.entropy_`` float    0.01           | The loss weight of entropy             | policy network weight
           | ``weight``                                 | regularization                         | is set to 1
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(type='coma', cuda=False, on_policy=False, priority=False, priority_IS_weight=False, learn=dict(update_per_collect=20, batch_size=32, learning_rate=0.0005, target_update_theta=0.001, discount_factor=0.99, td_lambda=0.8, policy_weight=0.001, value_weight=1, entropy_weight=0.01), collect=dict(unroll_len=20), eval=dict())

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Return this algorithm default model setting for demonstration.\n        Returns:\n            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names\n\n        .. note::\n            The user can define and use customized network model but must obey the same inferface definition indicated             by import_names path. For coma, ``ding.model.coma.coma``\n        '
        return ('coma', ['ding.model.template.coma'])

    def _init_learn(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Init the learner model of COMAPolicy\n\n        Arguments:\n            .. note::\n\n                The _init_learn method takes the argument from the self._cfg.learn in the config file\n\n            - learning_rate (:obj:`float`): The learning rate fo the optimizer\n            - gamma (:obj:`float`): The discount factor\n            - lambda (:obj:`float`): The lambda factor, determining the mix of bootstrapping                vs further accumulation of multistep returns at each timestep,\n            - value_wight(:obj:`float`): The weight of value loss in total loss\n            - entropy_weight(:obj:`float`): The weight of entropy loss in total loss\n            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.\n            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority, 'not implemented priority in COMA'
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor
        self._lambda = self._cfg.learn.td_lambda
        self._policy_weight = self._cfg.learn.policy_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_update_theta})
        self._target_model = model_wrap(self._target_model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        if False:
            return 10
        "\n        Overview:\n            Preprocess the data to fit the required data format for learning\n\n        Arguments:\n            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function, the Dict\n                in data should contain keys including at least ['obs', 'action', 'reward']\n\n        Returns:\n            - data (:obj:`Dict[str, Any]`): the processed data, including at least \\\n                ['obs', 'action', 'reward', 'done', 'weight']\n        "
        data = timestep_collate(data)
        assert set(data.keys()) > set(['obs', 'action', 'reward'])
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Forward and backward function of learn mode, acquire the data and calculate the loss and\\\n            optimize learner model\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \\\n                np.ndarray or dict/list combinations.\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \\\n                recorded in text log and tensorboard, values are python scalar or a list of scalars.\n        ArgumentsKeys:\n            - necessary: ``obs``, ``action``, ``reward``, ``done``, ``weight``\n        ReturnsKeys:\n            - necessary: ``cur_lr``, ``total_loss``, ``policy_loss``, ``value_loss``, ``entropy_loss``\n                - cur_lr (:obj:`float`): Current learning rate\n                - total_loss (:obj:`float`): The calculated loss\n                - policy_loss (:obj:`float`): The policy(actor) loss of coma\n                - value_loss (:obj:`float`): The value(critic) loss of coma\n                - entropy_loss (:obj:`float`): The entropy loss\n        '
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            target_q_value = self._target_model.forward(data, mode='compute_critic')['q_value']
        logit = self._learn_model.forward(data, mode='compute_actor')['logit']
        logit[data['obs']['action_mask'] == 0.0] = -9999999
        data = coma_data(logit, data['action'], q_value, target_q_value, data['reward'], data['weight'])
        coma_loss = coma_error(data, self._gamma, self._lambda)
        total_loss = self._policy_weight * coma_loss.policy_loss + self._value_weight * coma_loss.q_value_loss - self._entropy_weight * coma_loss.entropy_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr': self._optimizer.defaults['lr'], 'total_loss': total_loss.item(), 'policy_loss': coma_loss.policy_loss.item(), 'value_loss': coma_loss.q_value_loss.item(), 'entropy_loss': coma_loss.entropy_loss.item()}

    def _reset_learn(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'optimizer': self._optimizer.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Collect mode init moethod. Called by ``self.__init__``.\n            Init traj and unroll length, collect model.\n            Model has eps_greedy_sample wrapper and hidden state wrapper\n        '
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        if False:
            return 10
        '\n        Overview:\n            Collect output according to eps_greedy plugin\n\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            return 10
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Generate dict type transition data from inputs.\n        Arguments:\n            - obs (:obj:`Any`): Env observation\n            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']\n            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \\\n                (here 'obs' indicates obs after env step).\n        Returns:\n            - transition (:obj:`dict`): Dict type transition data.\n        "
        transition = {'obs': obs, 'next_obs': timestep.obs, 'prev_state': model_output['prev_state'], 'action': model_output['action'], 'reward': timestep.reward, 'done': timestep.done}
        return transition

    def _init_eval(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Evaluate mode init method. Called by ``self.__init__``.\n            Init eval model with argmax strategy and hidden_state plugin.\n        '
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num, save_prev_state=True, init_fn=lambda : [None for _ in range(self._cfg.model.agent_num)])
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Forward function of eval mode, similar to ``self._forward_collect``.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Get the train sample from trajectory\n\n        Arguments:\n            - data (:obj:`list`): The trajectory's cache\n\n        Returns:\n            - samples (:obj:`dict`): The training samples generated\n        "
        return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Return variables' name if variables are to used in monitor.\n        Returns:\n            - vars (:obj:`List[str]`): Variables' name list.\n        "
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']