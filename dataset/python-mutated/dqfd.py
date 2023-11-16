from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch
from torch.optim import AdamW
from ding.torch_utils import Adam, to_device
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample, dqfd_nstep_td_error, dqfd_nstep_td_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn
from copy import deepcopy

@POLICY_REGISTRY.register('dqfd')
class DQFDPolicy(DQNPolicy):
    """
    Overview:
        Policy class of DQFD algorithm, extended by Double DQN/Dueling DQN/PER/multi-step TD.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     True           | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     True           | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      10,            | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``lambda1``        float    1              | multiplicative factor for n-step
        9  | ``lambda2``        float    1              | multiplicative factor for the
                                                        | supervised margin loss
        10 | ``lambda3``        float    1e-5           | L2 loss
        11 | ``margin_fn``      float    0.8            | margin function in JE, here we set
                                                        | this as a constant
        12 | ``per_train_``     int      10             | number of pertraining iterations
           | ``iter_k``
        13 | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        14 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        15 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        16 | ``learn.target_``  int      100            | Frequency of target network update.    | Hard(assign) update
           | ``update_freq``
        17 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        18 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        19 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(type='dqfd', cuda=False, on_policy=False, priority=True, priority_IS_weight=True, discount_factor=0.99, nstep=10, learn=dict(lambda1=1.0, lambda2=1.0, lambda3=1e-05, margin_function=0.8, per_train_iter_k=10, update_per_collect=3, batch_size=64, learning_rate=0.001, target_update_freq=100, ignore_done=False), collect=dict(unroll_len=1, pho=0.5), eval=dict(), other=dict(eps=dict(type='exp', start=0.95, end=0.1, decay=10000), replay_buffer=dict(replay_buffer_size=10000)))

    def _init_learn(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``, initialize the optimizer, algorithm arguments, main             and target model.\n        '
        self.lambda1 = self._cfg.learn.lambda1
        self.lambda2 = self._cfg.learn.lambda2
        self.lambda3 = self._cfg.learn.lambda3
        self.margin_function = self._cfg.learn.margin_function
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer = AdamW(self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self.lambda3)
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='assign', update_kwargs={'freq': self._cfg.learn.target_update_freq})
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Forward computation graph of learn mode(updating policy).\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or                 np.ndarray or dict/list combinations.\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be                 recorded in text log and tensorboard, values are python scalar or a list of scalars.\n        ArgumentsKeys:\n            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``\n            - optional: ``value_gamma``, ``IS``\n        ReturnsKeys:\n            - necessary: ``cur_lr``, ``total_loss``, ``priority``\n            - optional: ``action_distribution``\n        '
        data = default_preprocess_learn(data, use_priority=self._priority, use_priority_IS_weight=self._cfg.priority_IS_weight, ignore_done=self._cfg.learn.ignore_done, use_nstep=True)
        data['done_1'] = data['done_1'].float()
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._target_model.train()
        q_value = self._learn_model.forward(data['obs'])['logit']
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            target_q_value_one_step = self._target_model.forward(data['next_obs_1'])['logit']
            target_q_action = self._learn_model.forward(data['next_obs'])['action']
            target_q_action_one_step = self._learn_model.forward(data['next_obs_1'])['action']
        is_expert = data['is_expert'].float()
        data_n = dqfd_nstep_td_data(q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['done_1'], data['weight'], target_q_value_one_step, target_q_action_one_step, is_expert)
        value_gamma = data.get('value_gamma')
        (loss, td_error_per_sample, loss_statistics) = dqfd_nstep_td_error(data_n, self._gamma, self.lambda1, self.lambda2, self.margin_function, nstep=self._nstep, value_gamma=value_gamma)
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr': self._optimizer.defaults['lr'], 'total_loss': loss.item(), 'priority': td_error_per_sample.abs().tolist()}

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that             can be used for training directly. A train sample can be a processed transition(DQN with nstep TD)             or some continuous transitions(DRQN).\n        Arguments:\n            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same                 format as the return value of ``self._process_transition`` method.\n        Returns:\n            - samples (:obj:`dict`): The list of training samples.\n\n        .. note::\n            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version.             And the user can customize the this data processing procecure by overriding this two methods and collector             itself.\n        '
        data_1 = deepcopy(get_nstep_return_data(data, 1, gamma=self._gamma))
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        for i in range(len(data)):
            data[i]['next_obs_1'] = data_1[i]['next_obs']
            data[i]['done_1'] = data_1[i]['done']
        return get_train_sample(data, self._unroll_len)