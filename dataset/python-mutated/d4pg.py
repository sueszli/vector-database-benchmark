from typing import List, Dict, Any, Tuple, Union
import torch
import copy
from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .ddpg import DDPGPolicy
from .common_utils import default_preprocess_learn
import numpy as np

@POLICY_REGISTRY.register('d4pg')
class D4PGPolicy(DDPGPolicy):
    """
    Overview:
        Policy class of D4PG algorithm.
    Property:
        learn_mode, collect_mode, eval_mode
    Config:
        == ====================  ========    =============  =================================   =======================
        ID Symbol                Type        Default Value  Description                         Other(Shape)
        == ====================  ========    =============  =================================   =======================
        1  ``type``              str         d4pg           | RL policy register name, refer    | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY``   | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network   |
        3  | ``random_``         int         25000          | Number of randomly collected      | Default to 25000 for
           | ``collect_size``                               | training samples in replay        | DDPG/TD3, 10000 for
           |                                                | buffer when training starts.      | sac.
        5  | ``learn.learning``  float       1e-3           | Learning rate for actor           |
           | ``_rate_actor``                                | network(aka. policy).             |
        6  | ``learn.learning``  float       1e-3           | Learning rates for critic         |
           | ``_rate_critic``                               | network (aka. Q-network).         |
        7  | ``learn.actor_``    int         1              | When critic network updates       | Default 1
           | ``update_freq``                                | once, how many times will actor   |
           |                                                | network update.                   |
        8  | ``learn.noise``     bool        False          | Whether to add noise on target    | Default False for
           |                                                | network's action.                 | D4PG.
           |                                                |                                   | Target Policy Smoo-
           |                                                |                                   | thing Regularization
           |                                                |                                   | in TD3 paper.
        9  | ``learn.-``         bool        False          | Determine whether to ignore       | Use ignore_done only
           | ``ignore_done``                                | done flag.                        | in halfcheetah env.
        10 | ``learn.-``         float       0.005          | Used for soft update of the       | aka. Interpolation
           | ``target_theta``                               | target network.                   | factor in polyak aver
           |                                                |                                   | aging for target
           |                                                |                                   | networks.
        11 | ``collect.-``       float       0.1            | Used for add noise during co-     | Sample noise from dis
           | ``noise_sigma``                                | llection, through controlling     | tribution, Gaussian
           |                                                | the sigma of distribution         | process.
        12 | ``model.v_min``      float      -10            | Value of the smallest atom        |
           |                                                | in the support set.               |
        13 | ``model.v_max``      float      10             | Value of the largest atom         |
           |                                                | in the support set.               |
        14 | ``model.n_atom``     int        51             | Number of atoms in the support    |
           |                                                | set of the value distribution.    |
        15 | ``nstep``            int        3, [1, 5]      | N-step reward discount sum for    |
           |                                                | target q_value estimation         |
        16 | ``priority``         bool       True           | Whether use priority(PER)         | priority sample,
                                                                                                | update priority
        == ====================  ========    =============  =================================   =======================
    """
    config = dict(type='d4pg', cuda=False, on_policy=False, priority=True, priority_IS_weight=True, random_collect_size=25000, nstep=3, action_space='continuous', reward_batch_norm=False, transition_with_policy_data=False, model=dict(v_min=-10, v_max=10, n_atom=51), learn=dict(update_per_collect=1, batch_size=256, learning_rate_actor=0.001, learning_rate_critic=0.001, ignore_done=False, target_theta=0.005, discount_factor=0.99, actor_update_freq=1, noise=False), collect=dict(unroll_len=1, noise_sigma=0.1), eval=dict(evaluator=dict(eval_freq=1000)), other=dict(replay_buffer=dict(replay_buffer_size=1000000)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            i = 10
            return i + 15
        return ('qac_dist', ['ding.model.template.qac_dist'])

    def _init_learn(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init actor and critic optimizers, algorithm config, main and target models.\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer_actor = Adam(self._model.actor.parameters(), lr=self._cfg.learn.learning_rate_actor)
        self._optimizer_critic = Adam(self._model.critic.parameters(), lr=self._cfg.learn.learning_rate_critic)
        self._reward_batch_norm = self._cfg.reward_batch_norm
        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.nstep
        self._actor_update_freq = self._cfg.learn.actor_update_freq
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_theta})
        if self._cfg.learn.noise:
            self._target_model = model_wrap(self._target_model, wrapper_name='action_noise', noise_type='gauss', noise_kwargs={'mu': 0.0, 'sigma': self._cfg.learn.noise_sigma}, noise_range=self._cfg.learn.noise_range)
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()
        self._v_max = self._cfg.model.v_max
        self._v_min = self._cfg.model.v_min
        self._n_atom = self._cfg.model.n_atom
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.\n        "
        loss_dict = {}
        data = default_preprocess_learn(data, use_priority=self._cfg.priority, use_priority_IS_weight=self._cfg.priority_IS_weight, ignore_done=self._cfg.learn.ignore_done, use_nstep=True)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._target_model.train()
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-08)
        q_value = self._learn_model.forward(data, mode='compute_critic')
        q_value_dict = {}
        q_dist = q_value['distribution']
        q_value_dict['q_value'] = q_value['q_value'].mean()
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_dist = self._target_model.forward(next_data, mode='compute_critic')['distribution']
        value_gamma = data.get('value_gamma')
        action_index = np.zeros(next_action.shape[0])
        td_data = dist_nstep_td_data(q_dist, target_q_dist, action_index, action_index, reward, data['done'], data['weight'])
        (critic_loss, td_error_per_sample) = dist_nstep_td_error(td_data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep, value_gamma=value_gamma)
        loss_dict['critic_loss'] = critic_loss
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()
            loss_dict['actor_loss'] = actor_loss
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr_actor': self._optimizer_actor.defaults['lr'], 'cur_lr_critic': self._optimizer_critic.defaults['lr'], 'q_value': q_value['q_value'].mean().item(), 'action': data['action'].mean().item(), 'priority': td_error_per_sample.abs().tolist(), **loss_dict, **q_value_dict}

    def _get_train_sample(self, traj: list) -> Union[None, List[Any]]:
        if False:
            print('Hello World!')
        "\n            Overview:\n                Get the trajectory and the n step return data, then sample from the n_step return data\n            Arguments:\n                - traj (:obj:`list`): The trajectory's buffer list\n            Returns:\n                - samples (:obj:`dict`): The training samples generated\n        "
        data = get_nstep_return_data(traj, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            return 10
        "\n        Overview:\n            Return variables' name if variables are to used in monitor.\n        Returns:\n            - vars (:obj:`List[str]`): Variables' name list.\n        "
        ret = ['cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'action']
        return ret