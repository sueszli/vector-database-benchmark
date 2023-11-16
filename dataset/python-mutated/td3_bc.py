from typing import List, Dict, Any, Tuple, Union
from easydict import EasyDict
from collections import namedtuple
import torch
import torch.nn.functional as F
import copy
from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from .ddpg import DDPGPolicy

@POLICY_REGISTRY.register('td3_bc')
class TD3BCPolicy(DDPGPolicy):
    """
    Overview:
        Policy class of TD3_BC algorithm.

        Since DDPG and TD3 share many common things, we can easily derive this TD3_BC
        class from DDPG class by changing ``_actor_update_freq``, ``_twin_critic`` and noise in model wrapper.

        https://arxiv.org/pdf/2106.06860.pdf

    Property:
        learn_mode, collect_mode, eval_mode

    Config:

    == ====================  ========    ==================  =================================   =======================
    ID Symbol                Type        Default Value       Description                         Other(Shape)
    == ====================  ========    ==================  =================================   =======================
    1  ``type``              str         td3_bc              | RL policy register name, refer    | this arg is optional,
                                                             | to registry ``POLICY_REGISTRY``   | a placeholder
    2  ``cuda``              bool        True                | Whether to use cuda for network   |
    3  | ``random_``         int         25000               | Number of randomly collected      | Default to 25000 for
       | ``collect_size``                                    | training samples in replay        | DDPG/TD3, 10000 for
       |                                                     | buffer when training starts.      | sac.
    4  | ``model.twin_``     bool        True                | Whether to use two critic         | Default True for TD3,
       | ``critic``                                          | networks or only one.             | Clipped Double
       |                                                     |                                   | Q-learning method in
       |                                                     |                                   | TD3 paper.
    5  | ``learn.learning``  float       1e-3                | Learning rate for actor           |
       | ``_rate_actor``                                     | network(aka. policy).             |
    6  | ``learn.learning``  float       1e-3                | Learning rates for critic         |
       | ``_rate_critic``                                    | network (aka. Q-network).         |
    7  | ``learn.actor_``    int         2                   | When critic network updates       | Default 2 for TD3, 1
       | ``update_freq``                                     | once, how many times will actor   | for DDPG. Delayed
       |                                                     | network update.                   | Policy Updates method
       |                                                     |                                   | in TD3 paper.
    8  | ``learn.noise``     bool        True                | Whether to add noise on target    | Default True for TD3,
       |                                                     | network's action.                 | False for DDPG.
       |                                                     |                                   | Target Policy Smoo-
       |                                                     |                                   | thing Regularization
       |                                                     |                                   | in TD3 paper.
    9  | ``learn.noise_``    dict        | dict(min=-0.5,    | Limit for range of target         |
       | ``range``                       |      max=0.5,)    | policy smoothing noise,           |
       |                                 |                   | aka. noise_clip.                  |
    10 | ``learn.-``         bool        False               | Determine whether to ignore       | Use ignore_done only
       | ``ignore_done``                                     | done flag.                        | in halfcheetah env.
    11 | ``learn.-``         float       0.005               | Used for soft update of the       | aka. Interpolation
       | ``target_theta``                                    | target network.                   | factor in polyak aver
       |                                                     |                                   | aging for target
       |                                                     |                                   | networks.
    12 | ``collect.-``       float       0.1                 | Used for add noise during co-     | Sample noise from dis
       | ``noise_sigma``                                     | llection, through controlling     | tribution, Ornstein-
       |                                                     | the sigma of distribution         | Uhlenbeck process in
       |                                                     |                                   | DDPG paper, Guassian
       |                                                     |                                   | process in ours.
    == ====================  ========    ==================  =================================   =======================
   """
    config = dict(type='td3_bc', cuda=False, on_policy=False, priority=False, priority_IS_weight=False, random_collect_size=25000, reward_batch_norm=False, action_space='continuous', model=dict(twin_critic=True, action_space='regression', actor_head_hidden_size=256, critic_head_hidden_size=256), learn=dict(update_per_collect=1, batch_size=256, learning_rate_actor=0.001, learning_rate_critic=0.001, ignore_done=False, target_theta=0.005, discount_factor=0.99, actor_update_freq=2, noise=True, noise_sigma=0.2, noise_range=dict(min=-0.5, max=0.5), alpha=2.5), collect=dict(unroll_len=1, noise_sigma=0.1), eval=dict(evaluator=dict(eval_freq=5000)), other=dict(replay_buffer=dict(replay_buffer_size=1000000)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            print('Hello World!')
        return ('continuous_qac', ['ding.model.template.qac'])

    def _init_learn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``. Init actor and critic optimizers, algorithm config.\n        '
        super(TD3BCPolicy, self)._init_learn()
        self._alpha = self._cfg.learn.alpha
        self._optimizer_actor = Adam(self._model.actor.parameters(), lr=self._cfg.learn.learning_rate_actor, grad_clip_type='clip_norm', clip_value=1.0)
        self._optimizer_critic = Adam(self._model.critic.parameters(), lr=self._cfg.learn.learning_rate_critic, grad_clip_type='clip_norm', clip_value=1.0)
        self.noise_sigma = self._cfg.learn.noise_sigma
        self.noise_range = self._cfg.learn.noise_range

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.\n        "
        loss_dict = {}
        data = default_preprocess_learn(data, use_priority=self._cfg.priority, use_priority_IS_weight=self._cfg.priority_IS_weight, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._target_model.train()
        next_obs = data['next_obs']
        reward = data['reward']
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-08)
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        q_value_dict = {}
        if self._twin_critic:
            q_value_dict['q_value'] = q_value[0].mean()
            q_value_dict['q_value_twin'] = q_value[1].mean()
        else:
            q_value_dict['q_value'] = q_value.mean()
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            noise = (torch.randn_like(next_action) * self.noise_sigma).clamp(self.noise_range['min'], self.noise_range['max'])
            next_action = (next_action + noise).clamp(-1, 1)
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            target_q_value = torch.min(target_q_value[0], target_q_value[1])
            td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
            (critic_loss, td_error_per_sample1) = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
            td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
            (critic_twin_loss, td_error_per_sample2) = v_1step_td_error(td_data_twin, self._gamma)
            loss_dict['critic_twin_loss'] = critic_twin_loss
            td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
        else:
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
            (critic_loss, td_error_per_sample) = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            if self._twin_critic:
                q_value = self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0]
                actor_loss = -q_value.mean()
            else:
                q_value = self._learn_model.forward(actor_data, mode='compute_critic')['q_value']
                actor_loss = -q_value.mean()
            lmbda = self._alpha / q_value.abs().mean().detach()
            bc_loss = F.mse_loss(actor_data['action'], data['action'])
            actor_loss = lmbda * actor_loss + bc_loss
            loss_dict['actor_loss'] = actor_loss
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr_actor': self._optimizer_actor.defaults['lr'], 'cur_lr_critic': self._optimizer_critic.defaults['lr'], 'action': data.get('action').mean(), 'priority': td_error_per_sample.abs().tolist(), 'td_error': td_error_per_sample.abs().mean(), **loss_dict, **q_value_dict}

    def _forward_eval(self, data: dict) -> dict:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Forward function of eval mode, similar to ``self._forward_collect``.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.\n        ReturnsKeys\n            - necessary: ``action``\n            - optional: ``logit``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}