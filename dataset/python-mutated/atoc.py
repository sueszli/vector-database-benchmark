from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import torch
from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn

@POLICY_REGISTRY.register('atoc')
class ATOCPolicy(Policy):
    """
    Overview:
        Policy class of ATOC algorithm.
    Interface:
        __init__, set_setting, __repr__, state_dict_handle
    Property:
        learn_mode, collect_mode, eval_mode
    """
    config = dict(type='atoc', cuda=False, on_policy=False, priority=False, priority_IS_weight=False, model=dict(communication=True, thought_size=8, agent_per_group=2), learn=dict(update_per_collect=5, batch_size=64, learning_rate_actor=0.001, learning_rate_critic=0.001, target_theta=0.005, discount_factor=0.99, communication=True, actor_update_freq=1, noise=True, noise_sigma=0.15, noise_range=dict(min=-0.5, max=0.5), reward_batch_norm=False, ignore_done=False), collect=dict(unroll_len=1, noise_sigma=0.4), eval=dict(), other=dict(replay_buffer=dict(replay_buffer_size=100000, max_use=10)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            i = 10
            return i + 15
        return ('atoc', ['ding.model.template.atoc'])

    def _init_learn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init actor and critic optimizers, algorithm config, main and target models.\n        '
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and (not self._priority_IS_weight)
        self._communication = self._cfg.learn.communication
        self._gamma = self._cfg.learn.discount_factor
        self._actor_update_freq = self._cfg.learn.actor_update_freq
        self._optimizer_actor = Adam(self._model.actor.parameters(), lr=self._cfg.learn.learning_rate_actor)
        self._optimizer_critic = Adam(self._model.critic.parameters(), lr=self._cfg.learn.learning_rate_critic)
        if self._communication:
            self._optimizer_actor_attention = Adam(self._model.actor.attention.parameters(), lr=self._cfg.learn.learning_rate_actor)
        self._reward_batch_norm = self._cfg.learn.reward_batch_norm
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_theta})
        if self._cfg.learn.noise:
            self._target_model = model_wrap(self._target_model, wrapper_name='action_noise', noise_type='gauss', noise_kwargs={'mu': 0.0, 'sigma': self._cfg.learn.noise_sigma}, noise_range=self._cfg.learn.noise_range)
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Forward and backward function of learn mode.\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']\n        Returns:\n            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.\n        "
        loss_dict = {}
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._target_model.train()
        next_obs = data['next_obs']
        reward = data['reward']
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-08)
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
        td_data = v_1step_td_data(q_value.mean(-1), target_q_value.mean(-1), reward, data['done'], data['weight'])
        (critic_loss, td_error_per_sample) = v_1step_td_error(td_data, self._gamma)
        loss_dict['critic_loss'] = critic_loss
        self._optimizer_critic.zero_grad()
        critic_loss.backward()
        self._optimizer_critic.step()
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            if self._communication:
                output = self._learn_model.forward(data['obs'], mode='compute_actor', get_delta_q=False)
                output['delta_q'] = data['delta_q']
                attention_loss = self._learn_model.forward(output, mode='optimize_actor_attention')['loss']
                loss_dict['attention_loss'] = attention_loss
                self._optimizer_actor_attention.zero_grad()
                attention_loss.backward()
                self._optimizer_actor_attention.step()
            output = self._learn_model.forward(data['obs'], mode='compute_actor', get_delta_q=False)
            critic_input = {'obs': data['obs'], 'action': output['action']}
            actor_loss = -self._learn_model.forward(critic_input, mode='compute_critic')['q_value'].mean()
            loss_dict['actor_loss'] = actor_loss
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {'cur_lr_actor': self._optimizer_actor.defaults['lr'], 'cur_lr_critic': self._optimizer_critic.defaults['lr'], 'priority': td_error_per_sample.abs().tolist(), 'q_value': q_value.mean().item(), **loss_dict}

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'optimizer_actor': self._optimizer_actor.state_dict(), 'optimizer_critic': self._optimizer_critic.state_dict(), 'optimize_actor_attention': self._optimizer_actor_attention.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            return 10
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self._optimizer_critic.load_state_dict(state_dict['optimizer_critic'])
        self._optimizer_actor_attention.load_state_dict(state_dict['optimize_actor_attention'])

    def _init_collect(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Collect mode init method. Called by ``self.__init__``.\n            Init traj and unroll length, collect model.\n        '
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='action_noise', noise_type='gauss', noise_kwargs={'mu': 0.0, 'sigma': self._cfg.collect.noise_sigma}, noise_range=None)
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Forward function of collect mode.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.\n        ReturnsKeys\n            - necessary: ``action``\n        '
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor', get_delta_q=True)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        if False:
            return 10
        "\n        Overview:\n            Generate dict type transition data from inputs.\n        Arguments:\n            - obs (:obj:`Any`): Env observation\n            - model_output (:obj:`dict`): Output of collect model, including at least ['action']\n            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \\\n                (here 'obs' indicates obs after env step, i.e. next_obs).\n        Return:\n            - transition (:obj:`Dict[str, Any]`): Dict type transition data.\n        "
        if self._communication:
            transition = {'obs': obs, 'next_obs': timestep.obs, 'action': model_output['action'], 'delta_q': model_output['delta_q'], 'reward': timestep.reward, 'done': timestep.done}
        else:
            transition = {'obs': obs, 'next_obs': timestep.obs, 'action': model_output['action'], 'reward': timestep.reward, 'done': timestep.done}
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            print('Hello World!')
        if self._communication:
            delta_q_batch = [d['delta_q'] for d in data]
            delta_min = torch.stack(delta_q_batch).min()
            delta_max = torch.stack(delta_q_batch).max()
            for i in range(len(data)):
                data[i]['delta_q'] = (data[i]['delta_q'] - delta_min) / (delta_max - delta_min + 1e-08)
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Evaluate mode init method. Called by ``self.__init__``.\n            Init eval model. Unlike learn and collect model, eval model does not need noise.\n        '
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Forward function of eval mode, similar to ``self._forward_collect``.\n        Arguments:\n            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \\\n                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.\n        ReturnsKeys\n            - necessary: ``action``\n        '
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

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Return variables' name if variables are to used in monitor.\n        Returns:\n            - vars (:obj:`List[str]`): Variables' name list.\n        "
        return ['cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'attention_loss', 'total_loss', 'q_value']