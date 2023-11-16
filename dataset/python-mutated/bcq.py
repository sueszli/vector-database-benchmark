from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, get_nstep_return_data
from ding.model import model_wrap
from ding.policy import Policy
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn

@POLICY_REGISTRY.register('bcq')
class BCQPolicy(Policy):
    config = dict(type='bcq', cuda=False, priority=False, priority_IS_weight=False, random_collect_size=10000, nstep=1, model=dict(actor_head_hidden_size=[400, 300], critic_head_hidden_size=[400, 300], phi=0.05), learn=dict(update_per_collect=1, batch_size=100, learning_rate_q=0.0003, learning_rate_policy=0.0003, learning_rate_vae=0.0003, ignore_done=False, target_theta=0.005, discount_factor=0.99, lmbda=0.75, init_w=0.003), collect=dict(unroll_len=1), eval=dict(), other=dict(replay_buffer=dict(replay_buffer_size=1000000)))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            i = 10
            return i + 15
        return ('bcq', ['ding.model.template.bcq'])

    def _init_learn(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init q, value and policy's optimizers, algorithm config, main and target models.\n        "
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.lmbda = self._cfg.learn.lmbda
        self.latent_dim = self._cfg.model.action_shape * 2
        self._optimizer_q = Adam(self._model.critic.parameters(), lr=self._cfg.learn.learning_rate_q)
        self._optimizer_policy = Adam(self._model.actor.parameters(), lr=self._cfg.learn.learning_rate_policy)
        self._optimizer_vae = Adam(self._model.vae.parameters(), lr=self._cfg.learn.learning_rate_vae)
        self._gamma = self._cfg.learn.discount_factor
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(self._target_model, wrapper_name='target', update_type='momentum', update_kwargs={'theta': self._cfg.learn.target_theta})
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            return 10
        loss_dict = {}
        data = default_preprocess_learn(data, use_priority=self._priority, use_priority_IS_weight=self._cfg.priority_IS_weight, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if len(data.get('action').shape) == 1:
            data['action'] = data['action'].reshape(-1, 1)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        batch_size = obs.shape[0]
        vae_out = self._model.forward(data, mode='compute_vae')
        (recon, mean, log_std) = (vae_out['recons_action'], vae_out['mu'], vae_out['log_var'])
        recons_loss = F.mse_loss(recon, data['action'])
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_std - mean ** 2 - log_std.exp(), dim=1), dim=0)
        loss_dict['recons_loss'] = recons_loss
        loss_dict['kld_loss'] = kld_loss
        vae_loss = recons_loss + 0.5 * kld_loss
        loss_dict['vae_loss'] = vae_loss
        self._optimizer_vae.zero_grad()
        vae_loss.backward()
        self._optimizer_vae.step()
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            next_obs_rep = torch.repeat_interleave(next_obs, 10, 0)
            z = torch.randn((next_obs_rep.shape[0], self.latent_dim)).to(self._device).clamp(-0.5, 0.5)
            vae_action = self._model.vae.decode_with_obs(z, next_obs_rep)['reconstruction_action']
            next_action = self._target_model.forward({'obs': next_obs_rep, 'action': vae_action}, mode='compute_actor')['action']
            next_data = {'obs': next_obs_rep, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            target_q_value = self.lmbda * torch.min(target_q_value[0], target_q_value[1]) + (1 - self.lmbda) * torch.max(target_q_value[0], target_q_value[1])
            target_q_value = target_q_value.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
        q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
        (loss_dict['critic_loss'], td_error_per_sample0) = v_1step_td_error(q_data0, self._gamma)
        q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
        (loss_dict['twin_critic_loss'], td_error_per_sample1) = v_1step_td_error(q_data1, self._gamma)
        td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        self._optimizer_q.zero_grad()
        (loss_dict['critic_loss'] + loss_dict['twin_critic_loss']).backward()
        self._optimizer_q.step()
        z = torch.randn((obs.shape[0], self.latent_dim)).to(self._device).clamp(-0.5, 0.5)
        sample_action = self._model.vae.decode_with_obs(z, obs)['reconstruction_action']
        input = {'obs': obs, 'action': sample_action}
        perturbed_action = self._model.forward(input, mode='compute_actor')['action']
        q_input = {'obs': obs, 'action': perturbed_action}
        q = self._learn_model.forward(q_input, mode='compute_critic')['q_value'][0]
        loss_dict['actor_loss'] = -q.mean()
        self._optimizer_policy.zero_grad()
        loss_dict['actor_loss'].backward()
        self._optimizer_policy.step()
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {'td_error': td_error_per_sample.detach().mean().item(), 'target_q_value': target_q_value.detach().mean().item(), **loss_dict}

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            print('Hello World!')
        return ['td_error', 'target_q_value', 'critic_loss', 'twin_critic_loss', 'actor_loss', 'recons_loss', 'kld_loss', 'vae_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        ret = {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'optimizer_q': self._optimizer_q.state_dict(), 'optimizer_policy': self._optimizer_policy.state_dict(), 'optimizer_vae': self._optimizer_vae.state_dict()}
        return ret

    def _init_eval(self):
        if False:
            for i in range(10):
                print('nop')
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_eval')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _init_collect(self) -> None:
        if False:
            print('Hello World!')
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        if False:
            return 10
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            return 10
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            for i in range(10):
                print('nop')
        "\n            Overview:\n                Get the trajectory and the n step return data, then sample from the n_step return data\n            Arguments:\n                - data (:obj:`list`): The trajectory's cache\n            Returns:\n                - samples (:obj:`dict`): The training samples generated\n            "
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)