from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple, defaultdict
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.policy import Policy
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, DatasetNormalizer
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn

@POLICY_REGISTRY.register('pd')
class PDPolicy(Policy):
    """
    Overview:
        Implicit Plan Diffuser
        https://arxiv.org/pdf/2205.09991.pdf

    """
    config = dict(type='pd', cuda=False, priority=False, priority_IS_weight=False, random_collect_size=10000, nstep=1, normalizer='GaussianNormalizer', model=dict(diffuser_model='GaussianDiffusion', diffuser_model_cfg=dict(model='TemporalUnet', model_cfg=dict(transition_dim=23, dim=32, dim_mults=[1, 2, 4, 8], returns_condition=False, condition_dropout=0.1, calc_energy=False, kernel_size=5, attention=False), horizon=80, n_timesteps=1000, predict_epsilon=True, loss_discount=1.0, clip_denoised=False, action_weight=10), value_model='ValueDiffusion', value_model_cfg=dict(model='TemporalValue', model_cfg=dict(horizon=4, transition_dim=23, dim=32, dim_mults=[1, 2, 4, 8], kernel_size=5), horizon=80, n_timesteps=1000, predict_epsilon=True, loss_discount=1.0, clip_denoised=False, action_weight=1.0), n_guide_steps=2, scale=0.1, t_stopgrad=2, scale_grad_by_std=True), learn=dict(update_per_collect=1, batch_size=100, learning_rate=0.0003, ignore_done=False, target_theta=0.005, discount_factor=0.99, gradient_accumulate_every=2, train_epoch=60000, plan_batch_size=64, step_start_update_target=2000, update_target_freq=10, target_weight=0.995, value_step=200000.0, include_returns=True, init_w=0.003))

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            while True:
                i = 10
        return ('pd', ['ding.model.template.diffusion'])

    def _init_learn(self) -> None:
        if False:
            return 10
        "\n        Overview:\n            Learn mode init method. Called by ``self.__init__``.\n            Init q, value and policy's optimizers, algorithm config, main and target models.\n        "
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.action_dim = self._cfg.model.diffuser_model_cfg.action_dim
        self.obs_dim = self._cfg.model.diffuser_model_cfg.obs_dim
        self.n_timesteps = self._cfg.model.diffuser_model_cfg.n_timesteps
        self.gradient_accumulate_every = self._cfg.learn.gradient_accumulate_every
        self.plan_batch_size = self._cfg.learn.plan_batch_size
        self.gradient_steps = 1
        self.update_target_freq = self._cfg.learn.update_target_freq
        self.step_start_update_target = self._cfg.learn.step_start_update_target
        self.target_weight = self._cfg.learn.target_weight
        self.value_step = self._cfg.learn.value_step
        self.use_target = False
        self.horizon = self._cfg.model.diffuser_model_cfg.horizon
        self.include_returns = self._cfg.learn.include_returns
        self._plan_optimizer = Adam(self._model.diffuser.model.parameters(), lr=self._cfg.learn.learning_rate)
        if self._model.value:
            self._value_optimizer = Adam(self._model.value.model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor
        self._target_model = copy.deepcopy(self._model)
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        loss_dict = {}
        data = default_preprocess_learn(data, use_priority=self._priority, use_priority_IS_weight=self._cfg.priority_IS_weight, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        conds = {}
        vals = data['condition_val']
        ids = data['condition_id']
        for i in range(len(ids)):
            conds[ids[i][0].item()] = vals[i]
        if len(ids) > 1:
            self.use_target = True
        data['conditions'] = conds
        if 'returns' in data.keys():
            data['returns'] = data['returns'].unsqueeze(-1)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        x = data['trajectories']
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        cond = data['conditions']
        if 'returns' in data.keys():
            target = data['returns']
        (loss_dict['diffuse_loss'], loss_dict['a0_loss']) = self._model.diffuser_loss(x, cond, t)
        loss_dict['diffuse_loss'] = loss_dict['diffuse_loss'] / self.gradient_accumulate_every
        loss_dict['diffuse_loss'].backward()
        if self._forward_learn_cnt < self.value_step and self._model.value:
            (loss_dict['value_loss'], logs) = self._model.value_loss(x, cond, target, t)
            loss_dict['value_loss'] = loss_dict['value_loss'] / self.gradient_accumulate_every
            loss_dict['value_loss'].backward()
            loss_dict.update(logs)
        if self.gradient_steps >= self.gradient_accumulate_every:
            self._plan_optimizer.step()
            self._plan_optimizer.zero_grad()
            if self._forward_learn_cnt < self.value_step and self._model.value:
                self._value_optimizer.step()
                self._value_optimizer.zero_grad()
            self.gradient_steps = 1
        else:
            self.gradient_steps += 1
        self._forward_learn_cnt += 1
        if self._forward_learn_cnt % self.update_target_freq == 0:
            if self._forward_learn_cnt < self.step_start_update_target:
                self._target_model.load_state_dict(self._model.state_dict())
            else:
                self.update_model_average(self._target_model, self._learn_model)
        if 'returns' in data.keys():
            loss_dict['max_return'] = target.max().item()
            loss_dict['min_return'] = target.min().item()
            loss_dict['mean_return'] = target.mean().item()
        loss_dict['max_traj'] = x.max().item()
        loss_dict['min_traj'] = x.min().item()
        loss_dict['mean_traj'] = x.mean().item()
        return loss_dict

    def update_model_average(self, ma_model, current_model):
        if False:
            i = 10
            return i + 15
        for (current_params, ma_params) in zip(current_model.parameters(), ma_model.parameters()):
            (old_weight, up_weight) = (ma_params.data, current_params.data)
            if old_weight is None:
                ma_params.data = up_weight
            else:
                old_weight * self.target_weight + (1 - self.target_weight) * up_weight

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            print('Hello World!')
        return ['diffuse_loss', 'value_loss', 'max_return', 'min_return', 'mean_return', 'max_traj', 'min_traj', 'mean_traj', 'mean_pred', 'max_pred', 'min_pred', 'a0_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        if self._model.value:
            return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'plan_optimizer': self._plan_optimizer.state_dict(), 'value_optimizer': self._value_optimizer.state_dict()}
        else:
            return {'model': self._learn_model.state_dict(), 'target_model': self._target_model.state_dict(), 'plan_optimizer': self._plan_optimizer.state_dict()}

    def _init_eval(self):
        if False:
            print('Hello World!')
        self._eval_model = model_wrap(self._target_model, wrapper_name='base')
        self._eval_model.reset()
        if self.use_target:
            self._plan_seq = []

    def init_data_normalizer(self, normalizer: DatasetNormalizer=None):
        if False:
            i = 10
            return i + 15
        self.normalizer = normalizer

    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        if False:
            return 10
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._eval_model.eval()
        if self.use_target:
            cur_obs = self.normalizer.normalize(data[:, :self.obs_dim], 'observations')
            target_obs = self.normalizer.normalize(data[:, self.obs_dim:], 'observations')
        else:
            obs = self.normalizer.normalize(data, 'observations')
        with torch.no_grad():
            if self.use_target:
                cur_obs = torch.tensor(cur_obs)
                target_obs = torch.tensor(target_obs)
                if self._cuda:
                    cur_obs = to_device(cur_obs, self._device)
                    target_obs = to_device(target_obs, self._device)
                conditions = {0: cur_obs, self.horizon - 1: target_obs}
            else:
                obs = torch.tensor(obs)
                if self._cuda:
                    obs = to_device(obs, self._device)
                conditions = {0: obs}
            if self.use_target:
                if self._plan_seq == [] or 0 in self._eval_t:
                    plan_traj = self._eval_model.get_eval(conditions, self.plan_batch_size)
                    plan_traj = to_device(plan_traj, 'cpu').numpy()
                    if self._plan_seq == []:
                        self._plan_seq = plan_traj
                        self._eval_t = [0] * len(data_id)
                    else:
                        for id in data_id:
                            if self._eval_t[id] == 0:
                                self._plan_seq[id] = plan_traj[id]
                action = []
                for id in data_id:
                    if self._eval_t[id] < len(self._plan_seq[id]) - 1:
                        next_waypoint = self._plan_seq[id][self._eval_t[id] + 1]
                    else:
                        next_waypoint = self._plan_seq[id][-1].copy()
                        next_waypoint[2:] = 0
                    cur_ob = cur_obs[id]
                    cur_ob = to_device(cur_ob, 'cpu').numpy()
                    act = next_waypoint[:2] - cur_ob[:2] + (next_waypoint[2:] - cur_ob[2:])
                    action.append(act)
                    self._eval_t[id] += 1
            else:
                action = self._eval_model.get_eval(conditions, self.plan_batch_size)
                if self._cuda:
                    action = to_device(action, 'cpu')
                action = self.normalizer.unnormalize(action, 'actions')
            action = torch.tensor(action).to('cpu')
        output = {'action': action}
        output = default_decollate(output)
        return {i: d for (i, d) in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.use_target and data_id:
            for id in data_id:
                self._eval_t[id] = 0

    def _init_collect(self) -> None:
        if False:
            return 10
        pass

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        if False:
            i = 10
            return i + 15
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        if False:
            while True:
                i = 10
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        if False:
            i = 10
            return i + 15
        pass