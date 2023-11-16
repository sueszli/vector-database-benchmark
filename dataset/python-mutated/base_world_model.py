from typing import Tuple, Callable, Optional
from collections import namedtuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn
from easydict import EasyDict
from ding.worker import IBuffer
from ding.envs import BaseEnv
from ding.utils import deep_merge_dicts
from ding.world_model.utils import get_rollout_length_scheduler
from ding.utils import import_module, WORLD_MODEL_REGISTRY

def get_world_model_cls(cfg):
    if False:
        while True:
            i = 10
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.get(cfg.type)

def create_world_model(cfg, *args, **kwargs):
    if False:
        while True:
            i = 10
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.build(cfg.type, cfg, *args, **kwargs)

class WorldModel(ABC):
    """
    Overview:
        Abstract baseclass for world model.

    Interfaces:
        should_train, should_eval, train, eval, step
    """
    config = dict(train_freq=250, eval_freq=250, cuda=True, rollout_length_scheduler=dict(type='linear', rollout_start_step=20000, rollout_end_step=150000, rollout_length_min=1, rollout_length_max=25))

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        if False:
            print('Hello World!')
        self.cfg = cfg
        self.env = env
        self.tb_logger = tb_logger
        self._cuda = cfg.cuda
        self.train_freq = cfg.train_freq
        self.eval_freq = cfg.eval_freq
        self.rollout_length_scheduler = get_rollout_length_scheduler(cfg.rollout_length_scheduler)
        self.last_train_step = 0
        self.last_eval_step = 0

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            while True:
                i = 10
        merge_cfg = EasyDict(cfg_type=cls.__name__ + 'Dict')
        while cls != ABC:
            merge_cfg = deep_merge_dicts(merge_cfg, cls.config)
            cls = cls.__base__
        return merge_cfg

    def should_train(self, envstep: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Check whether need to train world model.\n        '
        return envstep - self.last_train_step >= self.train_freq

    def should_eval(self, envstep: int):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Check whether need to evaluate world model.\n        '
        return envstep - self.last_eval_step >= self.eval_freq and self.last_train_step != 0

    @abstractmethod
    def train(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Train world model using data from env_buffer.\n\n        Arguments:\n            - env_buffer (:obj:`IBuffer`): the buffer which collects real environment steps\n            - envstep (:obj:`int`): the current number of environment steps in real environment\n            - train_iter (:obj:`int`): the current number of policy training iterations\n        '
        raise NotImplementedError

    @abstractmethod
    def eval(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Evaluate world model using data from env_buffer.\n\n        Arguments:\n            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps\n            - envstep (:obj:`int`): the current number of environment steps in real environment\n            - train_iter (:obj:`int`): the current number of policy training iterations\n        '
        raise NotImplementedError

    @abstractmethod
    def step(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Take one step in world model.\n\n        Arguments:\n            - obs (:obj:`torch.Tensor`): current observations :math:`S_t`\n            - action (:obj:`torch.Tensor`): current actions :math:`A_t`\n\n        Returns:\n            - reward (:obj:`torch.Tensor`): rewards :math:`R_t`\n            - next_obs (:obj:`torch.Tensor`): next observations :math:`S_t+1`\n            - done (:obj:`torch.Tensor`): whether the episodes ends\n\n        Shapes:\n            :math:`B`: batch size\n            :math:`O`: observation dimension\n            :math:`A`: action dimension\n\n            - obs:      [B, O]\n            - action:   [B, A]\n            - reward:   [B, ]\n            - next_obs: [B, O]\n            - done:     [B, ]\n        '
        raise NotImplementedError

class DynaWorldModel(WorldModel, ABC):
    """
    Overview:
        Dyna-style world model (summarized in arXiv: 1907.02057) which stores and\\
        reuses imagination rollout in the imagination buffer.

    Interfaces:
        sample, fill_img_buffer, should_train, should_eval, train, eval, step
    """
    config = dict(other=dict(real_ratio=0.05, rollout_retain=4, rollout_batch_size=100000, imagination_buffer=dict(type='elastic', replay_buffer_size=6000000, deepcopy=False, enable_track_used_data=False, periodic_thruput_seconds=60)))

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        if False:
            print('Hello World!')
        super().__init__(cfg, env, tb_logger)
        self.real_ratio = cfg.other.real_ratio
        self.rollout_batch_size = cfg.other.rollout_batch_size
        self.rollout_retain = cfg.other.rollout_retain
        self.buffer_size_scheduler = lambda x: self.rollout_length_scheduler(x) * self.rollout_batch_size * self.rollout_retain

    def sample(self, env_buffer: IBuffer, img_buffer: IBuffer, batch_size: int, train_iter: int) -> dict:
        if False:
            return 10
        '\n        Overview:\n            Sample from the combination of environment buffer and imagination buffer with\\\n            certain ratio to generate batched data for policy training.\n\n        Arguments:\n            - policy (:obj:`namedtuple`): policy in collect mode\n            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps\n            - img_buffer (:obj:`IBuffer`): the buffer that collects imagination steps\n            - batch_size (:obj:`int`): the batch size for policy training\n            - train_iter (:obj:`int`): the current number of policy training iterations\n\n        Returns:\n            - data (:obj:`int`): the training data for policy training\n        '
        env_batch_size = int(batch_size * self.real_ratio)
        img_batch_size = batch_size - env_batch_size
        env_data = env_buffer.sample(env_batch_size, train_iter)
        img_data = img_buffer.sample(img_batch_size, train_iter)
        train_data = env_data + img_data
        return train_data

    def fill_img_buffer(self, policy: namedtuple, env_buffer: IBuffer, img_buffer: IBuffer, envstep: int, train_iter: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Sample from the env_buffer, rollouts to generate new data, and push them into the img_buffer.\n\n        Arguments:\n            - policy (:obj:`namedtuple`): policy in collect mode\n            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps\n            - img_buffer (:obj:`IBuffer`): the buffer that collects imagination steps\n            - envstep (:obj:`int`): the current number of environment steps in real environment\n            - train_iter (:obj:`int`): the current number of policy training iterations\n        '
        from ding.torch_utils import to_tensor
        from ding.envs import BaseEnvTimestep
        from ding.worker.collector.base_serial_collector import to_tensor_transitions

        def step(obs, act):
            if False:
                return 10
            data_id = list(obs.keys())
            obs = torch.stack([obs[id] for id in data_id], dim=0)
            act = torch.stack([act[id] for id in data_id], dim=0)
            with torch.no_grad():
                (rewards, next_obs, terminals) = self.step(obs, act)
            timesteps = {id: BaseEnvTimestep(n, r, d, {}) for (id, n, r, d) in zip(data_id, next_obs.cpu().numpy(), rewards.unsqueeze(-1).cpu().numpy(), terminals.cpu().numpy())}
            return timesteps
        rollout_length = self.rollout_length_scheduler(envstep)
        data = env_buffer.sample(self.rollout_batch_size, train_iter, replace=True)
        obs = {id: data[id]['obs'] for id in range(len(data))}
        buffer = [[] for id in range(len(obs))]
        new_data = []
        for i in range(rollout_length):
            obs = to_tensor(obs, dtype=torch.float32)
            policy_output = policy.forward(obs)
            actions = {id: output['action'] for (id, output) in policy_output.items()}
            timesteps = step(obs, actions)
            obs_new = {}
            for (id, timestep) in timesteps.items():
                transition = policy.process_transition(obs[id], policy_output[id], timestep)
                transition['collect_iter'] = train_iter
                buffer[id].append(transition)
                if not timestep.done:
                    obs_new[id] = timestep.obs
                if timestep.done or i + 1 == rollout_length:
                    transitions = to_tensor_transitions(buffer[id])
                    train_sample = policy.get_train_sample(transitions)
                    new_data.extend(train_sample)
            if len(obs_new) == 0:
                break
            obs = obs_new
        img_buffer.push(new_data, cur_collector_envstep=envstep)

class DreamWorldModel(WorldModel, ABC):
    """
    Overview:
        Dreamer-style world model which uses each imagination rollout only once\\
        and backpropagate through time(rollout) to optimize policy.

    Interfaces:
        rollout, should_train, should_eval, train, eval, step
    """

    def rollout(self, obs: Tensor, actor_fn: Callable[[Tensor], Tuple[Tensor, Tensor]], envstep: int, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[bool]]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Generate batched imagination rollouts starting from the current observations.\\\n            This function is useful for value gradients where the policy is optimized by BPTT.\n\n        Arguments:\n            - obs (:obj:`Tensor`): the current observations :math:`S_t`\n            - actor_fn (:obj:`Callable`): the unified API :math:`(A_t, H_t) = pi(S_t)`\n            - envstep (:obj:`int`): the current number of environment steps in real environment\n\n        Returns:\n            - obss (:obj:`Tensor`):        :math:`S_t,  ..., S_t+n`\n            - actions (:obj:`Tensor`):     :math:`A_t,  ..., A_t+n`\n            - rewards (:obj:`Tensor`):     :math:`R_t,  ..., R_t+n-1`\n            - aug_rewards (:obj:`Tensor`): :math:`H_t,  ..., H_t+n`, this can be entropy bonus as in SAC,\n                                                otherwise it should be a zero tensor\n            - dones (:obj:`Tensor`):       :math:`\\text{done}_t, ..., \\text{done}_t+n`\n\n        Shapes:\n            :math:`N`: time step\n            :math:`B`: batch size\n            :math:`O`: observation dimension\n            :math:`A`: action dimension\n\n            - obss:        :math:`[N+1, B, O]`, where obss[0] are the real observations\n            - actions:     :math:`[N+1, B, A]`\n            - rewards:     :math:`[N,   B]`\n            - aug_rewards: :math:`[N+1, B]`\n            - dones:       :math:`[N,   B]`\n\n        .. note::\n            - The rollout length is determined by rollout length scheduler.\n\n            - actor_fn's inputs and outputs shape are similar to WorldModel.step()\n        "
        horizon = self.rollout_length_scheduler(envstep)
        if isinstance(self, nn.Module):
            self.requires_grad_(False)
        obss = [obs]
        actions = []
        rewards = []
        aug_rewards = []
        dones = []
        for _ in range(horizon):
            (action, aug_reward) = actor_fn(obs)
            (reward, obs, done) = self.step(obs, action, **kwargs)
            reward = reward + aug_reward
            obss.append(obs)
            actions.append(action)
            rewards.append(reward)
            aug_rewards.append(aug_reward)
            dones.append(done)
        (action, aug_reward) = actor_fn(obs)
        actions.append(action)
        aug_rewards.append(aug_reward)
        if isinstance(self, nn.Module):
            self.requires_grad_(True)
        return (torch.stack(obss), torch.stack(actions), torch.stack(rewards) if rewards else torch.tensor(rewards, device=obs.device), torch.stack(aug_rewards), torch.stack(dones) if dones else torch.tensor(dones, device=obs.device))

class HybridWorldModel(DynaWorldModel, DreamWorldModel, ABC):
    """
    Overview:
        The hybrid model that combines reused and on-the-fly rollouts.

    Interfaces:
        rollout, sample, fill_img_buffer, should_train, should_eval, train, eval, step
    """

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):
        if False:
            while True:
                i = 10
        DynaWorldModel.__init__(self, cfg, env, tb_logger)
        DreamWorldModel.__init__(self, cfg, env, tb_logger)