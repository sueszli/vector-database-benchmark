import pytest
import os
import torch
from easydict import EasyDict
from ding.world_model.base_world_model import DreamWorldModel, DynaWorldModel
from ding.worker.replay_buffer import NaiveReplayBuffer, EpisodeReplayBuffer

@pytest.mark.unittest
class TestDynaWorldModel:

    @pytest.mark.parametrize('buffer_type', [NaiveReplayBuffer, EpisodeReplayBuffer])
    def test_fill_img_buffer(self, buffer_type):
        if False:
            i = 10
            return i + 15
        env_buffer = buffer_type(buffer_type.default_config(), None, 'dyna_exp_name', 'env_buffer_for_test')
        img_buffer = buffer_type(buffer_type.default_config(), None, 'dyna_exp_name', 'img_buffer_for_test')
        fake_config = EasyDict(train_freq=250, eval_freq=250, cuda=False, rollout_length_scheduler=dict(type='linear', rollout_start_step=20000, rollout_end_step=150000, rollout_length_min=1, rollout_length_max=25), other=dict(real_ratio=0.05, rollout_retain=4, rollout_batch_size=100000, imagination_buffer=dict(type='elastic', replay_buffer_size=6000000, deepcopy=False, enable_track_used_data=False, periodic_thruput_seconds=60)))
        (T, B, O, A) = (25, 20, 100, 30)

        class FakeModel(DynaWorldModel):

            def train(self, env_buffer, envstep, train_iter):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def eval(self, env_buffer, envstep, train_iter):
                if False:
                    print('Hello World!')
                pass

            def step(self, obs, action):
                if False:
                    return 10
                return (torch.zeros(B), torch.rand(B, O), obs.sum(-1) > 0)
        from ding.policy import SACPolicy
        from ding.model import ContinuousQAC
        policy_config = SACPolicy.default_config()
        policy_config.model.update(dict(obs_shape=2, action_shape=2))
        model = ContinuousQAC(**policy_config.model)
        policy = SACPolicy(policy_config, model=model).collect_mode
        fake_model = FakeModel(fake_config, None, None)
        env_buffer.push([{'obs': torch.randn(2), 'next_obs': torch.randn(2), 'action': torch.randn(2), 'reward': torch.randn(1), 'done': False, 'collect_iter': 0}] * 20, 0)
        super(FakeModel, fake_model).fill_img_buffer(policy, env_buffer, img_buffer, 0, 0)
        os.popen('rm -rf dyna_exp_name')

@pytest.mark.unittest
class TestDreamWorldModel:

    def test_rollout(self):
        if False:
            print('Hello World!')
        fake_config = EasyDict(train_freq=250, eval_freq=250, cuda=False, rollout_length_scheduler=dict(type='linear', rollout_start_step=20000, rollout_end_step=150000, rollout_length_min=1, rollout_length_max=25))
        envstep = 150000
        (T, B, O, A) = (25, 20, 100, 30)

        class FakeModel(DreamWorldModel):

            def train(self, env_buffer, envstep, train_iter):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def eval(self, env_buffer, envstep, train_iter):
                if False:
                    i = 10
                    return i + 15
                pass

            def step(self, obs, action):
                if False:
                    print('Hello World!')
                return (torch.zeros(B), torch.rand(B, O), obs.sum(-1) > 0)

        def fake_policy_fn(obs):
            if False:
                return 10
            return (torch.randn(B, A), torch.zeros(B))
        fake_model = FakeModel(fake_config, None, None)
        obs = torch.rand(B, O)
        (obss, actions, rewards, aug_rewards, dones) = super(FakeModel, fake_model).rollout(obs, fake_policy_fn, envstep)
        assert obss.shape == (T + 1, B, O)
        assert actions.shape == (T + 1, B, A)
        assert rewards.shape == (T, B)
        assert aug_rewards.shape == (T + 1, B)
        assert dones.shape == (T, B)