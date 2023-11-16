import gymnasium as gym
from typing import Type
import unittest
import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.tf.ppo_tf_learner import PPOTfLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec, RLModule
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec, MultiAgentRLModule
from ray.rllib.utils.test_utils import check

class TestAlgorithmConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_running_specific_algo_with_generic_config(self):
        if False:
            print('Hello World!')
        'Tests, whether some algo can be run with the generic AlgorithmConfig.'
        config = AlgorithmConfig(algo_class=PPO).environment('CartPole-v0').training(lr=0.12345, train_batch_size=3000)
        algo = config.build()
        self.assertTrue(algo.config.lr == 0.12345)
        self.assertTrue(algo.config.train_batch_size == 3000)
        algo.train()
        algo.stop()

    def test_update_from_dict_works_for_multi_callbacks(self):
        if False:
            return 10
        'Test to make sure callbacks config dict works.'
        config_dict = {'callbacks': make_multi_callbacks([])}
        config = AlgorithmConfig()
        config.update_from_dict(config_dict)
        serialized = config.serialize()
        self.assertEqual(serialized['callbacks'], 'ray.rllib.algorithms.callbacks.make_multi_callbacks.<locals>._MultiCallbacks')

    def test_freezing_of_algo_config(self):
        if False:
            return 10
        'Tests, whether freezing an AlgorithmConfig actually works as expected.'
        config = AlgorithmConfig().environment('CartPole-v0').training(lr=0.12345, train_batch_size=3000).multi_agent(policies={'pol1': (None, None, None, AlgorithmConfig.overrides(lr=0.001))}, policy_mapping_fn=lambda agent_id, episode, worker, **kw: 'pol1')
        config.freeze()

        def set_lr(config):
            if False:
                i = 10
                return i + 15
            config.lr = 0.01
        self.assertRaisesRegex(AttributeError, 'Cannot set attribute.+of an already frozen AlgorithmConfig', lambda : set_lr(config))

        def set_one_policy(config):
            if False:
                for i in range(10):
                    print('nop')
            config.policies['pol1'] = (None, None, None, {'lr': 0.123})

    def test_rollout_fragment_length(self):
        if False:
            while True:
                i = 10
        'Tests the proper auto-computation of the `rollout_fragment_length`.'
        config = AlgorithmConfig().rollouts(num_rollout_workers=4, num_envs_per_worker=3, rollout_fragment_length='auto').training(train_batch_size=2456)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=0) == 205)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=1) == 205)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=2) == 205)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=3) == 204)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=4) == 204)
        config = AlgorithmConfig().rollouts(num_rollout_workers=3, num_envs_per_worker=2, rollout_fragment_length='auto').training(train_batch_size=4000)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=0) == 667)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=1) == 667)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=2) == 667)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=3) == 666)
        config = AlgorithmConfig().rollouts(num_rollout_workers=12, rollout_fragment_length='auto').training(train_batch_size=1342)
        for i in range(11):
            self.assertTrue(config.get_rollout_fragment_length(worker_index=i) == 112)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=11) == 111)
        self.assertTrue(config.get_rollout_fragment_length(worker_index=12) == 111)

    def test_detect_atari_env(self):
        if False:
            while True:
                i = 10
        'Tests that we can properly detect Atari envs.'
        config = AlgorithmConfig().environment(env='ALE/Breakout-v5', env_config={'frameskip': 1})
        self.assertTrue(config.is_atari)
        config = AlgorithmConfig().environment(env='ALE/Pong-v5')
        self.assertTrue(config.is_atari)
        config = AlgorithmConfig().environment(env='CartPole-v1')
        self.assertFalse(config.is_atari)
        config = AlgorithmConfig().environment(env=lambda ctx: gym.make('ALE/Breakout-v5', frameskip=1))
        self.assertFalse(config.is_atari)
        config = AlgorithmConfig().environment(env='NotAtari')
        self.assertFalse(config.is_atari)

    def test_rl_module_api(self):
        if False:
            while True:
                i = 10
        config = PPOConfig().experimental(_enable_new_api_stack=True).environment('CartPole-v1').framework('torch').rollouts(enable_connectors=True)
        self.assertEqual(config.rl_module_spec.module_class, PPOTorchRLModule)

        class A:
            pass
        config = config.rl_module(rl_module_spec=SingleAgentRLModuleSpec(A))
        self.assertEqual(config.rl_module_spec.module_class, A)

    def test_learner_hyperparameters_per_module(self):
        if False:
            print('Hello World!')
        'Tests, whether per-module config overrides (multi-agent) work as expected.'
        hps = PPOConfig().training(kl_coeff=0.5).multi_agent(policies={'module_1', 'module_2', 'module_3'}, algorithm_config_overrides_per_module={'module_1': PPOConfig.overrides(lr=0.01, kl_coeff=0.1), 'module_2': PPOConfig.overrides(grad_clip=100.0)}).get_learner_hyperparameters()
        check(hps.learning_rate, 5e-05)
        check(hps.grad_clip, None)
        check(hps.grad_clip_by, 'global_norm')
        check(hps.kl_coeff, 0.5)
        hps_1 = hps.get_hps_for_module('module_1')
        check(hps_1.learning_rate, 0.01)
        check(hps_1.grad_clip, None)
        check(hps_1.grad_clip_by, 'global_norm')
        check(hps_1.kl_coeff, 0.1)
        hps_2 = hps.get_hps_for_module('module_2')
        check(hps_2.learning_rate, 5e-05)
        check(hps_2.grad_clip, 100.0)
        check(hps_2.grad_clip_by, 'global_norm')
        check(hps_2.kl_coeff, 0.5)
        self.assertTrue('module_3' not in hps._per_module_overrides)
        hps_3 = hps.get_hps_for_module('module_3')
        self.assertTrue(hps_3 is hps)

    def test_learner_api(self):
        if False:
            print('Hello World!')
        config = PPOConfig().experimental(_enable_new_api_stack=True).environment('CartPole-v1').rollouts(enable_connectors=True).framework('tf2')
        self.assertEqual(config.learner_class, PPOTfLearner)

    def _assertEqualMARLSpecs(self, spec1, spec2):
        if False:
            i = 10
            return i + 15
        self.assertEqual(spec1.marl_module_class, spec2.marl_module_class)
        self.assertEqual(set(spec1.module_specs.keys()), set(spec2.module_specs.keys()))
        for (k, module_spec1) in spec1.module_specs.items():
            module_spec2 = spec2.module_specs[k]
            self.assertEqual(module_spec1.module_class, module_spec2.module_class)
            self.assertEqual(module_spec1.observation_space, module_spec2.observation_space)
            self.assertEqual(module_spec1.action_space, module_spec2.action_space)
            self.assertEqual(module_spec1.model_config_dict, module_spec2.model_config_dict)

    def _get_expected_marl_spec(self, config: AlgorithmConfig, expected_module_class: Type[RLModule], passed_module_class: Type[RLModule]=None, expected_marl_module_class: Type[MultiAgentRLModule]=None):
        if False:
            while True:
                i = 10
        'This is a utility function that retrieves the expected marl specs.\n\n        Args:\n            config: The algorithm config.\n            expected_module_class: This is the expected RLModule class that is going to\n                be reference in the SingleAgentRLModuleSpec parts of the\n                MultiAgentRLModuleSpec.\n            passed_module_class: This is the RLModule class that is passed into the\n                module_spec argument of get_marl_module_spec. The function is\n                designed so that it will use the passed in module_spec for the\n                SingleAgentRLModuleSpec parts of the MultiAgentRLModuleSpec.\n            expected_marl_module_class: This is the expected MultiAgentRLModule class\n                that is going to be reference in the MultiAgentRLModuleSpec.\n\n        Returns:\n            Tuple of the returned MultiAgentRLModuleSpec from config.\n            get_marl_module_spec() and the expected MultiAgentRLModuleSpec.\n        '
        from ray.rllib.policy.policy import PolicySpec
        if expected_marl_module_class is None:
            expected_marl_module_class = MultiAgentRLModule
        env = gym.make('CartPole-v1')
        policy_spec_ph = PolicySpec(observation_space=env.observation_space, action_space=env.action_space, config=AlgorithmConfig())
        marl_spec = config.get_marl_module_spec(policy_dict={'p1': policy_spec_ph, 'p2': policy_spec_ph}, single_agent_rl_module_spec=SingleAgentRLModuleSpec(module_class=passed_module_class) if passed_module_class else None)
        expected_marl_spec = MultiAgentRLModuleSpec(marl_module_class=expected_marl_module_class, module_specs={'p1': SingleAgentRLModuleSpec(module_class=expected_module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict=AlgorithmConfig().model), 'p2': SingleAgentRLModuleSpec(module_class=expected_module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict=AlgorithmConfig().model)})
        return (marl_spec, expected_marl_spec)

    def test_get_marl_module_spec(self):
        if False:
            i = 10
            return i + 15
        'Tests whether the get_marl_module_spec() method works properly.'
        from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

        class CustomRLModule1(DiscreteBCTorchModule):
            pass

        class CustomRLModule2(DiscreteBCTorchModule):
            pass

        class CustomRLModule3(DiscreteBCTorchModule):
            pass

        class CustomMARLModule1(MultiAgentRLModule):
            pass

        class SingleAgentAlgoConfig(AlgorithmConfig):

            def get_default_rl_module_spec(self):
                if False:
                    print('Hello World!')
                return SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule)

        class MultiAgentAlgoConfigWithNoSingleAgentSpec(AlgorithmConfig):

            def get_default_rl_module_spec(self):
                if False:
                    print('Hello World!')
                return MultiAgentRLModuleSpec(marl_module_class=CustomMARLModule1)

        class MultiAgentAlgoConfig(AlgorithmConfig):

            def get_default_rl_module_spec(self):
                if False:
                    while True:
                        i = 10
                return MultiAgentRLModuleSpec(marl_module_class=CustomMARLModule1, module_specs=SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule))
        config = SingleAgentAlgoConfig().experimental(_enable_new_api_stack=True)
        (spec, expected) = self._get_expected_marl_spec(config, DiscreteBCTorchModule)
        self._assertEqualMARLSpecs(spec, expected)
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule2, passed_module_class=CustomRLModule2)
        self._assertEqualMARLSpecs(spec, expected)
        config = SingleAgentAlgoConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'p1': SingleAgentRLModuleSpec(module_class=CustomRLModule1), 'p2': SingleAgentRLModuleSpec(module_class=CustomRLModule1)}))
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        config = SingleAgentAlgoConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=SingleAgentRLModuleSpec(module_class=CustomRLModule1))
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule2, passed_module_class=CustomRLModule2)
        self._assertEqualMARLSpecs(spec, expected)
        config = SingleAgentAlgoConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs=SingleAgentRLModuleSpec(module_class=CustomRLModule1)))
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule2, passed_module_class=CustomRLModule2)
        self._assertEqualMARLSpecs(spec, expected)
        config = SingleAgentAlgoConfig().experimental(_enable_new_api_stack=True).rl_module(rl_module_spec=MultiAgentRLModuleSpec(marl_module_class=CustomMARLModule1, module_specs={'p1': SingleAgentRLModuleSpec(module_class=CustomRLModule1), 'p2': SingleAgentRLModuleSpec(module_class=CustomRLModule1)}))
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1, expected_marl_module_class=CustomMARLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1, passed_module_class=CustomRLModule3, expected_marl_module_class=CustomMARLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        config = MultiAgentAlgoConfigWithNoSingleAgentSpec().experimental(_enable_new_api_stack=True)
        self.assertRaisesRegex(ValueError, 'Module_specs cannot be None', lambda : config.rl_module_spec)
        config = MultiAgentAlgoConfig().experimental(_enable_new_api_stack=True)
        (spec, expected) = self._get_expected_marl_spec(config, DiscreteBCTorchModule, expected_marl_module_class=CustomMARLModule1)
        self._assertEqualMARLSpecs(spec, expected)
        (spec, expected) = self._get_expected_marl_spec(config, CustomRLModule1, passed_module_class=CustomRLModule1, expected_marl_module_class=CustomMARLModule1)
        self._assertEqualMARLSpecs(spec, expected)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))