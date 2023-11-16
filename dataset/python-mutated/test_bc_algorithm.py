import unittest
import gymnasium as gym
import ray
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule, BCTorchRLModuleWithSharedGlobalEncoder, BCTorchMultiAgentModuleWithSharedEncoder
from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule, BCTfRLModuleWithSharedGlobalEncoder, BCTfMultiAgentModuleWithSharedEncoder
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.testing.bc_algorithm import BCConfigTest
from ray.rllib.utils.test_utils import framework_iterator
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole

class TestLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_bc_algorithm(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the Test BC algorithm in single -agent case.'
        config = BCConfigTest().training(model={'fcnet_hiddens': [32, 32]}).experimental(_enable_new_api_stack=True)
        for fw in framework_iterator(config, frameworks='torch'):
            algo = config.build(env='CartPole-v1')
            policy = algo.get_policy()
            rl_module = policy.model
            if fw == 'torch':
                assert isinstance(rl_module, DiscreteBCTorchModule)
            elif fw == 'tf2':
                assert isinstance(rl_module, DiscreteBCTFModule)

    def test_bc_algorithm_marl(self):
        if False:
            print('Hello World!')
        'Tests simple extension of single-agent to independent multi-agent case.'
        policies = {'policy_1', 'policy_2'}
        config = BCConfigTest().experimental(_enable_new_api_stack=True).training(model={'fcnet_hiddens': [32, 32]}).multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, **kwargs: list(policies)[agent_id]).environment(MultiAgentCartPole, env_config={'num_agents': 2})
        for fw in framework_iterator(config, frameworks='torch'):
            algo = config.build()
            for policy_id in policies:
                policy = algo.get_policy(policy_id=policy_id)
                rl_module = policy.model
                if fw == 'torch':
                    assert isinstance(rl_module, DiscreteBCTorchModule)
                elif fw == 'tf2':
                    assert isinstance(rl_module, DiscreteBCTFModule)

    def test_bc_algorithm_w_custom_marl_module(self):
        if False:
            i = 10
            return i + 15
        'Tests the independent multi-agent case with shared encoders.'
        policies = {'policy_1', 'policy_2'}
        for fw in ['torch']:
            if fw == 'torch':
                spec = MultiAgentRLModuleSpec(marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder, module_specs=SingleAgentRLModuleSpec(module_class=BCTorchRLModuleWithSharedGlobalEncoder))
            else:
                spec = MultiAgentRLModuleSpec(marl_module_class=BCTfMultiAgentModuleWithSharedEncoder, module_specs=SingleAgentRLModuleSpec(module_class=BCTfRLModuleWithSharedGlobalEncoder))
            config = BCConfigTest().experimental(_enable_new_api_stack=True).framework(fw).rl_module(rl_module_spec=spec).training(model={'fcnet_hiddens': [32, 32]}).multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, **kwargs: list(policies)[agent_id]).environment(observation_space=gym.spaces.Dict({'global': gym.spaces.Box(low=-1, high=1, shape=(10,)), 'local': gym.spaces.Box(low=-1, high=1, shape=(20,))}), action_space=gym.spaces.Discrete(2)).experimental(_disable_preprocessor_api=True)
            algo = config.build()
            for policy_id in policies:
                policy = algo.get_policy(policy_id=policy_id)
                rl_module = policy.model
                if fw == 'torch':
                    assert isinstance(rl_module, BCTorchRLModuleWithSharedGlobalEncoder)
                elif fw == 'tf2':
                    assert isinstance(rl_module, BCTfRLModuleWithSharedGlobalEncoder)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))