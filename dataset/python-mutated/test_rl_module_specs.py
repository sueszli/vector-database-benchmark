import unittest
import gymnasium as gym
import torch
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule, MultiAgentRLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule, BCTorchRLModuleWithSharedGlobalEncoder, BCTorchMultiAgentModuleWithSharedEncoder
from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule, BCTfRLModuleWithSharedGlobalEncoder, BCTfMultiAgentModuleWithSharedEncoder
MODULES = [DiscreteBCTorchModule, DiscreteBCTFModule]
CUSTOM_MODULES = {'torch': BCTorchRLModuleWithSharedGlobalEncoder, 'tf2': BCTfRLModuleWithSharedGlobalEncoder}
CUSTOM_MARL_MODULES = {'torch': BCTorchMultiAgentModuleWithSharedEncoder, 'tf2': BCTfMultiAgentModuleWithSharedEncoder}

class TestRLModuleSpecs(unittest.TestCase):

    def test_single_agent_spec(self):
        if False:
            i = 10
            return i + 15
        "Tests RLlib's default SingleAgentRLModuleSpec."
        env = gym.make('CartPole-v1')
        for module_class in MODULES:
            spec = SingleAgentRLModuleSpec(module_class=module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict={'fcnet_hiddens': [64]})
            module = spec.build()
            self.assertIsInstance(module, module_class)

    def test_multi_agent_spec(self):
        if False:
            i = 10
            return i + 15
        env = gym.make('CartPole-v1')
        num_agents = 2
        for module_class in MODULES:
            module_specs = {}
            for i in range(num_agents):
                module_specs[f'module_{i}'] = SingleAgentRLModuleSpec(module_class=module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict={'fcnet_hiddens': [32 * (i + 1)]})
            spec = MultiAgentRLModuleSpec(module_specs=module_specs)
            module = spec.build()
            self.assertIsInstance(module, MultiAgentRLModule)

    def test_customized_multi_agent_module(self):
        if False:
            print('Hello World!')
        'Tests creating a customized MARL BC module that owns a shared encoder.'
        global_dim = 10
        local_dims = [16, 32]
        action_dims = [2, 4]
        for fw in ['torch']:
            marl_module_cls = CUSTOM_MARL_MODULES[fw]
            rl_module_cls = CUSTOM_MODULES[fw]
            spec = MultiAgentRLModuleSpec(marl_module_class=marl_module_cls, module_specs={'agent_1': SingleAgentRLModuleSpec(module_class=rl_module_cls, observation_space=gym.spaces.Dict({'global': gym.spaces.Box(low=-1, high=1, shape=(global_dim,)), 'local': gym.spaces.Box(low=-1, high=1, shape=(local_dims[0],))}), action_space=gym.spaces.Discrete(action_dims[0]), model_config_dict={'fcnet_hiddens': [128]}), 'agent_2': SingleAgentRLModuleSpec(module_class=rl_module_cls, observation_space=gym.spaces.Dict({'global': gym.spaces.Box(low=-1, high=1, shape=(global_dim,)), 'local': gym.spaces.Box(low=-1, high=1, shape=(local_dims[1],))}), action_space=gym.spaces.Discrete(action_dims[1]), model_config_dict={'fcnet_hiddens': [128]})})
            model = spec.build()
            if fw == 'torch':
                foo = model['agent_1'].encoder[0].bias
                foo.data = torch.ones_like(foo.data)
                self.assertTrue(torch.allclose(model['agent_2'].encoder[0].bias, foo))

    def test_get_spec_from_module_multi_agent(self):
        if False:
            return 10
        'Tests wether MultiAgentRLModuleSpec.from_module() works.'
        env = gym.make('CartPole-v1')
        num_agents = 2
        for module_class in MODULES:
            module_specs = {}
            for i in range(num_agents):
                module_specs[f'module_{i}'] = SingleAgentRLModuleSpec(module_class=module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict={'fcnet_hiddens': [32 * (i + 1)]})
            spec = MultiAgentRLModuleSpec(module_specs=module_specs)
            module = spec.build()
            spec_from_module = MultiAgentRLModuleSpec.from_module(module)
            self.assertEqual(spec, spec_from_module)

    def test_get_spec_from_module_single_agent(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests wether SingleAgentRLModuleSpec.from_module() works.'
        env = gym.make('CartPole-v1')
        for module_class in MODULES:
            spec = SingleAgentRLModuleSpec(module_class=module_class, observation_space=env.observation_space, action_space=env.action_space, model_config_dict={'fcnet_hiddens': [32]})
            module = spec.build()
            spec_from_module = SingleAgentRLModuleSpec.from_module(module)
            self.assertEqual(spec, spec_from_module)

    def test_update_specs(self):
        if False:
            return 10
        'Tests wether SingleAgentRLModuleSpec.update() works.'
        env = gym.make('CartPole-v0')
        module_spec_1 = SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule, observation_space=env.observation_space, action_space=env.action_space, model_config_dict='Update me!')
        module_spec_2 = SingleAgentRLModuleSpec(model_config_dict={'fcnet_hiddens': [32]})
        self.assertEqual(module_spec_1.model_config_dict, 'Update me!')
        module_spec_1.update(module_spec_2)
        self.assertEqual(module_spec_1.model_config_dict, {'fcnet_hiddens': [32]})

    def test_update_specs_multi_agent(self):
        if False:
            i = 10
            return i + 15
        'Test if updating a SingleAgentRLModuleSpec in MultiAgentRLModuleSpec works.\n\n        This tests if we can update a `model_config_dict` field through different\n        kinds of updates:\n            - Create a SingleAgentRLModuleSpec and update its model_config_dict.\n            - Create two MultiAgentRLModuleSpecs and update the first one with the\n                second one without overwriting it.\n            - Check if the updated MultiAgentRLModuleSpec does not(!) have the\n                updated model_config_dict.\n            - Create two MultiAgentRLModuleSpecs and update the first one with the\n                second one with overwriting it.\n            - Check if the updated MultiAgentRLModuleSpec has(!) the updated\n                model_config_dict.\n\n        '
        env = gym.make('CartPole-v0')
        module_spec_1 = SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule, observation_space='Do not update me!', action_space=env.action_space, model_config_dict='Update me!')
        module_spec_2 = SingleAgentRLModuleSpec(model_config_dict={'fcnet_hiddens': [32]})
        self.assertEqual(module_spec_1.model_config_dict, 'Update me!')
        module_spec_1.update(module_spec_2)
        self.assertEqual(module_spec_1.module_class, DiscreteBCTorchModule)
        self.assertEqual(module_spec_1.observation_space, 'Do not update me!')
        self.assertEqual(module_spec_1.action_space, env.action_space)
        self.assertEqual(module_spec_1.model_config_dict, module_spec_2.model_config_dict)
        module_spec_1 = SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule, observation_space='Do not update me!', action_space=env.action_space, model_config_dict='Update me!')
        marl_spec_1 = MultiAgentRLModuleSpec(marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder, module_specs={'agent_1': module_spec_1})
        marl_spec_2 = MultiAgentRLModuleSpec(marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder, module_specs={'agent_1': module_spec_2})
        self.assertEqual(marl_spec_1.module_specs['agent_1'].model_config_dict, 'Update me!')
        marl_spec_1.update(marl_spec_2, overwrite=True)
        self.assertEqual(marl_spec_1.module_specs['agent_1'], module_spec_2)
        marl_spec_3 = MultiAgentRLModuleSpec(marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder, module_specs={'agent_1': module_spec_1})
        self.assertEqual(marl_spec_3.module_specs['agent_1'].observation_space, 'Do not update me!')
        marl_spec_3.update(marl_spec_2, overwrite=False)
        self.assertEqual(marl_spec_3.module_specs['agent_1'].observation_space, 'Do not update me!')
        module_spec_3 = SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule, observation_space=env.observation_space, action_space=env.action_space, model_config_dict="I'm new!")
        marl_spec_3 = MultiAgentRLModuleSpec(marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder, module_specs={'agent_2': module_spec_3})
        self.assertEqual(marl_spec_1.module_specs.get('agent_2'), None)
        marl_spec_1.update(marl_spec_3)
        self.assertEqual(marl_spec_1.module_specs['agent_2'].model_config_dict, "I'm new!")
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))