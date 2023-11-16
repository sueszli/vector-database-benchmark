import unittest
from dataclasses import dataclass
import gymnasium as gym
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.tf.base import TfModel
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
from ray.rllib.utils.torch_utils import _dynamo_is_available
(_, tf, _) = try_import_tf()
(torch, nn) = try_import_torch()
'\nTODO(Artur): There are a couple of tests for torch.compile that are outstanding:\n- Loading the states of a compile RLModule to a non-compile RLModule and vica-versa\n- Removing a Compiled and non-compiled module to make sure there is no leak\n- ...\n'

class TestModelBase(unittest.TestCase):

    def test_model_input_spec_checking(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests if model input spec checking works correctly.\n\n        This test is centered around the `always_check_shapes` flag of the\n        ModelConfig class. If this flag is set to True, the model will always\n        check if the inputs conform to the specs. If this flag is set to False,\n        the model will only check the input if we encounter an error in side\n        the forward call.\n        '
        for fw in ['torch', 'tf2']:

            class CatModel:
                """Simple model that concatenates parts of its input."""

                def __init__(self, config):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__(config)

                def get_output_specs(self):
                    if False:
                        i = 10
                        return i + 15
                    return SpecDict({'out_1': TensorSpec('b, h', h=1, framework=fw), 'out_2': TensorSpec('b, h', h=4, framework=fw)})

                def get_input_specs(self):
                    if False:
                        return 10
                    return SpecDict({'in_1': TensorSpec('b, h', h=1, framework=fw), 'in_2': TensorSpec('b, h', h=2, framework=fw)})
            if fw == 'tf2':

                class TestModel(CatModel, TfModel):

                    def _forward(self, input_dict):
                        if False:
                            for i in range(10):
                                print('nop')
                        out_2 = tf.concat([input_dict['in_2'], input_dict['in_2']], axis=1)
                        return {'out_1': input_dict['in_1'], 'out_2': out_2}
            else:

                class TestModel(CatModel, TorchModel):

                    def _forward(self, input_dict):
                        if False:
                            return 10
                        out_2 = torch.cat([input_dict['in_2'], input_dict['in_2']], dim=1)
                        return {'out_1': input_dict['in_1'], 'out_2': out_2}

            @dataclass
            class CatModelConfig(ModelConfig):

                def build(self, framework: str):
                    if False:
                        i = 10
                        return i + 15
                    return TestModel(self)
            config = CatModelConfig(always_check_shapes=True)
            model = config.build(framework='spam')
            with self.assertRaisesRegex(SpecCheckingError, 'input spec validation failed'):
                model({'in_1': [1], 'in_2': [1, 2]})
            if fw == 'torch':
                model({'in_1': torch.Tensor([[1]]), 'in_2': torch.Tensor([[1, 2]])})
            else:
                model({'in_1': tf.constant([[1]]), 'in_2': tf.constant([[1, 2]])})
            config = CatModelConfig(always_check_shapes=False)
            model = config.build(framework='spam')
            if fw == 'torch':
                model({'in_1': torch.Tensor([[1]]), 'in_2': torch.Tensor([[1, 2]])})
            else:
                model({'in_1': tf.constant([[1]]), 'in_2': tf.constant([[1, 2]])})
            if fw == 'torch':
                model({'in_1': torch.Tensor([[1]]), 'in_2': torch.Tensor([[1, 2, 3, 4]])})
            else:
                model({'in_1': tf.constant([[1]]), 'in_2': tf.constant([[1, 2, 3, 4]])})
            with self.assertRaisesRegex(SpecCheckingError, 'input spec validation failed'):
                model({'in_1': [1], 'in_2': [1, 2]})

    def test_model_output_spec_checking(self):
        if False:
            i = 10
            return i + 15
        'Tests if model output spec checking works correctly.\n\n        This test is centered around the `always_check_shapes` flag of the\n        ModelConfig class. If this flag is set to True, the model will always\n        check if the outputs conform to the specs. If this flag is set to False,\n        the model will never check the outputs.\n        '
        for fw in ['torch', 'tf2']:

            class BadModel:
                """Simple model that produces bad outputs."""

                def get_output_specs(self):
                    if False:
                        return 10
                    return SpecDict({'out': TensorSpec('b, h', h=1)})

                def get_input_specs(self):
                    if False:
                        print('Hello World!')
                    return SpecDict({'in': TensorSpec('b, h', h=1)})
            if fw == 'tf2':

                class TestModel(BadModel, TfModel):

                    def _forward(self, input_dict):
                        if False:
                            return 10
                        return {'out': torch.Tensor([[1, 2]])}
            else:

                class TestModel(BadModel, TfModel):

                    def _forward(self, input_dict):
                        if False:
                            i = 10
                            return i + 15
                        return {'out': tf.constant([[1, 2]])}

            @dataclass
            class CatModelConfig(ModelConfig):

                def build(self, framework: str):
                    if False:
                        while True:
                            i = 10
                    return TestModel(self)
            config = CatModelConfig(always_check_shapes=True)
            model = config.build(framework='spam')
            with self.assertRaisesRegex(SpecCheckingError, 'output spec validation failed'):
                model({'in': torch.Tensor([[1]])})
            config = CatModelConfig(always_check_shapes=False)
            model = config.build(framework='spam')
            model({'in_1': [[1]]})

    @unittest.skip('Failing with torch >= 2.0')
    @unittest.skipIf(not _dynamo_is_available(), 'torch._dynamo not available')
    def test_torch_compile_no_breaks(self):
        if False:
            print('Hello World!')
        'Tests if torch.compile() does not encounter any breaks.\n\n        torch.compile() should not encounter any breaks when model is on its\n        code path by default. This test checks if this is the case.\n        '

        class SomeTorchModel(TorchModel):
            """Simple model that produces bad outputs."""

            def __init__(self, config):
                if False:
                    return 10
                super().__init__(config)
                self._model = torch.nn.Linear(1, 1)

            def get_output_specs(self):
                if False:
                    for i in range(10):
                        print('nop')
                return SpecDict({'out': TensorSpec('b, h', h=1, framework='torch')})

            def get_input_specs(self):
                if False:
                    i = 10
                    return i + 15
                return SpecDict({'in': TensorSpec('b, h', h=1, framework='torch')})

            def _forward(self, input_dict):
                if False:
                    print('Hello World!')
                return {'out': self._model(input_dict['in'])}

        @dataclass
        class SomeTorchModelConfig(ModelConfig):

            def build(self, framework: str):
                if False:
                    return 10
                return SomeTorchModel(self)
        config = SomeTorchModelConfig()
        model = config.build(framework='spam')

        def compile_me(input_dict):
            if False:
                return 10
            return model(input_dict)
        import torch._dynamo as dynamo
        (explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose) = dynamo.explain(compile_me, {'in': torch.Tensor([[1]])})
        print(explanation_verbose)
        self.assertEquals(len(break_reasons), 1)

    @unittest.skipIf(not _dynamo_is_available(), 'torch._dynamo not available')
    def test_torch_compile_forwards(self):
        if False:
            while True:
                i = 10
        'Test if logic around TorchCompileConfig works as intended.'
        spec = SingleAgentRLModuleSpec(module_class=PPOTorchRLModule, observation_space=gym.spaces.Box(low=0, high=1, shape=(32,)), action_space=gym.spaces.Box(low=0, high=1, shape=(1,)), model_config_dict={}, catalog_class=PPOCatalog)
        torch_module = spec.build()
        compile_config = TorchCompileConfig()
        torch_module.compile(compile_config)
        torch_module._forward_train({'obs': torch.randn(1, 32)})
        torch_module._forward_inference({'obs': torch.randn(1, 32)})
        torch_module._forward_exploration({'obs': torch.randn(1, 32)})
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))