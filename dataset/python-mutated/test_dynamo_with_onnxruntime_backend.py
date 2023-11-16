from __future__ import annotations
import copy
import dataclasses
import os
import sys
from typing import Tuple
import onnxruntime
import torch
import torch._dynamo.backends.registry
from parameterized import parameterized
from torch import nn
from torch.onnx import _OrtBackend as OrtBackend, _OrtBackendOptions as OrtBackendOptions, ExportOptions
from torch.testing._internal import common_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onnx_test_common

def make_aot_ort(dynamic: bool=False):
    if False:
        i = 10
        return i + 15
    ort_backend = OrtBackend(options=OrtBackendOptions(export_options=ExportOptions(dynamic_shapes=dynamic)))
    return (ort_backend, ort_backend)

class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    def test_torch_compile_backend_registration(self):
        if False:
            print('Hello World!')
        self.assertIn('onnxrt', torch._dynamo.backends.registry.list_backends())
        backend = torch._dynamo.backends.registry.lookup_backend('onnxrt')
        self.assertEqual(backend.__module__, 'torch.onnx._internal.onnxruntime')

    def _test_torch_compile_backend_caching_assert_reused(self, options: OrtBackendOptions):
        if False:
            while True:
                i = 10
        self.assertFalse(OrtBackend.get_cached_instances())
        new_backend = OrtBackend.get_cached_instance_for_options(options)
        reused_backend = OrtBackend.get_cached_instance_for_options(options)
        self.assertEqual(len(OrtBackend.get_cached_instances()), 1)
        self.assertIs(reused_backend, new_backend)
        if options is None or options.ort_session_options is None:
            self.assertEqual(new_backend, OrtBackend.get_cached_instance_for_options(dataclasses.asdict(options) if options else None))

    @parameterized.expand([(None,), (OrtBackendOptions(),), (OrtBackendOptions(use_aot_autograd=True),), (OrtBackendOptions(use_aot_autograd=False),), (OrtBackendOptions(preallocate_output=True),), (OrtBackendOptions(preallocate_output=False),), (OrtBackendOptions(infer_execution_providers=True),), (OrtBackendOptions(infer_execution_providers=False),), (OrtBackendOptions(preferred_execution_providers=['A', 'B', 'C']),), (OrtBackendOptions(preferred_execution_providers=['A', 'B', ('C', {'option': 'value'})]),), (OrtBackendOptions(default_execution_providers=['Something']),), (OrtBackendOptions(export_options=ExportOptions(dynamic_shapes=True)),), (OrtBackendOptions(use_aot_autograd=False, export_options=ExportOptions(op_level_debug=True, dynamic_shapes=True)),)])
    def test_torch_compile_backend_caching_assert_reused(self, options: OrtBackendOptions):
        if False:
            i = 10
            return i + 15
        self._test_torch_compile_backend_caching_assert_reused(options)

    @parameterized.expand([(OrtBackendOptions(ort_session_options=onnxruntime.SessionOptions()),)])
    def test_torch_compile_backend_caching_assert_not_reused(self, options: OrtBackendOptions):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError):
            self._test_torch_compile_backend_caching_assert_reused(options)

    def _test_model_numerically(self, model, dynamo_backend, example_args_collection):
        if False:
            return 10
        'Run original and compiled model and compare the results.\n\n        Args:\n            model: The model to test.\n            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or\n              the first returned value of `make_aot_ort(dynamic=True)`.\n            example_args_collection: A tuple of example arguments to test. E.g.,\n                (\n                  (torch.randn(2), torch.randn(2)),\n                  (torch.randn(4), torch.randn(4)),\n                )\n              if you want to test\n                model(torch.randn(2), torch.randn(2)) and\n                model(torch.randn(4), torch.randn(4))\n              .\n        '
        compiled_model = torch.compile(model if not isinstance(model, torch.nn.Module) else copy.deepcopy(model), backend=dynamo_backend, dynamic=True)
        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            result = compiled_model(*example_args)
            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(baseline_result, result)
            else:
                for (baseline_elem, result_elem) in zip(baseline_result, result):
                    torch.testing.assert_close(baseline_elem, result_elem)

    def _assert_counting_information(self, ort_backend: OrtBackend, expected_execution_count: int, number_of_cached_graph_modules: int, number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...]):
        if False:
            print('Hello World!')
        self.assertEqual(expected_execution_count, ort_backend.execution_count)
        self.assertEqual(len(ort_backend._all_ort_execution_info.execution_info_per_graph_module), number_of_cached_graph_modules)
        self.assertEqual(len(ort_backend._all_ort_execution_info.execution_info_per_graph_module), len(number_of_exported_onnx_models_for_all_graph_modules))
        for (onnx_info, expected_number_of_onnx_models) in zip(ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(), number_of_exported_onnx_models_for_all_graph_modules):
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    @parameterized.expand([(True,), (False,)])
    def test_elementwise_function_single_output(self, test_local_backend: bool):
        if False:
            while True:
                i = 10
        example_args_collection = tuple(((torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 6, 8, 10)))

        def elementwise_model(x: torch.Tensor):
            if False:
                i = 10
                return i + 15
            y = x.relu()
            z = y.sigmoid()
            return z
        if test_local_backend:
            (local_aot_ort, local_ort) = make_aot_ort(dynamic=True)
        else:
            (local_aot_ort, local_ort) = ('onnxrt', None)
        self._test_model_numerically(elementwise_model, local_aot_ort, example_args_collection)
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(local_ort, expected_execution_count=len(example_args_collection), number_of_cached_graph_modules=1, number_of_exported_onnx_models_for_all_graph_modules=(1,))

    @parameterized.expand([(True,), (False,)])
    def test_elementwise_function_multiple_output(self, test_local_backend: bool):
        if False:
            return 10
        example_args_collection = tuple(((torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 8)))

        def elementwise_model_with_multiple_outputs(w: torch.Tensor):
            if False:
                while True:
                    i = 10
            x = w + w
            y = x.relu()
            z = y * y
            return (x, y, z)
        if test_local_backend:
            (local_aot_ort, local_ort) = make_aot_ort(dynamic=True)
        else:
            (local_aot_ort, local_ort) = ('onnxrt', None)
        self._test_model_numerically(elementwise_model_with_multiple_outputs, local_aot_ort, example_args_collection)
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(local_ort, expected_execution_count=len(example_args_collection), number_of_cached_graph_modules=1, number_of_exported_onnx_models_for_all_graph_modules=(1,))

    @parameterized.expand([(True,), (False,)])
    def test_mlp_with_local_backend(self, test_local_backend: bool):
        if False:
            i = 10
            return i + 15
        example_args_collection = tuple(((torch.randn(batch, 2, dtype=torch.float32),) for batch in (1, 2, 4, 6, 8)))

        class MLP(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = nn.Linear(2, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x
        if test_local_backend:
            (local_aot_ort, local_ort) = make_aot_ort(dynamic=True)
        else:
            (local_aot_ort, local_ort) = ('onnxrt', None)
        self._test_model_numerically(MLP(), local_aot_ort, example_args_collection)
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(local_ort, expected_execution_count=len(example_args_collection), number_of_cached_graph_modules=2, number_of_exported_onnx_models_for_all_graph_modules=(1, 1))
if __name__ == '__main__':
    common_utils.run_tests()