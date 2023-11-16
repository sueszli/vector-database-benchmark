from __future__ import annotations
import itertools
import math
import operator
import os
import tempfile
import unittest
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type
import onnx_test_common
import onnxruntime
import parameterized
import pytorch_test_common
import torch
import torch.onnx
import transformers
from torch import nn
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, exporter
from torch.onnx._internal.fx import fx_symbolic_graph_extractor, patcher, serialization as fx_serialization
from torch.testing._internal import common_utils
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

def _parameterized_class_attrs_and_values():
    if False:
        for i in range(10):
            print('nop')
    input_values = []
    input_values.extend(itertools.product((True, False), (True, False)))
    return {'attrs': ['op_level_debug', 'dynamic_shapes'], 'input_values': input_values}

def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    if False:
        return 10
    'Combine class name with the parameterized arguments.\n\n    This function is passed to `parameterized.parameterized_class` as the\n    `class_name_func` argument.\n    '
    suffixes = []
    for (k, v) in input_dicts.items():
        suffixes.append(f'{k}_{v}')
    return f"{cls.__name__}_{'_'.join(suffixes)}"

@parameterized.parameterized_class(**_parameterized_class_attrs_and_values(), class_name_func=_parameterize_class_name)
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    op_level_debug: bool
    dynamic_shapes: bool

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.ort_version = onnxruntime.__version__

    def test_simple_function(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            y = x + 1.0
            z = y.relu()
            return (y, z)
        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))

    @pytorch_test_common.xfail('AssertionError: Dynamo input/output is not consistent with traced input/output. Ref: https://github.com/pytorch/pytorch/issues/96379')
    def test_func_with_args_and_tensor_kwargs(self):
        if False:
            while True:
                i = 10

        def func(x, b=torch.tensor(1.0)):
            if False:
                i = 10
                return i + 15
            y = x + b
            z = y.relu()
            return (y, z)
        tensor_x = torch.randn(1, 2, 3, dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x, torch.tensor(8.0)))
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,), input_kwargs={'b': torch.tensor(5.0)})

    @pytorch_test_common.skip_dynamic_fx_test("sympy operation tests don't need dynamic shape")
    def test_sympy_operatons_return_numeric(self):
        if False:
            while True:
                i = 10

        def func(x, y):
            if False:
                while True:
                    i = 10
            return (torch.tensor([operator.add(x.item(), y.item())]), torch.tensor([operator.sub(x.item(), y.item())]), torch.tensor([operator.mul(x.item(), y.item())]), torch.tensor([operator.truediv(x.item(), y.item())]), torch.tensor([operator.floordiv(x.item(), y.item())]), torch.tensor([operator.pow(x.item(), y.item())]), torch.tensor([operator.abs(x.item())]), torch.tensor([operator.neg(x.item())]), torch.tensor([math.ceil(x.item())]), torch.tensor([math.floor(x.item())]))
        x = torch.randn(1, dtype=torch.float32)
        y = torch.randn(1, dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x, y))

    @pytorch_test_common.xfail('https://github.com/pytorch/pytorch/issues/99534Non-tensor input is not traceable in dynamo.')
    def test_xfail_func_with_non_tensor_args(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x, b=1.0):
            if False:
                for i in range(10):
                    print('nop')
            y = x + b
            z = y.relu()
            return (y, z)
        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)
        onnx_program = torch.onnx.dynamo_export(func, tensor_x, 8.0, export_options=torch.onnx.ExportOptions(op_level_debug=self.op_level_debug, dynamic_shapes=self.dynamic_shapes))
        onnx_test_common.assert_dynamic_shapes(onnx_program, self.dynamic_shapes)
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, 8.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 8.0))
        ort_outputs = onnx_test_common.run_ort(onnx_program, onnx_format_args)
        for (ref_output, ort_output) in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))
        onnx_format_args = onnx_program.adapt_torch_inputs_to_onnx(tensor_x, 9.0)
        ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(func(tensor_x, 9.0))
        _ = onnx_test_common.run_ort(onnx_program, onnx_format_args)
        for (ref_output, ort_output) in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    def test_func_with_nested_input_structure(self):
        if False:
            print('Hello World!')

        def func(x_dict: Dict[str, torch.Tensor], y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], z_list: List[List[torch.Tensor]]):
            if False:
                print('Hello World!')
            if 'a' in x_dict:
                x = x_dict['a']
            elif 'b' in x_dict:
                x = x_dict['b']
            else:
                x = torch.randn(3)
            (y1, (y2, y3)) = y_tuple
            z = x + y1 + y2 + y3
            for z_sub_list in z_list:
                z = z + torch.stack(z_sub_list).sum()
            return z
        x_dict = {'a': torch.randn(3), 'c': torch.randn(3)}
        y_tuple = (torch.randn(3), (torch.randn(3), torch.randn(3)))
        z_list = [[torch.randn(3), torch.randn(3)], [torch.randn(3), torch.randn(3), torch.randn(3)]]
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x_dict, y_tuple, z_list))

    def test_func_with_nested_output_structure(self):
        if False:
            print('Hello World!')

        def func(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            x = x + y
            y = y + z
            z = x + y
            out1 = (x, (y, z))
            out2 = [[x, y], [y, z]]
            out3 = {'z': z, 'x': x}
            return (out1, out2, out3)
        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x, y, z))

    def test_mnist(self):
        if False:
            for i in range(10):
                print('nop')

        class MNISTModel(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                if False:
                    while True:
                        i = 10
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = torch.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = torch.log_softmax(tensor_x, dim=1)
                return output
        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(MNISTModel(), (tensor_x,))

    def test_log_sigmoid(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = torch.nn.LogSigmoid()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.m(x)
        input = torch.randn(2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (input,))

    @skip_if_no_torchvision
    def test_resnet18(self):
        if False:
            return 10
        model = torchvision.models.resnet18(weights=None).eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(model, (dummy_input,))

    @pytorch_test_common.skip_dynamic_fx_test('[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: arg0 for the following indices index: 0 Got: 3 Expected: 1')
    @skip_if_no_torchvision
    def test_shufflenet_v2(self):
        if False:
            i = 10
            return i + 15
        model = torchvision.models.shufflenet_v2_x0_5(weights=None).eval()
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=False)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(model, (dummy_input,), additional_test_inputs=[((test_inputs,),)], rtol=0.001, atol=1e-05)

    def test_add(self):
        if False:
            while True:
                i = 10

        class DynamicAdd(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return torch.ops.aten.add(x, y)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        another_x = torch.randn(3, 4)
        another_y = torch.randn(3, 4)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(DynamicAdd(), (x, y), additional_test_inputs=[((another_x, another_y),)])

    def test_sigmoid_add(self):
        if False:
            i = 10
            return i + 15

        class DynamicAdd(torch.nn.Module):

            def __init__(self, *args, **kwargs) -> None:
                if False:
                    print('Hello World!')
                super().__init__(*args, **kwargs)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                z = torch.ops.aten.add(x, y)
                return self.sigmoid(z)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        x = x[1:, :]
        y = y[1:, :]
        input_x = torch.randn(1, 4)
        input_y = torch.randn(1, 4)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(DynamicAdd(), (x, y), additional_test_inputs=[((input_x, input_y),)])

    def test_matmul(self):
        if False:
            for i in range(10):
                print('nop')

        class DynamicMatMul(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.ops.aten.matmul(x, y)
        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(DynamicMatMul(), (x, y), additional_test_inputs=[((input_x, input_y),)])

    @pytorch_test_common.skip_dynamic_fx_test('fx graph does not capture symbolic value for aten::scalar_tensor.')
    def test_scalar_tensor(self):
        if False:
            while True:
                i = 10

        class test(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (torch.scalar_tensor(x.size(0)), torch.scalar_tensor(x.size(1), dtype=torch.int64))
        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(test(), (x,), additional_test_inputs=[((y,),)])

    def test_transpose_infer_shape(self):
        if False:
            i = 10
            return i + 15

        class TransposeModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                return x.transpose(0, 1)
        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(TransposeModule(), (x,), additional_test_inputs=[((y,),)])

    @pytorch_test_common.xfail('torch._dynamo.exc.Unsupported: guard on data-dependent symbolic int/float')
    def test_squeeze_runtime_dim(self):
        if False:
            i = 10
            return i + 15

        class Squeeze(torch.nn.Module):

            def forward(self, d1, d2):
                if False:
                    while True:
                        i = 10
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)
        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Squeeze(), (d1, d4), additional_test_inputs=[((d3, d4),)])
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Squeeze(), (d3, d4), additional_test_inputs=[((d1, d3),)])

    def test_slice(self):
        if False:
            return 10

        class DynamicSliceExportMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                results = []
                for i in range(4):
                    results.append(x[:x.size(0) - i, i:x.size(2), i:3])
                return tuple(results)
        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(DynamicSliceExportMod(), (x,), additional_test_inputs=[((y,),)])

    def test_mutation(self):
        if False:
            print('Hello World!')

        class MutationModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x.view(3, 2, -1).add_(2.0)
                return x
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(MutationModel(), (torch.randn(12),), has_mutation=True)

    def test_arange(self):
        if False:
            print('Hello World!')

        class ArangeModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return (torch.arange(input.shape[0]), torch.arange(12), torch.arange(start=input.shape[0], end=input.shape[0] + 5))
        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(ArangeModel(), (x,), additional_test_inputs=[((y,),)])

    @pytorch_test_common.skip_dynamic_fx_test("[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Slice node. Name:'_inline_aten_slice_scattern13' Status Message: slice.cc:193 FillVectorsFromInput Starts must be a 1-D array")
    def test_expand_as_fill_zero(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x[:, x.size(0):] = 0
                return x
        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (x,), additional_test_inputs=[((x2,),)])

    @pytorch_test_common.xfail('[ONNXRuntimeError] : 1 : FAIL : Type Error: Type (tensor(float)) of output arg (copy) of node (n0__4) does not match expected type (tensor(int64))')
    def test_expand_as_fill_tensor(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x[:, x.size(0):] = torch.tensor([1, 2, 3])
                return x
        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (x,), additional_test_inputs=[((x2,),)])

    @pytorch_test_common.xfail("RuntimeError: at::functionalization::impl::isFunctionalTensor(self_) INTERNAL ASSERT FAILED at '/path/to/pytorch/torch/csrc/autograd/python_torch_functions_manual.cpp':514, please report a bug to PyTorch.")
    def test_expand_as_fill_seperate_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)
        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (x,), additional_test_inputs=[((x2,),)])

    def test_view_dynamic_zero_dim(self):
        if False:
            i = 10
            return i + 15

        class ViewModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                input = input.view(-1, 2)
                return input.view(1, -1)
        x = torch.ones(2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(ViewModel(), (x,), skip_dynamic_shapes_check=True)

    def test_flatten_dynamic_axes(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.flatten(x, start_dim=2, end_dim=3)
        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(model, (x,), additional_test_inputs=[((y,),)])

    def test_none_input(self):
        if False:
            return 10

        class NoneInputModel(torch.nn.Module):

            def forward(self, x: torch.Tensor, y: Optional[torch.Tensor], z: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                if y is None:
                    return x + z
                return x + y + z
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(NoneInputModel(), (torch.randn(1, 2), None, torch.randn(1, 2)))

    def test_operator_with_data_dependent_output(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            return x + torch.full(x.shape, torch.tensor(torch.finfo(x.dtype).min))
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (torch.randn(3, 4),))

    def test_operator_with_scalar_output(self):
        if False:
            return 10

        def func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x.item() + y
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (torch.tensor([1]), torch.randn(3, 4)))

    def test_operator_with_dynamic_output_shape(self):
        if False:
            while True:
                i = 10

        def func(x):
            if False:
                while True:
                    i = 10
            return x.nonzero()
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (torch.randn(3, 4),))

    def test_gpt2_tiny_from_config(self):
        if False:
            while True:
                i = 10
        config = transformers.GPT2Config(num_hidden_layers=4, vocab_size=8096, hidden_size=16, intermediate_size=16, max_position_embeddings=512, num_attention_heads=2, hidden_dropout_prob=0.0, attention_dropout_prob=0.0)
        model = transformers.GPT2Model(config).eval()

        def input_generator(batch: int, seq: int):
            if False:
                i = 10
                return i + 15
            input_ids = torch.randint(0, 8096, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)
            return (input_ids, attention_mask, position_ids)
        (input_ids, attention_mask, position_ids) = input_generator(2, 128)
        (another_input_ids, another_attention_mask, another_position_ids) = input_generator(3, 256)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(model, (input_ids,), input_kwargs={'attention_mask': attention_mask, 'position_ids': position_ids}, additional_test_inputs=[((another_input_ids,), {'attention_mask': another_attention_mask, 'position_ids': another_position_ids})])

    def test_prims_device_put(self):
        if False:
            print('Hello World!')

        class CustomModule(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = torch.ops.prims.device_put(x, 'cpu')
                return x
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(CustomModule(), (torch.randn(1, 2, 3),))

    @_beartype.beartype
    def _test_fx_symbolic_tracer_large_scale_exporter(self, model_name: str, create_model: Callable, create_args: Callable, create_pytorch_only_kwargs: Callable):
        if False:
            i = 10
            return i + 15
        "Test helper for large-scale exporter.\n\n        Arguments:\n            model_name: Name of the model. It used to name temporary files.\n            create_model: A function that creates a model. It should always create the same model.\n            create_args: A function that creates random input arguments for the model.\n            create_pytorch_only_kwargs: A function that creates kwargs for calling PyTorch model with real tensors.\n\n        This test contains several steps.\n\n        1. Create a toy model.\n        2. Save the toy's state (parameters) to a file. This is for simulating a checkpoint file.\n        3. Load it back and export it to ONNX with large-scale exporter.\n            All operations (including model loading) are done under\n            FakeTensorMode so no real tensor is created and no real\n            computation happens.\n        4. The ONNX model generated in step 3 doesn't contain parameters,\n            and this step adds them as external data and save a new ONNX model.\n        5. Run PyTorch and ONNX models and compare their results.\n        "
        model = create_model()
        with tempfile.NamedTemporaryFile(prefix=model_name, suffix='.pt') as tmp_file, tempfile.TemporaryDirectory(suffix='large_scale_export') as tmp_folder:
            torch.save(model.state_dict(), tmp_file.name)
            ftm = fake_tensor.FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
            ctx = patcher.ONNXTorchPatcher()
            with ctx, ftm:
                fake_model = create_model()
                fake_model.load_state_dict(torch.load(tmp_file.name))
                fake_args = create_args()
                options = torch.onnx.ExportOptions(dynamic_shapes=self.dynamic_shapes, op_level_debug=self.op_level_debug)
                export_options = exporter.ResolvedExportOptions(options)
                export_options.fx_tracer = fx_symbolic_graph_extractor.FXSymbolicTracer()
                onnx_program = torch.onnx.dynamo_export(fake_model, *fake_args, export_options=export_options)
                onnx_model = onnx_program.model_proto
            onnx_test_common.assert_dynamic_shapes(onnx_program, self.dynamic_shapes)
            onnx_model_location = model_name + '_external_data.onnx'
            onnx_initializer_location = model_name + '_initializers'
            fx_serialization.save_model_with_external_data(tmp_folder, onnx_model_location, onnx_initializer_location, tuple(ctx.paths), onnx_model, rename_initializer=True)
            args = create_args()
            kwargs = create_pytorch_only_kwargs()
            ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(model(*args, **kwargs))
            args_not_none = onnx_program.adapt_torch_inputs_to_onnx(*args)
            args_not_none = args_not_none[:len(args) - len(kwargs)]
            ort_outputs = onnx_test_common.run_ort(os.path.join(tmp_folder, onnx_model_location), args_not_none)
            assert len(ref_outputs) == len(ort_outputs)
            for (ref_output, ort_output) in zip(ref_outputs, ort_outputs):
                torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    @pytorch_test_common.skip_dynamic_fx_test('FakeTensor exporting is not supported by dynamic axes.')
    def test_fx_symbolic_tracer_large_scale_exporter_with_toy_mlp(self):
        if False:
            print('Hello World!')

        class MLPModel(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc0 = nn.Linear(8, 8, bias=True)
                self.fc1 = nn.Linear(8, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)
                self.fc3 = nn.Linear(2, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                if False:
                    return 10
                tensor_x = self.fc0(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc3(tensor_x)
                return output

        def create_model() -> nn.Module:
            if False:
                print('Hello World!')
            return MLPModel()

        def create_args():
            if False:
                return 10
            return (torch.rand((97, 8), dtype=torch.float32),)

        def create_pytorch_only_extra_kwargs():
            if False:
                print('Hello World!')
            return {}
        self._test_fx_symbolic_tracer_large_scale_exporter('toy_mlp1', create_model, create_args, create_pytorch_only_extra_kwargs)

    @pytorch_test_common.xfail("[ONNXRuntimeError] : 1 : FAIL : Type Error: Data in initializer 'h_0_attn_bias' has element type tensor(uint8) but usage of initializer in graph expects tensor(bool)https://github.com/huggingface/transformers/issues/21013")
    @pytorch_test_common.skip_dynamic_fx_test('FakeTensor exporting is not supported by dynamic axes.')
    def test_fx_symbolic_tracer_large_scale_exporter_with_tiny_gpt2(self):
        if False:
            i = 10
            return i + 15
        model_name = 'sshleifer/tiny-gpt2'
        device = 'cpu'

        def create_model() -> nn.Module:
            if False:
                for i in range(10):
                    print('nop')
            return transformers.AutoModel.from_pretrained(model_name).to(device).eval()

        def create_args():
            if False:
                return 10
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            kwargs = tokenizer('Hello world!', return_tensors='pt')
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
            return (input_ids, None, attention_mask)

        def create_pytorch_only_extra_kwargs():
            if False:
                for i in range(10):
                    print('nop')
            return {'return_dict': False}
        self._test_fx_symbolic_tracer_large_scale_exporter('tiny_gpt2', create_model, create_args, create_pytorch_only_extra_kwargs)

    def test_exported_program_as_input(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x + 1.0
        x = torch.randn(1, 1, 2, dtype=torch.float)
        exported_program = torch.export.export(Model(), args=(x,))
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(exported_program, (x,), skip_dynamic_shapes_check=True)

    def test_exported_program_as_input_from_file(self):
        if False:
            return 10
        import tempfile

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x + 1.0
        x = torch.randn(1, 1, 2, dtype=torch.float)
        exported_program = torch.export.export(Model(), args=(x,))
        with tempfile.NamedTemporaryFile(suffix='.pte') as f:
            torch.export.save(exported_program, f.name)
            del exported_program
            loaded_exported_program = torch.export.load(f.name)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(loaded_exported_program, (x,), skip_dynamic_shapes_check=True)

def _parameterized_class_attrs_and_values_with_fake_options():
    if False:
        return 10
    input_values = []
    input_values.extend(itertools.product((True, False), (True, False), (True, False), (True, False)))
    return {'attrs': ['op_level_debug', 'dynamic_shapes', 'load_checkpoint_during_init', 'export_within_fake_mode'], 'input_values': input_values}

@parameterized.parameterized_class(**_parameterized_class_attrs_and_values_with_fake_options(), class_name_func=_parameterize_class_name)
class TestFxToOnnxFakeTensorWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    """ONNX export test for specific Fake Tensor scenarios

    TODO: Should we merge this with  `TestFxToOnnxWithOnnxRuntime`? Considerably increases export time
    """
    op_level_debug: bool
    dynamic_shapes: bool
    load_checkpoint_during_init: bool
    export_within_fake_mode: bool

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.ort_version = onnxruntime.__version__

    @_beartype.beartype
    def _test_fake_tensor_mode_exporter(self, model_name: str, create_model: Callable, create_args: Callable, create_kwargs: Callable, load_checkpoint_during_init: bool, export_within_fake_mode: bool):
        if False:
            print('Hello World!')
        "Test helper for FakeTensorMode-enabled exporter.\n\n        Arguments:\n            model_name: Name of the model. It used to name temporary files.\n            create_model: A function that creates a model.\n            create_args: A function that creates positional inputs for the model.\n            create_kwargs: A function that creates keyword inputs for ther model.\n            load_checkpoint_during_init: Whether to load a checkpoint during model initialization.\n                (after or during model creation, but before exporting starts)\n            export_within_fake_mode: Whether to call torch.onnx._dynamo_export within torch._subclasses.FakeTensorMode\n\n        This test contains several steps.\n\n        1. Create a toy model.\n        2. Save the toy's state (parameters) to a file. This is for simulating a checkpoint file.\n        3. Load it back and export it to ONNX with Fake Mode enabled.\n            Because all operations (including model and input loading) are done under\n            FakeTensorMode, no real tensor are created and no real computation happens.\n        4. The ONNX model generated in step 3 doesn't contain parameters,\n            and this step adds them as external data on an ONNX model.\n        5. Run PyTorch and ONNX models and compare their results.\n        "
        real_model = create_model()
        with tempfile.NamedTemporaryFile(prefix=model_name, suffix='.pt') as tmp_checkpoint_file:
            state_dict = real_model.state_dict()
            torch.save(state_dict, tmp_checkpoint_file.name)
            with torch.onnx.enable_fake_mode() as fake_context:
                fake_args = create_args()
                fake_kwargs = create_kwargs()
                fake_model = create_model()
                if load_checkpoint_during_init:
                    fake_model.load_state_dict(torch.load(tmp_checkpoint_file.name))
                export_options = torch.onnx.ExportOptions(dynamic_shapes=self.dynamic_shapes, op_level_debug=self.op_level_debug, fake_context=fake_context)
                if export_within_fake_mode:
                    onnx_program = torch.onnx.dynamo_export(fake_model, *fake_args, **fake_kwargs, export_options=export_options)
            if not export_within_fake_mode:
                onnx_program = torch.onnx.dynamo_export(fake_model, *fake_args, **fake_kwargs, export_options=export_options)
            onnx_test_common.assert_dynamic_shapes(onnx_program, self.dynamic_shapes)
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp_onnx_file:
                onnx_program.save(tmp_onnx_file.name, model_state_dict=tmp_checkpoint_file.name)
                args = create_args()
                kwargs = create_kwargs()
                ref_outputs = onnx_program.adapt_torch_outputs_to_onnx(real_model(*args, **kwargs))
                args_not_none = onnx_program.adapt_torch_inputs_to_onnx(*args, **kwargs)
                ort_outputs = onnx_test_common.run_ort(tmp_onnx_file.name, args_not_none)
                assert len(ref_outputs) == len(ort_outputs)
                for (ref_output, ort_output) in zip(ref_outputs, ort_outputs):
                    torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    def test_fake_tensor_mode_simple(self):
        if False:
            i = 10
            return i + 15

        def create_model() -> nn.Module:
            if False:
                return 10

            class Model(torch.nn.Module):

                def __init__(self) -> None:
                    if False:
                        return 10
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2)

                def forward(self, x):
                    if False:
                        print('Hello World!')
                    out = self.linear(x)
                    return out
            return Model()

        def create_args():
            if False:
                return 10
            return (torch.rand(5, 2, 2),)

        def create_kwargs():
            if False:
                for i in range(10):
                    print('nop')
            return {}
        self._test_fake_tensor_mode_exporter('simple', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    @pytorch_test_common.xfail("[ONNXRuntimeError] : 1 : FAIL : Type Error: Data in initializer 'h_0_attn_bias' has element type tensor(uint8) but usage of initializer in graph expects tensor(bool)https://github.com/huggingface/transformers/issues/21013This can be addressed by using GPT2Config, but it is not now supported by FakeTensor exporting.")
    def test_large_scale_exporter_with_tiny_gpt2(self):
        if False:
            for i in range(10):
                print('nop')
        model_name = 'sshleifer/tiny-gpt2'
        device = 'cpu'

        def create_model() -> nn.Module:
            if False:
                print('Hello World!')
            return transformers.AutoModel.from_pretrained(model_name).to(device).eval()

        def create_args():
            if False:
                while True:
                    i = 10
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            kwargs = tokenizer('Hello world!', return_tensors='pt')
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
            return (input_ids, None, attention_mask)

        def create_kwargs():
            if False:
                return 10
            return {'return_dict': False}
        self._test_fake_tensor_mode_exporter('tiny_gpt2', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    def test_large_scale_exporter_with_toy_mlp(self):
        if False:
            print('Hello World!')

        class MLPModel(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fc0 = nn.Linear(8, 8, bias=True)
                self.fc1 = nn.Linear(8, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)
                self.fc3 = nn.Linear(2, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                if False:
                    while True:
                        i = 10
                tensor_x = self.fc0(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc3(tensor_x)
                return output

        def create_model() -> nn.Module:
            if False:
                i = 10
                return i + 15
            return MLPModel()

        def create_args():
            if False:
                while True:
                    i = 10
            return (torch.rand((97, 8), dtype=torch.float32),)

        def create_kwargs():
            if False:
                i = 10
                return i + 15
            return {}
        self._test_fake_tensor_mode_exporter('toy_mlp1', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    def test_fake_tensor_mode_huggingface_google_t5(self):
        if False:
            for i in range(10):
                print('nop')
        config = transformers.T5Config(vocab_size=8096, d_model=64, num_layers=2, num_heads=2)
        (batch, seq) = (4, 256)

        def create_args():
            if False:
                return 10
            return tuple()

        def create_kwargs():
            if False:
                print('Hello World!')
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones((batch, seq), dtype=torch.bool)
            decoder_input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids}

        def create_model():
            if False:
                print('Hello World!')
            return transformers.T5Model(config).eval()
        self._test_fake_tensor_mode_exporter('huggingface_google_t5', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    def test_fake_tensor_mode_huggingface_openai_whisper(self):
        if False:
            for i in range(10):
                print('nop')
        config = transformers.WhisperConfig(vocab_size=8096, num_mel_bins=40, encoder_layers=2, encoder_attention_heads=2, decoder_layers=2, decoder_attention_heads=2, decoder_ffn_dim=384, encoder_ffn_dim=384, d_model=64, decoder_start_token_id=8001, pad_token_id=8000, bos_token_id=8000, eos_token_id=8000, begin_suppress_tokens=[220, 8000])
        feature_extractor = transformers.WhisperFeatureExtractor(feature_size=40)
        device = 'cpu'
        batch = 4

        def create_model() -> nn.Module:
            if False:
                for i in range(10):
                    print('nop')
            return transformers.AutoModel.from_config(config).to(device).eval()

        def create_args():
            if False:
                return 10
            return ()

        def create_kwargs():
            if False:
                print('Hello World!')
            input_features = torch.randn((batch, feature_extractor.feature_size, feature_extractor.nb_max_frames), dtype=torch.float32)
            decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
            return {'input_features': input_features, 'decoder_input_ids': decoder_input_ids, 'return_dict': False}
        self._test_fake_tensor_mode_exporter('openai_whisper', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    @pytorch_test_common.xfail('AssertionError: whole graph export entails exactly one guard export')
    def test_fake_tensor_mode_huggingface_mosaicml_mpt(self):
        if False:
            return 10
        config = transformers.MptConfig(vocab_size=8096, d_model=64, n_heads=2, n_layers=3)
        (batch, seq) = (4, 256)

        def create_args():
            if False:
                return 10
            return tuple()

        def create_kwargs():
            if False:
                i = 10
                return i + 15
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        def create_model():
            if False:
                print('Hello World!')
            return transformers.MptModel(config).eval()
        self._test_fake_tensor_mode_exporter('huggingface_mosaicml_mpt', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)

    @pytorch_test_common.skip_dynamic_fx_test('RuntimeError:: SymIntArrayRef expected to contain only concrete integers')
    def test_fake_tensor_mode_huggingface_bigscience_bloom_560m(self):
        if False:
            while True:
                i = 10
        config = transformers.BloomConfig()
        (batch, seq) = (4, 256)

        def create_args():
            if False:
                return 10
            return tuple()

        def create_kwargs():
            if False:
                i = 10
                return i + 15
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        def create_model():
            if False:
                return 10
            return transformers.BloomModel(config).eval()
        self._test_fake_tensor_mode_exporter('huggingface_bigscience_bloom_560m', create_model, create_args, create_kwargs, load_checkpoint_during_init=self.load_checkpoint_during_init, export_within_fake_mode=self.export_within_fake_mode)
if __name__ == '__main__':
    common_utils.run_tests()