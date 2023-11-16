"""Tests for onnx export that don't run the exported model."""
from __future__ import annotations
import contextlib
import io
import itertools
import unittest
import unittest.mock
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnx.numpy_helper
import pytorch_test_common
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.onnx import OperatorExportTypes, symbolic_helper, utils
from torch.onnx._internal import registration
from torch.testing._internal import common_quantization, common_utils, jit_utils

def export_to_onnx(model: Union[torch.nn.Module, torch.jit.ScriptFunction], input: Union[torch.Tensor, Tuple[torch.Tensor]], custom_ops: Optional[Iterable[Union[contextlib.AbstractContextManager, contextlib.ContextDecorator]]]=None, mocks: Optional[Iterable]=None, operator_export_type: torch.onnx.OperatorExportTypes=torch.onnx.OperatorExportTypes.ONNX, opset_version: int=17, **torch_onnx_export_kwargs) -> onnx.ModelProto:
    if False:
        print('Hello World!')
    'Exports `model(input)` to ONNX and returns it.\n\n    Custom operators and/or unittest patches can be used help reproducing specific behaviors.\n\n    Args:\n        model: model to export\n        input: model input with same format as `torch.onnx.export(..,args,...)`\n        custom_ops: list of custom operators to use during export\n        mocks: list of mocks to use during export\n        operator_export_type: export type as described by `torch.onnx.export(...operator_export_type,...)`\n        opset_version: ONNX opset version as described by `torch.onnx.export(...opset_version,...)`\n        torch_onnx_export_kwargs: extra torch.onnx.export kwargs arguments\n    Returns:\n        A valid ONNX model (`onnx.ModelProto`)\n    '
    custom_ops = custom_ops or []
    mocks = mocks or []
    with contextlib.ExitStack() as stack:
        for ctx in itertools.chain(custom_ops, mocks):
            stack.enter_context(ctx)
        f = io.BytesIO()
        torch.onnx.export(model, input, f, operator_export_type=operator_export_type, opset_version=opset_version, **torch_onnx_export_kwargs)
    onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model)
    return onnx_model

@common_utils.instantiate_parametrized_tests
class TestONNXExport(pytorch_test_common.ExportTestCase):

    def test_fuse_addmm(self):
        if False:
            print('Hello World!')

        class AddmmModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, x) + x
        x = torch.ones(3, 3)
        f = io.BytesIO()
        torch.onnx.export(AddmmModel(), x, f, verbose=False)

    def test_onnx_transpose_incomplete_tensor_type(self):
        if False:
            print('Hello World!')

        class Foo(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.contiguous().transpose(0, 1).sum()

        class TraceMe(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo(x)
        tm = TraceMe()
        tm = torch.jit.trace(tm, torch.rand(3, 4))
        f = io.BytesIO()
        torch.onnx.export(tm, (torch.rand(3, 4),), f)

    def test_export_tensoroption_to(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                return 10
            return x[0].clone().detach().cpu() + x
        traced = torch.jit.trace(foo, torch.rand([2]))
        torch.onnx.export_to_pretty_string(traced, (torch.rand([2]),))

    def test_onnx_export_script_module(self):
        if False:
            print('Hello World!')

        class ModuleToExport(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                y = x - x
                return x + x
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    @common_utils.suppress_warnings
    def test_onnx_export_func_with_warnings(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def func_with_warning(inp):
            if False:
                print('Hello World!')
            return torch.nn.functional.sigmoid(inp)

        class WarningTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return func_with_warning(x)
        torch.onnx.export_to_pretty_string(WarningTest(), torch.randn(42), verbose=False)

    def test_onnx_export_script_python_fail(self):
        if False:
            for i in range(10):
                print('nop')

        class PythonModule(torch.jit.ScriptModule):

            @torch.jit.ignore
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mod = PythonModule()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                y = self.mod(x)
                return y + y
        mte = ModuleToExport()
        f = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "Couldn't export Python"):
            torch.onnx.export(mte, (torch.zeros(1, 2, 3),), f, verbose=False)

    def test_onnx_export_script_inline_trace(self):
        if False:
            while True:
                i = 10

        class ModuleToInline(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mod = torch.jit.trace(ModuleToInline(), torch.zeros(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                y = self.mod(x)
                return y + y
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_inline_script(self):
        if False:
            return 10

        class ModuleToInline(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                y = self.mod(x)
                return y + y
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_module_loop(self):
        if False:
            return 10

        class ModuleToExport(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                for _ in range(5):
                    for i in range(3):
                        x = x + i
                return x
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    @common_utils.suppress_warnings
    def test_onnx_export_script_truediv(self):
        if False:
            return 10

        class ModuleToExport(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                z = x.size(0) / 2
                return x + z
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3, dtype=torch.float),), verbose=False)

    def test_onnx_export_script_non_alpha_add_sub(self):
        if False:
            return 10

        class ModuleToExport(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                bs = x.size(0) + 1
                return bs - 1
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.rand(3, 4),), verbose=False)

    def test_onnx_export_script_module_if(self):
        if False:
            return 10

        class ModuleToExport(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if bool(torch.sum(x) > 0):
                    x = torch.neg(x)
                return x
        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_inline_params(self):
        if False:
            while True:
                i = 10

        class ModuleToInline(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = torch.nn.Parameter(torch.ones(3, 3))
                self.unused = torch.nn.Parameter(torch.ones(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, self.m)

        class ModuleToExport(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mod = ModuleToInline()
                self.param = torch.nn.Parameter(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.mod(x)
                return torch.mm(y, self.param)
        mte = ModuleToExport()
        result = mte(torch.zeros(2, 3))
        reference = torch.mm(torch.mm(torch.zeros(2, 3), torch.ones(3, 3)), torch.ones(3, 4))
        self.assertEqual(result, reference)
        torch.onnx.export_to_pretty_string(mte, (torch.ones(2, 3),), verbose=False)

    def test_onnx_export_speculate(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.jit.ScriptModule):

            def __init__(self, m):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = m

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x += x
                c = torch.sum(x) > 4
                if bool(c):
                    if bool(c):
                        y = self.m(x)
                    else:
                        y = self.m(x)
                else:
                    y = self.m(x)
                return y
        linear = torch.jit.trace(torch.nn.Linear(10, 20).float(), torch.zeros(1, 10, dtype=torch.float))

        @torch.jit.script
        def transpose(x):
            if False:
                print('Hello World!')
            return x.t()
        f1 = Foo(transpose)
        f2 = Foo(linear)
        torch.onnx.export_to_pretty_string(f1, (torch.ones(1, 10, dtype=torch.float),))
        torch.onnx.export_to_pretty_string(f2, (torch.ones(1, 10, dtype=torch.float),))

    def test_onnx_export_shape_reshape(self):
        if False:
            print('Hello World!')

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                import torch.onnx.operators
                x = x.repeat(5, 1, 1)
                shape = torch.onnx.operators.shape_as_tensor(x)
                reshaped = torch.onnx.operators.reshape_from_tensor_shape(x, shape)
                return reshaped
        foo = torch.jit.trace(Foo(), torch.zeros(1, 2, 3))
        torch.onnx.export_to_pretty_string(foo, torch.zeros(1, 2, 3))

    def test_listconstruct_erasure(self):
        if False:
            while True:
                i = 10

        class FooMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                mask = x < 0.0
                return x[mask]
        torch.onnx.export_to_pretty_string(FooMod(), (torch.rand(3, 4),), add_node_names=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    def test_export_dynamic_slice(self):
        if False:
            return 10

        class DynamicSliceExportMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                retval = x[0]
                for i in range(x.size(1)):
                    retval += torch.sum(x[0:i], dim=0)
                return retval
        mod = DynamicSliceExportMod()
        input = torch.rand(3, 4, 5)
        torch.onnx.export_to_pretty_string(DynamicSliceExportMod(), (input,), opset_version=10)

    def test_export_dict(self):
        if False:
            while True:
                i = 10

        class DictModule(torch.nn.Module):

            def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
                if False:
                    while True:
                        i = 10
                return {'test_key_out': x_in}
        x_in = torch.tensor(1)
        mod = DictModule()
        mod.train(False)
        torch.onnx.export_to_pretty_string(mod, (x_in,))
        with self.assertRaisesRegex(RuntimeError, 'DictConstruct.+is not supported.'):
            torch.onnx.export_to_pretty_string(torch.jit.script(mod), (x_in,))

    def test_source_range_propagation(self):
        if False:
            i = 10
            return i + 15

        class ExpandingModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.ln = torch.nn.LayerNorm([1])

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return self.ln(input)
        mod = ExpandingModule()
        (graph, _, _) = utils._model_to_graph(mod, (torch.zeros(1),), operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
        for node in graph.nodes():
            self.assertTrue(node.sourceRange())

    @common_utils.skipIfCaffe2
    def test_clip_aten_fallback_due_exception(self):
        if False:
            i = 10
            return i + 15

        def bad_clamp(g, self, min, max):
            if False:
                i = 10
                return i + 15
            return symbolic_helper._onnx_unsupported('Bad boy!')

        class MyClip(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.clamp(x, min=-0.5, max=0.5)
        onnx_model = export_to_onnx(MyClip(), torch.randn(3, 4, requires_grad=True), custom_ops=[common_utils.custom_op('aten::clamp', bad_clamp, 17)], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        self.assertAtenOp(onnx_model, 'clamp', 'Tensor')

    @common_utils.skipIfCaffe2
    def test_clip_aten_fallback_explicit_request(self):
        if False:
            for i in range(10):
                print('nop')

        class MyClip(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.clamp(x, min=-0.5, max=0.5)
        original_get_function_group = registration.registry.get_function_group

        def break_is_registered_op_api(name):
            if False:
                i = 10
                return i + 15
            fake_missing_symbolics = {'aten::clamp'}
            if name in fake_missing_symbolics:
                return None
            return original_get_function_group(name)
        onnx_model = export_to_onnx(MyClip(), torch.randn(3, 4, requires_grad=True), mocks=[unittest.mock.patch('torch.onnx._internal.registration.registry.get_function_group', side_effect=break_is_registered_op_api)], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        self.assertAtenOp(onnx_model, 'clamp', 'Tensor')

    def _helper_test_to_(self, cast_fn: Callable[[torch.Tensor], torch.Tensor]):
        if False:
            for i in range(10):
                print('nop')
        'Helper to test aten::to(device) variants.\n\n        `cast_fn` is converted into a `torch.jit.script`. It wraps `aten::to`\n        during export to preventing the devices to be hard-coded.\n\n        Needed by detectron2 after https://github.com/facebookresearch/detectron2/pull/4132/\n        '
        cast_fn = torch.jit.script(cast_fn)
        onnx_model = export_to_onnx(cast_fn, torch.zeros([1, 3, 32, 32]))
        for n in onnx_model.graph.node:
            self.assertNotEqual(n.op_type, 'To')
            self.assertNotEqual(n.op_type, 'Cast')

    def test_to__cpu_string(self):
        if False:
            return 10

        def cast_cpu_string(src: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            return src.to('cpu')
        self._helper_test_to_(cast_cpu_string)

    def test_to__device_cpu_string(self):
        if False:
            i = 10
            return i + 15

        def cast_device_cpu_string(src: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            return src.to(device='cpu')
        self._helper_test_to_(cast_device_cpu_string)

    def test_script_custom_class_error(self):
        if False:
            i = 10
            return i + 15

        class BoxCoder:

            def __init__(self, bbox_xform_clip: float) -> None:
                if False:
                    while True:
                        i = 10
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                boxes = torch.cat(boxes, dim=0)
                pred_ctr_x = torch.clamp(rel_codes[:, 0::4], max=self.bbox_xform_clip) * boxes[:, 2]
                return pred_ctr_x

        class MyModule(torch.nn.Module):
            __annotations__ = {'box_coder': BoxCoder}

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.box_coder = BoxCoder(1.4)

            def forward(self, box_regression: Tensor, proposals: List[Tensor]):
                if False:
                    i = 10
                    return i + 15
                return self.box_coder.decode(box_regression, proposals)
        model = torch.jit.script(MyModule())
        box_regression = torch.randn([4, 4])
        proposal = [torch.randn(2, 4), torch.randn(2, 4)]
        with self.assertRaises(RuntimeError) as cm:
            onnx_model = io.BytesIO()
            torch.onnx.export(model, (box_regression, proposal), onnx_model)

    def test_initializer_sequence(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self, input_size, hidden_size, num_classes):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        test_model = MyModule(3, 4, 10)
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        named_params_list = [k for (k, v) in test_model.named_parameters()]
        x = torch.randn(32, 3)
        f = io.BytesIO()
        torch.onnx.export(test_model, (x,), f, do_constant_folding=False)
        loaded_model = onnx.load_from_string(f.getvalue())
        actual_list = [p.name for p in loaded_model.graph.initializer]
        assert actual_list == state_dict_list, "Initializers' sequence is not as same as state_dict(). Expected: (" + ', '.join(state_dict_list) + '). Actual:(' + ', '.join(actual_list) + ').'
        assert actual_list == named_params_list, "Initializers' sequence is not as same as named_parameters(). Expected: (" + ', '.join(named_params_list) + '). Actual:(' + ', '.join(actual_list) + ').'

    def test_initializer_sequence_script_model(self):
        if False:
            while True:
                i = 10

        def list_is_expected(short_list, long_list) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            if len(short_list) > len(long_list):
                return False
            for i in range(len(short_list)):
                if short_list[i] not in long_list[i]:
                    return False
            return True

        def loop(x, y):
            if False:
                while True:
                    i = 10
            for i in range(int(y)):
                x = x + i
            return x

        class MyModule(torch.nn.Module):

            def __init__(self, input_size, hidden_size, num_classes):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = loop(x, y)
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        test_model = torch.jit.script(MyModule(3, 4, 10))
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        named_params_list = [k for (k, v) in test_model.named_parameters()]
        x = torch.ones(2, 3, dtype=torch.float)
        y = torch.tensor(5, dtype=torch.long)
        f = io.BytesIO()
        torch.onnx.export(test_model, (x, y), f, do_constant_folding=False)
        loaded_model = onnx.load_from_string(f.getvalue())
        actual_list = [p.name for p in loaded_model.graph.initializer]
        assert list_is_expected(state_dict_list, actual_list), "ScriptModel - Initializers' sequence is not as same as state_dict(). Expected: (" + ', '.join(state_dict_list) + '). Actual:(' + ', '.join(actual_list) + ').'
        assert list_is_expected(named_params_list, actual_list), "ScriptModel - Initializers' sequence is not as same as named_parameters(). Expected: (" + ', '.join(named_params_list) + '). Actual:(' + ', '.join(actual_list) + ').'

    def test_onnx_checker_invalid_graph(self):
        if False:
            return 10

        class CustomAddModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return torch.add(x, y)

        def symbolic_custom_invalid_add(g, input, other, alpha=None):
            if False:
                while True:
                    i = 10
            return g.op('Add', input, other, invalid_attr_i=1)
        torch.onnx.register_custom_op_symbolic('::add', symbolic_custom_invalid_add, opset_version=9)
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        test_model = CustomAddModule()
        f = io.BytesIO()
        try:
            with self.assertRaises(torch.onnx.errors.CheckerError):
                torch.onnx.export(test_model, (x, y), f, opset_version=9)
        finally:
            torch.onnx.unregister_custom_op_symbolic('::add', 9)
        self.assertTrue(f.getvalue(), 'ONNX graph was not exported.')
        loaded_model = onnx.load_from_string(f.getvalue())

    def test_shape_value_map(self):
        if False:
            while True:
                i = 10

        class RSoftMax(torch.nn.Module):

            def __init__(self, radix, cardinality):
                if False:
                    return 10
                super().__init__()
                self.radix = radix
                self.cardinality = cardinality

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                batch = x.size(0)
                x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
                x = F.softmax(x, dim=1)
                x = x.reshape(batch, -1)
                return x
        radix = 2
        cardinality = 1
        x = torch.randn(10, 1, 128, 1)
        f = io.BytesIO()
        torch.onnx.export(RSoftMax(radix, cardinality), (x,), f, input_names=['x'], dynamic_axes={'x': [0]})
        loaded_model = onnx.load_from_string(f.getvalue())
        self.assertEqual(loaded_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, 128)

    def test_onnx_proto_checker(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return 2 * x
        x = torch.randn(1, 2, 3, requires_grad=True)
        f = io.BytesIO()
        torch.onnx.export(Model(), x, f)
        model = onnx.load(f)
        model.ir_version = 0

        def check_proto():
            if False:
                return 10
            torch._C._check_onnx_proto(model.SerializeToString())
        self.assertRaises(RuntimeError, check_proto)

    def test_maintain_dynamic_shapes_of_unreliable_nodes(self):
        if False:
            for i in range(10):
                print('nop')

        def symbolic_pythonop(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
            if False:
                return 10
            return g.op('com.microsoft::PythonOp')
        torch.onnx.register_custom_op_symbolic('prim::PythonOp', symbolic_pythonop, 1)
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, 'prim::PythonOp', 1)
        hidden_size = 48
        max_position_embeddings = 32
        batch_size = 2

        class CustomLayerNorm(torch.autograd.Function):

            @staticmethod
            def forward(ctx, embedding):
                if False:
                    while True:
                        i = 10
                layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
                return layer_norm(embedding)

        class EmbeddingModule(torch.nn.Module):

            def forward(self, embeddings=None):
                if False:
                    print('Hello World!')
                embedding_output = CustomLayerNorm.apply(embeddings)
                query = embedding_output.transpose(0, 1)
                (target_len, batch_size, embedding_dim) = query.size()
                query = query.reshape(target_len, batch_size, embedding_dim)
                return query
        embeddings = torch.randn(batch_size, max_position_embeddings, hidden_size)
        f = io.BytesIO()
        torch.onnx.export(EmbeddingModule().eval(), (embeddings,), f, input_names=['embeddings'], dynamic_axes={'embeddings': {0: 'batch_size', 1: 'max_position_embeddings', 2: 'hidden_size'}}, custom_opsets={'com.microsoft': 1})
        model = onnx.load(io.BytesIO(f.getvalue()))
        const_node = [n for n in model.graph.node if n.op_type == 'Constant']
        self.assertNotEqual(len(const_node), 0)
        for node in const_node:
            for a in node.attribute:
                if a.name == 'value':
                    shape = onnx.numpy_helper.to_array(a.t)
                    self.assertNotEqual(shape.tolist(), [max_position_embeddings, batch_size, hidden_size])

    def test_is_fp_for_C_TypeList(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x.squeeze(1)
                w = x.shape[2]
                pos = x.view(2, -1).argmax(1)
                x_int = pos % w
                y_int = (pos - x_int) // w
                return (y_int, x_int)
        model = torch.jit.script(M())
        inputs = torch.randn(2, 4, 6)
        f = io.BytesIO()
        torch.onnx.export(model, inputs, f, dynamic_axes={'x': [0, 1]}, input_names=['x'])

    def test_dropout_script(self):
        if False:
            while True:
                i = 10
        eg = torch.zeros(1, 2, 3, requires_grad=True)

        @jit_utils._trace(eg)
        def foo(x):
            if False:
                i = 10
                return i + 15
            x = torch.neg(x)
            return F.dropout(x)

        class MyDrop(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return foo(x)
        f = io.BytesIO()
        with warnings.catch_warnings(record=True):
            torch.onnx.export(MyDrop(), (eg,), f, verbose=False)

    def test_pack_padded_pad_packed_trace(self):
        if False:
            i = 10
            return i + 15
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        (T, B, C) = (3, 5, 7)

        class PadPackedWrapper(torch.nn.Module):

            def forward(self, x, seq_lens):
                if False:
                    while True:
                        i = 10
                x = pack_padded_sequence(x, seq_lens)
                (x, _) = pad_packed_sequence(x)
                return x
        x = np.ones((T, B, C))
        seq_lens = np.array([3, 3, 2, 2, 1], dtype=np.int32)
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b]:, b, :] = 0
        seq_lens = torch.from_numpy(seq_lens)
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=True)
        m = PadPackedWrapper()
        m_traced = torch.jit.trace(m, (x, seq_lens))
        y = m(x, seq_lens)
        loss = torch.sum(y)
        loss.backward()
        grad = x.grad.clone()
        x.grad.zero_()
        y_traced = m_traced(x, seq_lens)
        loss_traced = torch.sum(y_traced)
        loss_traced.backward()
        grad_traced = x.grad.clone()
        self.assertEqual(y_traced, x)
        self.assertEqual(y_traced, y)
        self.assertEqual(grad, grad_traced)
        f = io.BytesIO()
        torch.onnx.export(m, (x, seq_lens), f, verbose=False)

    @common_utils.suppress_warnings
    def test_rnn_trace_override(self):
        if False:
            while True:
                i = 10
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        num_layers = 3
        (T, B, C) = (11, 5, 7)

        class RNNTraceWrapper(torch.nn.Module):

            def __init__(self, cell_type):
                if False:
                    while True:
                        i = 10
                super().__init__()
                if cell_type == 'RNN':
                    self.rnn = torch.nn.RNN(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'LSTM':
                    self.rnn = torch.nn.LSTM(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'GRU':
                    self.rnn = torch.nn.GRU(input_size=C, hidden_size=C, num_layers=num_layers)

            def forward(self, x, seq_lens):
                if False:
                    return 10
                x = pack_padded_sequence(x, seq_lens)
                (x, _) = self.rnn(x)
                (x, _) = pad_packed_sequence(x)
                return x
        for cell_type in ['RNN', 'LSTM', 'GRU']:
            x = torch.ones(T, B, C, requires_grad=True)
            seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))
            m = RNNTraceWrapper(cell_type)
            m_traced = torch.jit.trace(m, (x, seq_lens))
            y = m(x, seq_lens)
            loss = torch.sum(y)
            loss.backward()
            grad = x.grad.clone()
            x.grad.zero_()
            y_traced = m_traced(x, seq_lens)
            loss_traced = torch.sum(y_traced)
            loss_traced.backward()
            grad_traced = x.grad.clone()
            self.assertEqual(y_traced, y)
            self.assertEqual(grad, grad_traced)
            f = io.BytesIO()
            torch.onnx.export(m, (x, seq_lens), f, verbose=False)

    def test_pushpackingpastrnn_in_peephole_create_own_gather_input(self):
        if False:
            while True:
                i = 10
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        num_layers = 3
        (T, B, C) = (11, 5, 7)
        mask_start_point = 0

        class LSTMTraceWrapper(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=C, hidden_size=C, num_layers=num_layers)

            def forward(self, x, seq_lens):
                if False:
                    for i in range(10):
                        print('nop')
                mask = torch.arange(mask_start_point, x.shape[1])
                seq_lens = seq_lens[mask]
                x = pack_padded_sequence(x, seq_lens)
                max_batch_size = x.batch_sizes[0]
                hx = torch.randn(num_layers, max_batch_size, C)
                cx = torch.randn(num_layers, max_batch_size, C)
                (x, _) = self.rnn(x, (hx, cx))
                (x, _) = pad_packed_sequence(x)
                return x
        x = torch.ones(T, B, C)
        seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))
        m = LSTMTraceWrapper()
        f = io.BytesIO()
        torch.onnx.export(m, (x, seq_lens), f, verbose=True, input_names=['input', 'seq_len'], dynamic_axes={'input': {1: 'B'}})
        onnx_proto = onnx.load_model_from_string(f.getvalue())
        const_node = []
        constant_input_name = None
        for n in onnx_proto.graph.node:
            if n.op_type == 'Constant':
                const_node.append(n)
            elif n.op_type == 'Range':
                constant_input_name = n.input[0]
        self.assertNotEqual(constant_input_name, None)
        self.assertNotEqual(len(const_node), 0)
        value = None
        for n in const_node:
            if n.output[0] == constant_input_name:
                value = np.frombuffer(n.attribute[0].t.raw_data, dtype=np.int64)
        self.assertEqual(value, 0)

    def test_trace_fork_wait_inline_onnx(self):
        if False:
            i = 10
            return i + 15

        def fork_body(x):
            if False:
                return 10
            return (torch.neg(x), torch.neg(x))

        class MyMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                fut = torch.jit._fork(fork_body, x)
                val = torch.jit._wait(fut)
                return val[1]
        f = io.BytesIO()
        torch.onnx.export(MyMod(), (torch.rand(3, 4),), f)

    def test_trace_detach_onnx_erase(self):
        if False:
            while True:
                i = 10

        class Mod(torch.nn.Module):

            def forward(self, x, w):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.matmul(x, w).detach()
        torch.onnx.export_to_pretty_string(Mod(), (torch.rand(3, 4), torch.rand(4, 5)))

    @common_utils.skipIfNoCaffe2
    def test_caffe2_aten_fallback_must_fallback(self):
        if False:
            i = 10
            return i + 15

        class ModelWithAtenNotONNXOp(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                abcd = x + y
                defg = torch.linalg.qr(abcd)
                return defg
        for operator_export_type in (OperatorExportTypes.ONNX_ATEN, OperatorExportTypes.ONNX_ATEN_FALLBACK):
            x = torch.rand(3, 4)
            y = torch.rand(3, 4)
            f = io.BytesIO()
            torch.onnx.export(ModelWithAtenNotONNXOp(), (x, y), f, do_constant_folding=False, operator_export_type=operator_export_type, opset_version=9)
            onnx_model = onnx.load(io.BytesIO(f.getvalue()))
            self.assertAtenOp(onnx_model, 'linalg_qr')

    @common_utils.skipIfNoCaffe2
    def test_caffe2_onnx_aten_must_not_fallback(self):
        if False:
            for i in range(10):
                print('nop')

        class ModelWithAtenFmod(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.fmod(x, y)
        for operator_export_type in (OperatorExportTypes.ONNX_ATEN_FALLBACK, OperatorExportTypes.ONNX_ATEN):
            x = torch.randn(3, 4, dtype=torch.float32)
            y = torch.randn(3, 4, dtype=torch.float32)
            f = io.BytesIO()
            torch.onnx.export(ModelWithAtenFmod(), (x, y), f, do_constant_folding=False, operator_export_type=operator_export_type, opset_version=10)
            onnx_model = onnx.load(io.BytesIO(f.getvalue()))
            assert onnx_model.graph.node[0].op_type == 'Mod'

    @common_utils.skipIfCaffe2
    def test_aten_fallback_must_fallback(self):
        if False:
            for i in range(10):
                print('nop')

        class ModelWithAtenNotONNXOp(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                abcd = x + y
                defg = torch.linalg.qr(abcd)
                return defg
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        f = io.BytesIO()
        torch.onnx.export(ModelWithAtenNotONNXOp(), (x, y), f, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertAtenOp(onnx_model, 'linalg_qr')

    @common_utils.skipIfCaffe2
    def test_onnx_aten(self):
        if False:
            i = 10
            return i + 15

        class ModelWithAtenFmod(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return torch.fmod(x, y)
        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        f = io.BytesIO()
        torch.onnx.export(ModelWithAtenFmod(), (x, y), f, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertAtenOp(onnx_model, 'fmod', 'Tensor')

    @common_utils.skipIfCaffe2
    def test_onnx_aten_fallback_must_not_fallback(self):
        if False:
            for i in range(10):
                print('nop')

        class ONNXExportable(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.fc1 = torch.nn.Linear(12, 8)
                self.fc2 = torch.nn.Linear(8, 4)
                self.fc3 = torch.nn.Linear(4, 6)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.quant(x)
                x = x.view((-1, 12))
                h = F.relu(self.fc1(x))
                h = F.relu(self.fc2(h))
                h = F.relu(self.fc3(h))
                h = self.dequant(h)
                return h
        dummy_input = torch.randn(12)
        f = io.BytesIO()
        torch.onnx.export(ONNXExportable(), (dummy_input,), f, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        all_aten_nodes = [p for p in onnx_model.graph.node if p.op_type == 'ATen' and p.domain == 'org.pytorch.aten']
        self.assertEqual(len(all_aten_nodes), 0)

    def test_cat_with_empty_tensor(self):
        if False:
            print('Hello World!')

        class NoopConcat(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.cat((torch.Tensor([]), x))
        x = torch.randn(4, 5, 6)
        for opset_version in {9, 11}:
            f = io.BytesIO()
            torch.onnx.export(NoopConcat(), (x,), f, opset_version=opset_version)
            loaded_model = onnx.load_from_string(f.getvalue())
            self.assertEqual(len(loaded_model.graph.output[0].type.tensor_type.shape.dim), 3)
            for (idx, dim) in enumerate(x.shape):
                self.assertEqual(loaded_model.graph.output[0].type.tensor_type.shape.dim[idx].dim_value, dim)

    def test_col2im(self):
        if False:
            return 10
        original_image_inputs = torch.randn((64, 3, 32, 32))
        output_size = tuple(original_image_inputs.shape[2:])
        kernel_size = (1, 2)
        dilation = 3
        padding = 2
        stride = 1
        model_im2col = torch.nn.Unfold(kernel_size, dilation=dilation, padding=padding, stride=stride)
        blocks = model_im2col(original_image_inputs)
        model = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        f = io.BytesIO()
        torch.onnx.export(model, (blocks,), f, opset_version=18)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(onnx_model.graph.node[-1].op_type, 'Col2Im')
        self.assertEqual(onnx_model.graph.node[-1].domain, '')
        self.assertEqual(len(onnx_model.graph.node[-1].input), 3)
        self.assertEqual(onnx_model.graph.node[-1].attribute[0].name, 'dilations')
        self.assertEqual(onnx_model.graph.node[-1].attribute[1].name, 'pads')
        self.assertEqual(onnx_model.graph.node[-1].attribute[2].name, 'strides')

    @unittest.skipIf(not torch.hub._check_module_exists('torch_scatter'), 'torch_scatter not installed.')
    def test_random_namespace_custom_op_is_onnx_exportable(self):
        if False:
            i = 10
            return i + 15
        from torch_scatter import scatter_max

        class MyModel(torch.nn.Module):

            def forward(self, src: torch.Tensor, idx: torch.Tensor):
                if False:
                    print('Hello World!')
                return scatter_max(src, idx)
        m = MyModel().eval()
        src = torch.ones([3, 10], dtype=torch.float32)
        idx = torch.randint(0, 4, [3, 10], dtype=torch.long)

        def sym_scatter_max(g, src, index, dim, out, dim_size):
            if False:
                return 10
            return g.op('torch_scatter::scatter_max', src, index, dim_size_i=-1, outputs=2)
        torch.onnx.register_custom_op_symbolic('torch_scatter::scatter_max', sym_scatter_max, 1)
        with torch.no_grad():
            torch.onnx.export(m, (src, idx), 'mymodel.onnx', verbose=False, opset_version=13, custom_opsets={'torch_scatter': 1}, do_constant_folding=True)

    @common_utils.parametrize('fp8_dtype', [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_fp8_export(self, fp8_dtype: torch.dtype):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.to(torch.float32)
        x = torch.randn(2, 3).to(fp8_dtype)
        f = io.BytesIO()
        torch.onnx.export(Model(), x, f, opset_version=19)
        onnx.checker.check_model(f.getvalue())
        onnx_type = {torch.float8_e4m3fn: 17, torch.float8_e5m2: 19}
        loaded_model = onnx.load_from_string(f.getvalue())
        self.assertEqual(loaded_model.graph.input[0].type.tensor_type.elem_type, onnx_type[fp8_dtype])

class TestQuantizeEagerONNXExport(common_utils.TestCase):

    def _test_lower_graph_impl(self, model, data):
        if False:
            return 10
        model.qconfig = torch.ao.quantization.default_qconfig
        model = torch.ao.quantization.prepare(model)
        model = torch.ao.quantization.convert(model)
        _ = model(data)
        input_names = ['x']

        def _export_to_onnx(model, input, input_names):
            if False:
                i = 10
                return i + 15
            traced = torch.jit.trace(model, input)
            buf = io.BytesIO()
            torch.jit.save(traced, buf)
            buf.seek(0)
            model = torch.jit.load(buf)
            f = io.BytesIO()
            torch.onnx.export(model, input, f, input_names=input_names, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
        _export_to_onnx(model, data, input_names)

    @common_quantization.skipIfNoFBGEMM
    @common_utils.skipIfNoCaffe2
    def test_lower_graph_linear(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.ao.quantization.QuantWrapper(torch.nn.Linear(5, 10, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 2, 5).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)

    @common_quantization.skipIfNoFBGEMM
    @common_utils.skipIfNoCaffe2
    def test_lower_graph_conv2d(self):
        if False:
            while True:
                i = 10
        model = torch.ao.quantization.QuantWrapper(torch.nn.Conv2d(3, 5, 2, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)

    @common_quantization.skipIfNoFBGEMM
    @unittest.skip('onnx opset9 does not support quantize_per_tensor and caffe2     does not support conv3d')
    def test_lower_graph_conv3d(self):
        if False:
            return 10
        model = torch.ao.quantization.QuantWrapper(torch.nn.Conv3d(3, 5, 2, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 3, 6, 6, 6).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)

    @pytorch_test_common.skipIfNoCuda
    def test_composed_layer_norm_small_eps_fp16_keep_double(self):
        if False:
            i = 10
            return i + 15

        class Net(torch.nn.Module):

            def __init__(self, C):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(C, eps=1e-08)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.layer_norm(x)
        (N, C) = (8, 4)
        model = Net(C).cuda().half()
        x = torch.randn(N, C).cuda().half()
        f = io.BytesIO()
        torch.onnx.export(model, x, f, opset_version=14)
        onnx_model = onnx.load_from_string(f.getvalue())
        const_node = [n for n in onnx_model.graph.node if n.op_type == 'Constant']
        self.assertNotEqual(len(const_node), 0)
        double_type_count = 0
        for node in const_node:
            for a in node.attribute:
                if a.name == 'value' and a.t.data_type == 11:
                    double_type_count += 1
        self.assertNotEqual(double_type_count, 0)

    @pytorch_test_common.skipIfNoCuda
    def test_aten_device_with_index(self):
        if False:
            i = 10
            return i + 15
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        model = torch.compile(model, backend='onnxrt')
        model = model.eval()
        device = 'cuda:0'
        model = model.to(device)
        ids = tokenizer.batch_encode_plus(['This is a test'], return_tensors='pt').to(device)
        with torch.no_grad():
            _ = model(**{'input_ids': ids['input_ids'], 'attention_mask': ids['attention_mask'], 'decoder_input_ids': ids['input_ids'], 'decoder_attention_mask': ids['attention_mask']})
if __name__ == '__main__':
    common_utils.run_tests()