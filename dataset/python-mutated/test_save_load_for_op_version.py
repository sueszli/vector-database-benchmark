from itertools import product as product
import io
import os
import sys
import hypothesis.strategies as st
from hypothesis import example, settings, given
from typing import Union
import torch
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
from torch.jit.mobile import _load_for_lite_interpreter
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestSaveLoadForOpVersion(JitTestCase):

    def _save_load_module(self, m):
        if False:
            return 10
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        return torch.jit.load(buffer)

    def _save_load_mobile_module(self, m):
        if False:
            for i in range(10):
                print('nop')
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        return _load_for_lite_interpreter(buffer)

    def _try_fn(self, fn, *args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def _verify_no(self, kind, m):
        if False:
            return 10
        self._verify_count(kind, m, 0)

    def _verify_count(self, kind, m, count):
        if False:
            while True:
                i = 10
        node_count = sum((str(n).count(kind) for n in m.graph.nodes()))
        self.assertEqual(node_count, count)
    '\n    Tests that verify Torchscript remaps aten::div(_) from versions 0-3\n    to call either aten::true_divide(_), if an input is a float type,\n    or truncated aten::divide(_) otherwise.\n    NOTE: currently compares against current div behavior, too, since\n      div behavior has not yet been updated.\n    '

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_tensor(self, sample_input):
        if False:
            print('Hello World!')

        def historic_div(self, other):
            if False:
                return 10
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode='trunc')

        class MyModule(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                result_0 = a / b
                result_1 = torch.div(a, b)
                result_2 = a.div(b)
                return (result_0, result_1, result_2)
        try:
            v3_mobile_module = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_tensor_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module = self._save_load_mobile_module(MyModule)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                if False:
                    for i in range(10):
                        print('nop')
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)
                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    for result in m_results:
                        self.assertEqual(result, fn_result)
            _helper(v3_mobile_module, historic_div)
            _helper(current_mobile_module, torch.div)

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_tensor_inplace(self, sample_input):
        if False:
            for i in range(10):
                print('nop')

        def historic_div_(self, other):
            if False:
                print('Hello World!')
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide_(other)
            return self.divide_(other, rounding_mode='trunc')

        class MyModule(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    return 10
                a /= b
                return a
        try:
            v3_mobile_module = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_tensor_inplace_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module = self._save_load_mobile_module(MyModule)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                if False:
                    for i in range(10):
                        print('nop')
                fn_result = self._try_fn(fn, a.clone(), b)
                m_result = self._try_fn(m, a, b)
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)
                    self.assertEqual(m_result, a)
            _helper(v3_mobile_module, historic_div_)
            a = torch.tensor((val_a,))
            _helper(current_mobile_module, torch.Tensor.div_)

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_tensor_out(self, sample_input):
        if False:
            return 10

        def historic_div_out(self, other, out):
            if False:
                while True:
                    i = 10
            if self.is_floating_point() or other.is_floating_point() or out.is_floating_point():
                return torch.true_divide(self, other, out=out)
            return torch.divide(self, other, out=out, rounding_mode='trunc')

        class MyModule(torch.nn.Module):

            def forward(self, a, b, out):
                if False:
                    i = 10
                    return i + 15
                return a.div(b, out=out)
        try:
            v3_mobile_module = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_tensor_out_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module = self._save_load_mobile_module(MyModule)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))
            for out in (torch.empty((1,)), torch.empty((1,), dtype=torch.long)):

                def _helper(m, fn):
                    if False:
                        return 10
                    fn_result = None
                    if fn is torch.div:
                        fn_result = self._try_fn(fn, a, b, out=out.clone())
                    else:
                        fn_result = self._try_fn(fn, a, b, out.clone())
                    m_result = self._try_fn(m, a, b, out)
                    if isinstance(m_result, Exception):
                        self.assertTrue(fn_result, Exception)
                    else:
                        self.assertEqual(m_result, fn_result)
                        self.assertEqual(m_result, out)
                _helper(v3_mobile_module, historic_div_out)
                _helper(current_mobile_module, torch.div)

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_scalar(self, sample_input):
        if False:
            i = 10
            return i + 15

        def historic_div_scalar_float(self, other: float):
            if False:
                for i in range(10):
                    print('nop')
            return torch.true_divide(self, other)

        def historic_div_scalar_int(self, other: int):
            if False:
                i = 10
                return i + 15
            if self.is_floating_point():
                return torch.true_divide(self, other)
            return torch.divide(self, other, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):

            def forward(self, a, b: float):
                if False:
                    i = 10
                    return i + 15
                return a / b

        class MyModuleInt(torch.nn.Module):

            def forward(self, a, b: int):
                if False:
                    for i in range(10):
                        print('nop')
                return a / b
        try:
            v3_mobile_module_float = _load_for_lite_interpreter(pytorch_test_dir + '/jit/fixtures/test_versioned_div_scalar_float_v2.ptl')
            v3_mobile_module_int = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_int_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                if False:
                    return 10
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)
            if isinstance(b, float):
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, historic_div_scalar_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_scalar_reciprocal(self, sample_input):
        if False:
            for i in range(10):
                print('nop')

        def historic_div_scalar_float_reciprocal(self, other: float):
            if False:
                for i in range(10):
                    print('nop')
            return other / self

        def historic_div_scalar_int_reciprocal(self, other: int):
            if False:
                print('Hello World!')
            if self.is_floating_point():
                return other / self
            return torch.divide(other, self, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):

            def forward(self, a, b: float):
                if False:
                    for i in range(10):
                        print('nop')
                return b / a

        class MyModuleInt(torch.nn.Module):

            def forward(self, a, b: int):
                if False:
                    i = 10
                    return i + 15
                return b / a
        try:
            v3_mobile_module_float = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl')
            v3_mobile_module_int = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                if False:
                    print('Hello World!')
                m_result = self._try_fn(m, a, b)
                fn_result = None
                if fn is torch.div:
                    fn_result = self._try_fn(torch.div, b, a)
                else:
                    fn_result = self._try_fn(fn, a, b)
                if isinstance(m_result, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                elif fn is torch.div or a.is_floating_point():
                    self.assertEqual(m_result, fn_result)
                else:
                    pass
            if isinstance(b, float):
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, current_mobile_module_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(max_examples=10, deadline=200000)
    @given(sample_input=st.tuples(st.integers(min_value=5, max_value=199), st.floats(min_value=5.0, max_value=199.0)))
    @example((2, 3, 2.0, 3.0))
    def test_versioned_div_scalar_inplace(self, sample_input):
        if False:
            for i in range(10):
                print('nop')

        def historic_div_scalar_float_inplace(self, other: float):
            if False:
                i = 10
                return i + 15
            return self.true_divide_(other)

        def historic_div_scalar_int_inplace(self, other: int):
            if False:
                return 10
            if self.is_floating_point():
                return self.true_divide_(other)
            return self.divide_(other, rounding_mode='trunc')

        class MyModuleFloat(torch.nn.Module):

            def forward(self, a, b: float):
                if False:
                    return 10
                a /= b
                return a

        class MyModuleInt(torch.nn.Module):

            def forward(self, a, b: int):
                if False:
                    print('Hello World!')
                a /= b
                return a
        try:
            v3_mobile_module_float = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl')
            v3_mobile_module_int = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module_float = self._save_load_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_module(MyModuleInt)
        for (val_a, val_b) in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                if False:
                    for i in range(10):
                        print('nop')
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)
            if isinstance(b, float):
                _helper(current_mobile_module_float, torch.Tensor.div_)
            else:
                _helper(current_mobile_module_int, torch.Tensor.div_)

    def test_versioned_div_scalar_scalar(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, a: float, b: int, c: float, d: int):
                if False:
                    while True:
                        i = 10
                result_0 = a / b
                result_1 = a / c
                result_2 = b / c
                result_3 = b / d
                return (result_0, result_1, result_2, result_3)
        try:
            v3_mobile_module = _load_for_lite_interpreter(pytorch_test_dir + '/cpp/jit/upgrader_models/test_versioned_div_scalar_scalar_v2.ptl')
        except Exception as e:
            self.skipTest('Failed to load fixture!')
        current_mobile_module = self._save_load_mobile_module(MyModule)

        def _helper(m, fn):
            if False:
                while True:
                    i = 10
            vals = (5.0, 3, 2.0, 7)
            m_result = m(*vals)
            fn_result = fn(*vals)
            for (mr, hr) in zip(m_result, fn_result):
                self.assertEqual(mr, hr)
        _helper(v3_mobile_module, current_mobile_module)

    def test_versioned_linspace(self):
        if False:
            print('Hello World!')

        class Module(torch.nn.Module):

            def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
                if False:
                    for i in range(10):
                        print('nop')
                c = torch.linspace(a, b, steps=5)
                d = torch.linspace(a, b, steps=100)
                return (c, d)
        scripted_module = torch.jit.load(pytorch_test_dir + '/jit/fixtures/test_versioned_linspace_v7.ptl')
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for (a, b) in sample_inputs:
            (output_with_step, output_without_step) = v7_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_linspace_out(self):
        if False:
            i = 10
            return i + 15

        class Module(torch.nn.Module):

            def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
                if False:
                    return 10
                return torch.linspace(a, b, steps=100, out=out)
        model_path = pytorch_test_dir + '/jit/fixtures/test_versioned_linspace_out_v7.ptl'
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)
        sample_inputs = ((3, 10, torch.empty((100,), dtype=torch.int64), torch.empty((100,), dtype=torch.int64)), (-10, 10, torch.empty((100,), dtype=torch.int64), torch.empty((100,), dtype=torch.int64)), (4.0, 6.0, torch.empty((100,), dtype=torch.float64), torch.empty((100,), dtype=torch.float64)), (3 + 4j, 4 + 5j, torch.empty((100,), dtype=torch.complex64), torch.empty((100,), dtype=torch.complex64)))
        for (start, end, out_for_old, out_for_new) in sample_inputs:
            output = v7_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            self.assertTrue(output.size(dim=0) == 100)
            self.assertEqual(output, output_current)

    def test_versioned_logspace(self):
        if False:
            print('Hello World!')

        class Module(torch.nn.Module):

            def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
                if False:
                    for i in range(10):
                        print('nop')
                c = torch.logspace(a, b, steps=5)
                d = torch.logspace(a, b, steps=100)
                return (c, d)
        scripted_module = torch.jit.load(pytorch_test_dir + '/jit/fixtures/test_versioned_logspace_v8.ptl')
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for (a, b) in sample_inputs:
            (output_with_step, output_without_step) = v8_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_logspace_out(self):
        if False:
            for i in range(10):
                print('nop')

        class Module(torch.nn.Module):

            def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                return torch.logspace(a, b, steps=100, out=out)
        model_path = pytorch_test_dir + '/jit/fixtures/test_versioned_logspace_out_v8.ptl'
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)
        sample_inputs = ((3, 10, torch.empty((100,), dtype=torch.int64), torch.empty((100,), dtype=torch.int64)), (-10, 10, torch.empty((100,), dtype=torch.int64), torch.empty((100,), dtype=torch.int64)), (4.0, 6.0, torch.empty((100,), dtype=torch.float64), torch.empty((100,), dtype=torch.float64)), (3 + 4j, 4 + 5j, torch.empty((100,), dtype=torch.complex64), torch.empty((100,), dtype=torch.complex64)))
        for (start, end, out_for_old, out_for_new) in sample_inputs:
            output = v8_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            self.assertTrue(output.size(dim=0) == 100)
            self.assertEqual(output, output_current)