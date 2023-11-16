from contextlib import nullcontext
from numbers import Number
import random
import re
import torch
import unittest
import itertools
import weakref
from torch.testing import make_tensor
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, skipIfTorchDynamo, parametrize, gradcheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, onlyCUDA, ops, OpDTypes
from torch.testing._internal.common_methods_invocations import foreach_unary_op_db, foreach_binary_op_db, foreach_pointwise_op_db, foreach_reduce_op_db, foreach_other_op_db
from torch.testing._internal.common_dtype import all_types_and_complex_and, floating_types_and, floating_types, integral_types_and
_BOOL_SUB_ERR_MSG = 'Subtraction, the `-` operator'

class RegularFuncWrapper:

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        self.func = func

    def __call__(self, inputs, values=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if values is not None:
            assert len(inputs) == 3
            if isinstance(values, Number):
                values = [values for _ in range(len(inputs[0]))]
            return [self.func(*i, value=values[idx], **kwargs) for (idx, i) in enumerate(zip(*inputs))]
        if len(inputs) == 2 and isinstance(inputs[1], (Number, torch.Tensor)):
            inputs[1] = [inputs[1] for _ in range(len(inputs[0]))]
        return [self.func(*i, **kwargs) for i in zip(*inputs)]

class ForeachFuncWrapper:

    def __init__(self, func):
        if False:
            print('Hello World!')
        self.func = func
        self.is_inplace = False if func is None else func.__name__.endswith('_')

    def __call__(self, inputs, is_cuda, expect_fastpath, **kwargs):
        if False:
            while True:
                i = 10
        actual = None
        zero_size = kwargs.pop('zero_size', False)
        if is_cuda and torch.autograd.kineto_available() and (torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()):
            with torch.profiler.profile() as p:
                actual = self.func(*inputs, **kwargs)
            keys = tuple([e.key for e in p.key_averages()])
            mta_called = any(('multi_tensor_apply_kernel' in k for k in keys))
            assert mta_called == (expect_fastpath and (not zero_size))
        else:
            actual = self.func(*inputs, **kwargs)
        return inputs[0] if self.is_inplace else actual

class InplaceForeachVersionBumpCheck:

    def __init__(self, testcase: TestCase, tensorlist: 'List[torch.Tensor]') -> None:
        if False:
            i = 10
            return i + 15
        self._testcase = testcase
        self._tensorlist = tensorlist
        self._orig_version_counts = [t._version for t in tensorlist]

    def __enter__(self):
        if False:
            return 10
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self._testcase.assertGreaterEqual([t._version for t in self._tensorlist], self._orig_version_counts)

def get_transform_func(num_tensors, dtype, device, is_fastpath):
    if False:
        while True:
            i = 10

    def transform(t):
        if False:
            for i in range(10):
                print('nop')
        if not torch.is_tensor(t):
            return t
        if torch.is_tensor(t) and t.ndim == 0:
            return t
        return make_tensor((num_tensors, num_tensors), dtype=dtype, device=device, requires_grad=True, noncontiguous=not is_fastpath)
    return transform

class TestForeach(TestCase):

    @property
    def is_cuda(self):
        if False:
            i = 10
            return i + 15
        return self.device_type == 'cuda'

    def _get_funcs(self, op):
        if False:
            i = 10
            return i + 15
        return (ForeachFuncWrapper(op.method_variant), RegularFuncWrapper(op.ref), ForeachFuncWrapper(op.inplace_variant), RegularFuncWrapper(op.ref_inplace))

    @onlyCUDA
    @ops(foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_reduce_op_db + foreach_other_op_db, dtypes=(torch.float32,))
    def test_all_zero_size_tensors_do_not_launch_kernel(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        (wrapped_op, _, inplace_op, _) = self._get_funcs(op)
        for sample in op.sample_zero_size_inputs(device, dtype):
            if not op.has_no_out_of_place:
                wrapped_op((sample.input, *sample.args), is_cuda=self.is_cuda, expect_fastpath=True, zero_size=True)
            with InplaceForeachVersionBumpCheck(self, sample.input):
                inplace_op((sample.input, *sample.args), is_cuda=self.is_cuda, expect_fastpath=True, zero_size=True)

    @unittest.skipIf(TEST_WITH_ROCM, 'Skipped on ROCm, since it is failing on ROCm 5.7')
    @ops(foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_reduce_op_db + foreach_other_op_db)
    @parametrize('noncontiguous,inplace', [(False, False), (False, True), (True, False), (True, True)], name_fn=lambda x, y: '{}_{}'.format('fastpath' if not x else 'slowpath', 'inplace' if y else 'outplace'))
    def test_parity(self, device, dtype, op, noncontiguous, inplace):
        if False:
            for i in range(10):
                print('nop')
        if inplace:
            (_, _, func, ref) = self._get_funcs(op)
        else:
            (func, ref, _, _) = self._get_funcs(op)
        for sample in op.sample_inputs(device, dtype, noncontiguous=noncontiguous):
            ref_kwargs = sample.kwargs
            kwargs = ref_kwargs.copy()
            div_slowpath = dtype in integral_types_and(torch.bool) and op.name == '_foreach_div'
            expect_fastpath = not (noncontiguous or sample.disable_fastpath or div_slowpath)
            if op in foreach_pointwise_op_db:
                values = kwargs.pop('values', None)
                if values is not None:
                    sample.args = (*sample.args, values)
            (ref_input, ctxmgr) = (sample.input, nullcontext())
            if inplace:
                with torch.no_grad():
                    ref_input = [t.clone().detach() for t in sample.input]
                ctxmgr = InplaceForeachVersionBumpCheck(self, sample.input)
            try:
                with ctxmgr:
                    actual = func([sample.input, *sample.args], self.is_cuda, expect_fastpath, **kwargs)
            except Exception as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))) if not (op.has_no_in_place or op.has_no_out_of_place) else self.assertRaises(type(e)):
                    ref([ref_input, *sample.ref_args], **ref_kwargs)
            else:
                expected = ref([ref_input, *sample.ref_args], **ref_kwargs)
                self.assertEqual(expected, actual)

    def _binary_test(self, dtype, op, ref, inputs, is_fastpath, is_inplace, *, alpha, scalar_self_arg: bool):
        if False:
            i = 10
            return i + 15
        ref_inputs = [[t.clone().detach() for t in inputs[0]], inputs[1]] if is_inplace else inputs
        try:
            with InplaceForeachVersionBumpCheck(self, inputs[0]) if op.is_inplace else nullcontext():
                actual = op(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                if not scalar_self_arg:
                    ref(ref_inputs)
                else:
                    [ref.func(ref_inputs[0], t) for t in ref_inputs[1]]
        else:
            expected = ref(ref_inputs) if not scalar_self_arg else [ref.func(ref_inputs[0], t) for t in ref_inputs[1]]
            self.assertEqual(actual, expected)
        if alpha is not None and (not scalar_self_arg):
            kwargs = {'alpha': alpha}
            ref_inputs = inputs
            try:
                op_kwargs = {}
                op_kwargs.update(kwargs)
                with InplaceForeachVersionBumpCheck(self, inputs[0]) if op.is_inplace else nullcontext():
                    actual = op(inputs, self.is_cuda, is_fastpath, **op_kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref(ref_inputs, **kwargs)
            else:
                expected = ref(ref_inputs, **kwargs)
                if dtype in (torch.float16, torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(expected, actual, atol=0.001, rtol=default_tolerances(dtype)[0])
                else:
                    self.assertEqual(expected, actual)

    @ops(filter(lambda op: op.supports_scalar_self_arg, foreach_binary_op_db))
    @parametrize('is_fastpath', (True, False))
    def test_binary_op_with_scalar_self_support(self, device, dtype, op, is_fastpath):
        if False:
            i = 10
            return i + 15

        def clone(arg):
            if False:
                while True:
                    i = 10
            if isinstance(arg, (list, tuple)):
                return [clone(a) for a in arg]
            if torch.is_tensor(arg):
                return arg.clone().detach().requires_grad_()
            else:
                return arg
        scalar_self_arg_test_complete = False
        for (i, sample) in enumerate(op.sample_inputs(device, dtype, noncontiguous=not is_fastpath)):
            (rhs_arg,) = sample.args
            kwargs = {} or sample.kwargs
            alpha = kwargs.pop('alpha', None)
            (wrapped_op, ref, inplace_op, inplace_ref) = self._get_funcs(op)
            if isinstance(rhs_arg, Number) and (not scalar_self_arg_test_complete):
                scalar_self_arg_test_complete = True
                self._binary_test(dtype, wrapped_op, ref, [rhs_arg, sample.input], is_fastpath, False, alpha=alpha, scalar_self_arg=True)
                if op.supports_autograd and dtype == torch.float32:
                    transformed_sample = sample.transform(get_transform_func(len(sample.input), dtype, device, is_fastpath))
                    tensors = transformed_sample.input
                    (rhs_arg,) = transformed_sample.args
                    (ref_tensors, ref_rhs_arg) = (clone(tensors), clone(rhs_arg))
                    sum(wrapped_op([rhs_arg, tensors], is_cuda=False, expect_fastpath=False)).mean().backward()
                    sum([ref.func(ref_rhs_arg, t) for t in ref_tensors]).mean().backward()
                    self.assertEqual([t.grad for t in tensors], [t.grad for t in ref_tensors])

    @ops(foreach_pointwise_op_db)
    @parametrize('is_fastpath', (True, False))
    def test_pointwise_op_with_tensor_of_scalarlist_overload(self, device, dtype, op, is_fastpath):
        if False:
            print('Hello World!')
        for sample in op.sample_inputs(device, dtype, noncontiguous=not is_fastpath):
            assert isinstance(sample.args, tuple)
            assert len(sample.args) == 2
            inputs = [sample.input, *sample.args]
            kwargs = sample.kwargs
            disable_fastpath = sample.disable_fastpath and is_fastpath
            (wrapped_op, ref, inplace_op, inplace_ref) = self._get_funcs(op)
            values = kwargs.pop('values', None)
            if is_fastpath and isinstance(values, list):
                sample = sample.transform(lambda t: t.clone().detach() if torch.is_tensor(t) else t)
                inputs = [sample.input, *sample.args]
                tensor_values = torch.tensor(values)
                for (is_inplace, op_, ref_) in ((False, wrapped_op, ref), (True, inplace_op, inplace_ref)):
                    self._pointwise_test(op_, ref_, inputs, is_fastpath and (not disable_fastpath), is_inplace, values=tensor_values)
                    self._pointwise_test(op_, ref_, inputs, is_fastpath and (not disable_fastpath), is_inplace, values=tensor_values[0], custom_values_err='Expected packed scalar Tensor to be of dimension 1. Got 0 instead.')
                    if self.is_cuda:
                        self._pointwise_test(op_, ref_, inputs, is_fastpath and (not disable_fastpath), is_inplace, values=tensor_values.cuda(), custom_values_err='Expected scalars to be on CPU, got cuda:0 instead.')
                    self._pointwise_test(op_, ref_, inputs, is_fastpath and (not disable_fastpath), is_inplace, values=tensor_values[:2], custom_values_err=f'Expected length of scalars to match input of length {len(values)} but got 2 instead.')
                    self._pointwise_test(op_, ref_, inputs, is_fastpath and (not disable_fastpath), is_inplace, values=torch.tensor([[0, 1], [2, 3]])[:, 1], custom_values_err='Expected scalars to be contiguous.')
            N = len(sample.input)
            inputs = [[make_tensor((N, N), device=device, dtype=dtype, noncontiguous=not is_fastpath) for _ in range(N)], [make_tensor((N - i, 1), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)], [make_tensor((1, N - i), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)]]
            self._pointwise_test(wrapped_op, ref, inputs, is_fastpath and disable_fastpath, is_inplace=False, values=values)
            self._pointwise_test(inplace_op, inplace_ref, inputs, is_fastpath and disable_fastpath, is_inplace=True, values=values)

    def _pointwise_test(self, op, ref, inputs, is_fastpath, is_inplace, *, values=None, custom_values_err=None):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {}
        ref_inputs = [[t.clone().detach() for t in inputs[0]], inputs[1], inputs[2]] if is_inplace else inputs
        try:
            with InplaceForeachVersionBumpCheck(self, inputs[0]) if is_inplace else nullcontext():
                actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(ref_inputs)
        else:
            expected = ref(ref_inputs)
            self.assertEqual(expected, actual)
        if values is not None:
            try:
                actual = op(inputs + [values], self.is_cuda, is_fastpath, **kwargs)
            except RuntimeError as e:
                if custom_values_err is None:
                    with self.assertRaisesRegex(type(e), re.escape(str(e))):
                        ref(ref_inputs, values=values)
                else:
                    self.assertEqual(re.escape(str(e)), re.escape(custom_values_err))
            else:
                expected = ref(ref_inputs, values=values)
                self.assertEqual(expected, actual)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        if False:
            return 10
        for tensors in [[torch.randn([0], device=device, dtype=dtype)]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)
            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db), dtypes=OpDTypes.supported)
    def test_binary_op_scalar_with_overlapping_tensors(self, device, dtype, op):
        if False:
            while True:
                i = 10
        (foreach_op, ref) = (op.method_variant, op.ref)
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        if ref == torch.sub and dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                [ref(t, 1) for t in tensors]
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                foreach_op(tensors, 1)
            return
        expected = [ref(t, 1) for t in tensors]
        res = foreach_op(tensors, 1)
        self.assertEqual(res, expected)

    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db), allowed_dtypes=[torch.float])
    def test_binary_op_scalar_with_different_tensor_dtypes(self, device, dtype, op):
        if False:
            while True:
                i = 10
        foreach_op = op.method_variant
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device), torch.tensor([1], dtype=torch.long, device=device)]
        runtime_error = None
        try:
            foreach_op(tensors, 1)
        except RuntimeError as e:
            runtime_error = e
        self.assertIsNone(runtime_error)

    @skipIfTorchDynamo('Different error msgs, TODO')
    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db), dtypes=OpDTypes.supported)
    def test_binary_op_list_error_cases(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        (foreach_op, foreach_op_, ref, ref_) = (op.method_variant, op.inplace_variant, op.ref, op.ref_inplace)
        tensors1 = []
        tensors2 = []
        ops_to_test = [foreach_op, foreach_op_]
        for fop in ops_to_test:
            with self.assertRaisesRegex(RuntimeError, 'There were no tensor arguments to this function'):
                fop(tensors1, tensors2)
        tensors1.append(torch.tensor([1], device=device, dtype=dtype))
        for fop in ops_to_test:
            with self.assertRaisesRegex(RuntimeError, 'Tensor list must have same number of elements as scalar list.'):
                fop(tensors1, tensors2)
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        for fop in ops_to_test:
            with self.assertRaisesRegex(RuntimeError, 'Tensor lists must have the same number of tensors, got 1 and 2'):
                fop(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, 'Tensor lists must have the same number of tensors, got 2 and 1'):
                fop(tensors2, tensors1)
        tensors1 = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, device=device, dtype=dtype) for _ in range(10)]
        try:
            foreach_op(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [ref(t1, t2) for (t1, t2) in zip(tensors1, tensors2)]
        try:
            foreach_op_(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [ref_(t1, t2) for (t1, t2) in zip(tensors1, tensors2)]
        if self.device_type == 'cuda' and torch.cuda.device_count() > 1:
            tensor1 = torch.zeros(10, 10, device='cuda:0', dtype=dtype)
            tensor2 = torch.ones(10, 10, device='cuda:1', dtype=dtype)
            if dtype == torch.bool and foreach_op == torch._foreach_sub:
                for fop in ops_to_test:
                    with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                        fop([tensor1], [tensor2])
                return
            with self.assertRaisesRegex(RuntimeError, 'Expected all tensors to be on the same device'):
                foreach_op([tensor1], [tensor2])
            if dtype in integral_types_and(torch.bool) and foreach_op == torch._foreach_div:
                with self.assertRaisesRegex(RuntimeError, 'result type'):
                    foreach_op_([tensor1], [tensor2])
            else:
                with self.assertRaisesRegex(RuntimeError, 'Expected all tensors to be on the same device'):
                    foreach_op_([tensor1], [tensor2])

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not found')
    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db), dtypes=OpDTypes.supported)
    def test_binary_op_list_slow_path(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        (foreach_op, native_op, foreach_op_, native_op_) = self._get_funcs(op)
        tensor1 = make_tensor((10, 10), dtype=dtype, device=device)
        tensor2 = make_tensor((1,), device=device, dtype=dtype).expand_as(tensor1)
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False, alpha=None, scalar_self_arg=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True, alpha=None, scalar_self_arg=False)
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        inputs = ([tensor1], [tensor2.t()])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False, alpha=None, scalar_self_arg=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True, alpha=None, scalar_self_arg=False)
        tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
        tensor2 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False, alpha=None, scalar_self_arg=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True, alpha=None, scalar_self_arg=False)
        tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype)
        tensor2 = make_tensor((5, 2, 1, 3 * 7), device=device, dtype=dtype)[:, :, :, ::7]
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False, alpha=None, scalar_self_arg=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True, alpha=None, scalar_self_arg=False)

    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db), dtypes=floating_types_and(torch.half, torch.bfloat16))
    def test_binary_op_float_inf_nan(self, device, dtype, op):
        if False:
            while True:
                i = 10
        inputs = ([torch.tensor([float('inf')], device=device, dtype=dtype), torch.tensor([-float('inf')], device=device, dtype=dtype), torch.tensor([float('nan')], device=device, dtype=dtype), torch.tensor([float('nan')], device=device, dtype=dtype)], [torch.tensor([-float('inf')], device=device, dtype=dtype), torch.tensor([float('inf')], device=device, dtype=dtype), torch.tensor([float('inf')], device=device, dtype=dtype), torch.tensor([float('nan')], device=device, dtype=dtype)])
        (op, ref, inplace_op, inplace_ref) = self._get_funcs(op)
        self._binary_test(dtype, op, ref, inputs, True, False, alpha=None, scalar_self_arg=False)
        self._binary_test(dtype, inplace_op, inplace_ref, inputs, True, True, alpha=None, scalar_self_arg=False)

    @onlyCUDA
    @ops(foreach_unary_op_db)
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        if False:
            while True:
                i = 10
        op.has_no_out_of_place = op.name != '_foreach_zero'
        (method, ref, inplace_method, ref_inplace) = self._get_funcs(op)
        tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[2]))[0].input
        tensors[1] = tensors[1].to('cpu')
        if op.has_no_out_of_place:
            try:
                actual = method((tensors,), False, False, zero_size=False)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), str(e)):
                    ref((tensors,))
            else:
                expected = ref((tensors,))
                self.assertEqual(expected, actual)
        try:
            inplace_method((tensors,), False, False, zero_size=False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                ref_inplace((tensors,))
        else:
            if op.has_no_out_of_place:
                self.assertEqual(expected, tensors)
            else:
                self.assertEqual([torch.zeros_like(t) for t in tensors], tensors)

    @onlyCUDA
    @ops(filter(lambda op: not op.has_no_out_of_place, foreach_binary_op_db))
    def test_binary_op_tensors_on_different_devices(self, device, dtype, op):
        if False:
            while True:
                i = 10
        _cuda_tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[2], same_size=True))[0].input
        _cpu_tensors = list(op.sample_inputs('cpu', dtype, num_input_tensors=[2], same_size=True))[0].input
        (tensors1, tensors2) = list(zip(_cuda_tensors, _cpu_tensors))
        (foreach_op, foreach_op_) = (op.method_variant, op.inplace_variant)
        (native_op, native_op_) = (op.ref, op.ref_inplace)
        try:
            actual = foreach_op(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op(t1, t2) for (t1, t2) in zip(tensors1, tensors2)]
        else:
            expected = [native_op(t1, t2) for (t1, t2) in zip(tensors1, tensors2)]
            self.assertEqual(expected, actual)
        try:
            foreach_op_(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op_(t1, t2) for (t1, t2) in zip(tensors1, tensors2)]
        else:
            self.assertEqual(actual, tensors1)

    @onlyCUDA
    @ops(foreach_pointwise_op_db, allowed_dtypes=floating_types())
    def test_pointwise_op_tensors_on_different_devices(self, device, dtype, op):
        if False:
            print('Hello World!')
        _cuda_tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[3], same_size=True))[int(dtype == torch.float32)].input
        _cpu_tensors = list(op.sample_inputs('cpu', dtype, num_input_tensors=[3], same_size=True))[0].input
        (tensors1, tensors2, tensors3) = list(zip(_cuda_tensors, _cpu_tensors))
        (foreach_op, foreach_op_, native_op) = (op.method_variant, op.inplace_variant, op.ref)
        actual = foreach_op(tensors1, tensors2, tensors3)
        expected = [native_op(*_cuda_tensors), native_op(*_cpu_tensors)]
        self.assertEqual(expected, actual)
        foreach_op_(tensors1, tensors2, tensors3)
        self.assertEqual(expected, tensors1)

    @onlyCUDA
    @ops(foreach_reduce_op_db, allowed_dtypes=(torch.half, torch.bfloat16))
    def test_foreach_l2_large_value_input(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        (ord, N) = (2, 10)
        max_value = torch.finfo(dtype).max
        scaler = torch.tensor([max_value]).sqrt().to(device=device, dtype=dtype)
        inputs = ([t * scaler for t in list(op.sample_inputs(device, dtype, requries_grad=True, num_input_tensors=[N], low=1))[0].input],)
        self.assertTrue(scaler * scaler * N > max_value)
        (fn, ref_fn, *_) = self._get_funcs(op)
        actual = fn(inputs, is_cuda=True, expect_fastpath=True, ord=ord, zero_size=False)
        expect = ref_fn(inputs, ord=ord)
        if dtype == torch.float16:
            self.assertFalse(any((torch.isinf(e) for e in expect)))
        else:
            self.assertTrue(all((inputs[0][i].numel() == 0 or torch.isinf(e) for (i, e) in enumerate(expect))))
        self.assertEqual(expect, actual, equal_nan=False)

    @onlyCUDA
    @ops(foreach_reduce_op_db)
    def test_foreach_reduce_large_input(self, device, dtype, op):
        if False:
            while True:
                i = 10
        (ord, N) = (2, 65536 * 2)
        disable_fastpath = True
        if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
            disable_fastpath = False
        inputs = ([make_tensor((N,), dtype=dtype, device=device, noncontiguous=False)],)
        (wrapped_op, ref, _, _) = self._get_funcs(op)
        self.assertEqual(ref(inputs, ord=ord), wrapped_op(inputs, self.is_cuda, not disable_fastpath, ord=ord, zero_size=False))

    @onlyCUDA
    @ops(foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_other_op_db, dtypes=(torch.float,))
    def test_inplace_foreach_leaf_check_and_grad_fn(self, device, dtype, op):
        if False:
            return 10
        inplace_op = op.inplace_variant
        if inplace_op is None:
            self.skipTest('no in-place op available')
        sample = list(op.sample_inputs(dtype=dtype, device=device, num_input_tensors=[2], same_size=True))[0]
        sample.input[0].requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'a leaf Variable that requires grad'):
            inplace_op(sample.input, *sample.args)
        sample.input[1].requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'a leaf Variable that requires grad'):
            inplace_op(sample.input, *sample.args)
        _tensors = [t.clone().detach().requires_grad_(i == 0) for (i, t) in enumerate(sample.input)]
        tensors = [t.clone() for t in _tensors]
        inplace_op(tensors, *sample.args)
        self.assertIsNotNone(tensors[0].grad_fn)
        self.assertIsNone(tensors[1].grad_fn)

    @onlyCUDA
    @ops(foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_other_op_db, dtypes=(torch.float,))
    def test_outplace_with_invalid_grads(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        if op.has_no_out_of_place:
            self.skipTest(f'{op.name} does not have out-of-place implementation')
        (func, *_) = self._get_funcs(op)
        sample = list(op.sample_inputs(dtype=dtype, device=device, requires_grad=True, num_input_tensors=[2], same_size=True))[0]
        self.assertTrue(all((t.requires_grad for t in sample.input)))
        if func.func in foreach_pointwise_op_db:
            sample.kwargs.pop('values', None)
        (out1, out2) = func([sample.input, *sample.args], is_cuda=False, expect_fastpath=False, **sample.kwargs)
        out1.backward(torch.ones_like(out1))
        self.assertIsNotNone(sample.input[0].grad)
        self.assertIsNone(sample.input[1].grad)

    @ops(filter(lambda op: op.backward_requires_result, foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_other_op_db), dtypes=(torch.float32,))
    def test_lifetime_of_grad_fn_when_result_is_saved(self, device, dtype, op):
        if False:
            i = 10
            return i + 15

        def get_ref(func, sample):
            if False:
                print('Hello World!')

            class Foo:
                pass
            out = func((sample.input, *sample.args), is_cuda=False, expect_fastpath=False, **sample.kwargs)
            foo = Foo()
            meta_dict = out[0].grad_fn.metadata
            meta_dict[0] = foo
            ref = weakref.ref(foo)
            return (out, ref)

        def _test(func, sample):
            if False:
                for i in range(10):
                    print('nop')
            (out, ref) = get_ref(func, sample)
            self.assertIsNotNone(ref())
            del out
            self.assertIsNone(ref())
        func = self._get_funcs(op)[0]
        for sample in op.sample_inputs(device, dtype, requires_grad=True, num_input_tensors=[1]):
            for key in ('is_fastpath', 'disable_fastpath'):
                if key in sample.kwargs:
                    del sample.kwargs[key]
            if op.name == '_foreach_pow':
                if isinstance(sample.args[0], list) and isinstance(sample.args[0][0], Number) or (isinstance(sample.args[0], Number) and (not isinstance(sample.args[0], float))):
                    continue
                if isinstance(sample.args[0], float):
                    new_args = (sample.input,)
                    sample.input = sample.args[0]
                    sample.args = new_args
            _test(func, sample)

    @unittest.skipIf(not (torch.cuda.is_available() and torch.cuda.device_count() > 1), 'requires multiple GPUs')
    def test_tensors_grouping(self):
        if False:
            print('Hello World!')
        num_tensors_per_list = 10
        num_devices = torch.cuda.device_count()
        dtypes = (torch.float16, torch.float32, torch.float64)
        list1 = [torch.tensor(i, device=torch.device('cuda', random.randint(0, num_devices - 1)), dtype=dtypes[random.randint(0, 2)]) for i in range(num_tensors_per_list)]
        list2 = [None for _ in list1]
        list3 = [torch.rand_like(t) for t in list1]
        nested_tensorlists = [list1, list2, list3]
        grouped_tensors = torch.utils._foreach_utils._group_tensors_by_device_and_dtype(nested_tensorlists, with_indices=True)
        num_tensors_seen = 0
        for ((device, dtype), ([l1, l2, l3], indices)) in grouped_tensors.items():
            for t in itertools.chain(l1, l3):
                self.assertEqual(t.device, device)
                self.assertEqual(t.dtype, dtype)
                num_tensors_seen += 1
            self.assertEqual(len(l1), len(l2))
            self.assertTrue(all((p is None for p in l2)))
            for (i, index) in enumerate(indices):
                self.assertEqual(l1[i], list1[index])
                self.assertEqual(l2[i], list2[index])
                self.assertEqual(l3[i], list3[index])
        self.assertEqual(num_tensors_seen, 2 * num_tensors_per_list)

    @onlyCUDA
    def test_0dim_tensor_overload_cpu_ok(self):
        if False:
            for i in range(10):
                print('nop')
        tensors = [torch.ones((), device='cuda', dtype=torch.float32) for _ in range(2)]
        scalar_cpu_tensor = torch.tensor(4.0, device='cpu')
        actual = torch._foreach_mul(tensors, scalar_cpu_tensor)
        self.assertEqual(actual, [t.mul(scalar_cpu_tensor) for t in tensors])
        actual = torch._foreach_div(tensors, scalar_cpu_tensor)
        self.assertEqual(actual, [t.div(scalar_cpu_tensor) for t in tensors])

    @onlyCUDA
    def test_0dim_tensor_overload_exception(self):
        if False:
            for i in range(10):
                print('nop')
        tensors = [make_tensor((2, 2), dtype=torch.float, device='cuda') for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, 'scalar tensor expected to be on'):
            torch._foreach_add(tensors, torch.tensor(1.0, device='cpu'), alpha=1.0)
        tensors = [make_tensor((2, 2), dtype=torch.float, device=d) for d in ('cpu', 'cuda')]
        with self.assertRaisesRegex(RuntimeError, 'scalar tensor expected to be 0 dim but'):
            torch._foreach_mul(tensors, torch.tensor([1.0, 1.0], device='cuda'))
        with self.assertRaisesRegex(RuntimeError, 'scalar tensor expected to be 0 dim but'):
            torch._foreach_add(tensors, torch.tensor([1.0, 1.0], device='cuda'))

    @onlyCUDA
    @ops(filter(lambda op: op.name == '_foreach_copy', foreach_binary_op_db))
    def test_foreach_copy_with_multi_device_inputs(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        foreach_copy_ = op.inplace_variant
        copy_ = op.ref_inplace
        for non_blocking in (False, True):
            for sample in op.sample_inputs(device, dtype, noncontiguous=False):
                with torch.no_grad():
                    ref_input = [t.clone().detach() for t in sample.input]
                foreach_copy_(sample.input, sample.args[0], non_blocking)
                for (t, s) in zip(ref_input, sample.args[0]):
                    copy_(t, s, non_blocking)
                self.assertEqual(sample.input, ref_input)
                if torch.cuda.device_count() > 1:
                    device = torch.device('cuda', 1)
                    rhs_tensors = [t.to(device) for t in sample.args[0]]
                    foreach_copy_(sample.input, rhs_tensors, non_blocking)
                    for (t, s) in zip(ref_input, rhs_tensors):
                        copy_(t, s, non_blocking)
                    self.assertEqual(ref_input, sample.input)

    @onlyCUDA
    @ops(foreach_unary_op_db + foreach_binary_op_db + foreach_pointwise_op_db + foreach_reduce_op_db + foreach_other_op_db, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float64, torch.complex128))
    @parametrize('inplace', (False, True), name_fn=lambda x: 'inplace' if x else 'outplace')
    def test_autodiff(self, device, dtype, op, inplace):
        if False:
            while True:
                i = 10
        if not (op.supports_autograd or op.supports_forward_ad):
            self.skipTest('neither reverse mode nor forward mode supported')
        if not inplace and op.has_no_out_of_place:
            self.skipTest('out-of-place not implemented')
        if inplace and op.has_no_in_place:
            self.skipTest('in-place not implemented')
        if not inplace and dtype == torch.float64 and (op.name in ('_foreach_acos', '_foreach_asin', '_foreach_log10', '_foreach_log1p', '_foreach_log2', '_foreach_log', '_foreach_pow', '_foreach_sqrt')):
            value_range = {'low': 0.5, 'high': 1.0}
        else:
            value_range = {}
        for sample in op.sample_inputs(device, dtype, requires_grad=True, num_input_tensors=[5], **value_range):
            if op.name == '_foreach_pow' and isinstance(sample.input, Number):
                continue
            func = None
            if inplace:

                def inplace_func(*tensorlist):
                    if False:
                        for i in range(10):
                            print('nop')
                    kwargs = {'alpha': sample.kwargs['alpha']} if 'alpha' in sample.kwargs else {}
                    op.inplace_variant(tuple((t.clone() for t in tensorlist)), *sample.args, **kwargs)
                    return tensorlist
                func = inplace_func
            else:

                def outplace_func(*tensorlist):
                    if False:
                        for i in range(10):
                            print('nop')
                    kwargs = {'alpha': sample.kwargs['alpha']} if 'alpha' in sample.kwargs else {}
                    return op.method_variant(tensorlist, *sample.args, **kwargs)
                func = outplace_func
            (working_sample, err_msg_pattern) = check_autodiff_sample(op, sample, dtype, inplace)

            def call_gradcheck():
                if False:
                    print('Hello World!')
                gradcheck(func, sample.input, raise_exception=True, check_forward_ad=op.supports_forward_ad, check_batched_forward_grad=False, check_backward_ad=op.supports_autograd, check_batched_grad=False)
            if not working_sample:
                if not err_msg_pattern:
                    continue
                with self.assertRaisesRegex(RuntimeError, re.escape(err_msg_pattern)):
                    call_gradcheck()
                continue
            call_gradcheck()
            if inplace and op.supports_inplace_autograd:
                hook_buffer = []

                def get_grad_fn_hook(i):
                    if False:
                        print('Hello World!')

                    def hook(grad_inputs, grad_outputs) -> None:
                        if False:
                            print('Hello World!')
                        hook_buffer.append(i)
                    return hook
                _inputs = [t.clone().detach().requires_grad_() for t in sample.input]
                inputs = [t.clone() for t in _inputs]
                kwargs = {'alpha': sample.kwargs['alpha']} if 'alpha' in sample.kwargs else {}
                op.inplace_variant(inputs, *sample.args, **kwargs)
                self.assertEqual(len({t.grad_fn for t in inputs}), len(inputs))
                for (i, t) in enumerate(inputs):
                    t.grad_fn.register_hook(get_grad_fn_hook(i))
                torch.autograd.grad(inputs[0], inputs=(_inputs[0],), grad_outputs=(torch.rand_like(inputs[0]),), retain_graph=True)
                self.assertEqual(hook_buffer, [0])
                hook_buffer.clear()
                sum_of_cloned_tensors = torch.cat([t.view(-1) for t in inputs]).sum()
                grad_output = torch.rand_like(sum_of_cloned_tensors)
                torch.autograd.grad(sum_of_cloned_tensors, inputs=tuple(_inputs), grad_outputs=(grad_output,), retain_graph=False)
                self.assertEqual(hook_buffer, list(reversed(range(len(inputs)))))

def check_autodiff_sample(op, sample, dtype, is_inplace):
    if False:
        while True:
            i = 10
    if op.name == '_foreach_abs' and is_inplace and (dtype == torch.complex128):
        return (False, 'In-place abs is not supported for complex tensors.')
    if op.name == '_foreach_sub' and (isinstance(sample.args[0], list) and any((isinstance(a, bool) for a in sample.args[0])) or isinstance(sample.args[0], bool)):
        return (False, _BOOL_SUB_ERR_MSG)
    if op.name == '_foreach_norm' and (not is_inplace):
        return (False, 'Trying to set a forward gradient that has a different size than that of the original Tensor, this is not supported. Tensor is of size [] while the given forward gradient is of size [1, 1].')
    rhs_arg_has_complex_number = sample.args and (isinstance(sample.args[0], list) and any((isinstance(a, complex) for a in sample.args[0])) or isinstance(sample.args[0], complex))
    if rhs_arg_has_complex_number and dtype == torch.float64:
        if op.name in ('_foreach_clamp_max', '_foreach_clamp_min', '_foreach_maximum', '_foreach_minimum'):
            return (False, 'clamp is not supported for complex types')
        if not is_inplace:
            return (False, '')
        else:
            if op.name == '_foreach_pow':
                return (False, 'Found dtype Double but expected ComplexDouble')
            if op.name in ('_foreach_add', '_foreach_sub', '_foreach_mul', '_foreach_div'):
                return (False, "result type ComplexDouble can't be cast to the desired output type Double")
    return (True, '')
instantiate_device_type_tests(TestForeach, globals())
if __name__ == '__main__':
    run_tests()