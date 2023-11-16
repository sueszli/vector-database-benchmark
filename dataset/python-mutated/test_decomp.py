from collections import defaultdict
from torch import Tensor
import torch.autograd
from torch._decomp import core_aten_decompositions, decomposition_table
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import is_iterable_of_tensors, TestCase, skipIfCrossRef, suppress_warnings, TEST_WITH_ASAN, TEST_WITH_SLOW, run_tests, skipIfTorchDynamo
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_device_type import onlyNativeDeviceTypes, ops, instantiate_device_type_tests, onlyCUDA
from torch.testing._internal.common_methods_invocations import op_db, skip, skipOps, xfail
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import DispatchKey
import itertools
import functools
from functools import partial
import unittest
aten = torch.ops.aten

def overload_to_aten_name(op):
    if False:
        for i in range(10):
            print('nop')
    return op._schema.name.split('::')[1]
decomposition_names = {overload_to_aten_name(k) for k in decomposition_table if isinstance(k, torch._ops.OpOverload)}
core_decomposition_names = {overload_to_aten_name(k) for k in core_aten_decompositions() if isinstance(k, torch._ops.OpOverload)}
_decomp_test_ops = [op for op in op_db if op.aten_name in decomposition_names or op.aten_backward_name in decomposition_names]
_decomp_test_ops_core_autograd = [op for op in op_db if op.aten_name in core_decomposition_names and op.supports_autograd]

def diff_arg(arg, requires_grad=True):
    if False:
        for i in range(10):
            print('nop')

    def is_differentiable_arg(arg):
        if False:
            for i in range(10):
                print('nop')
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all((is_differentiable_arg(a) for a in arg)):
            return True
        if all((not is_differentiable_arg(a) for a in arg)):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)

def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    if False:
        print('Hello World!')
    (inputs, inputs_spec) = tree_flatten(inputs)
    diff_inputs = tuple((inp for inp in inputs if inp.requires_grad))
    if grad_outputs is None:
        diff_outputs = tuple((out for out in outputs if out.requires_grad))
    else:
        diff_grad_outputs = [(out, go) for (out, go) in zip(outputs, grad_outputs) if out.requires_grad]
        if len(diff_grad_outputs) == 0:
            (diff_outputs, grad_outputs) = ((), ())
        else:
            (diff_outputs, grad_outputs) = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(diff_outputs, diff_inputs, grad_outputs, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)

def _as_tuple(val):
    if False:
        while True:
            i = 10
    if isinstance(val, tuple):
        return val
    return (val,)

def ref_vjp_no_create(f, *primals):
    if False:
        print('Hello World!')
    result = f(*primals)

    def wrapped(cotangents):
        if False:
            print('Hello World!')
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents), create_graph=False)
    return (result, wrapped)
dtype_precisions = {torch.float16: (0.001, 1e-05), torch.bfloat16: (0.016, 0.0001), torch.float32: (1.3e-06, 1e-05), torch.float64: (1e-07, 1e-07), torch.complex32: (0.001, 1e-05), torch.complex64: (1.3e-06, 1e-05), torch.complex128: (1e-07, 1e-07)}

def _getDefaultRtolAndAtol(dtype0, dtype1):
    if False:
        print('Hello World!')
    rtol = max(dtype_precisions.get(dtype0, (0, 0))[0], dtype_precisions.get(dtype1, (0, 0))[0])
    atol = max(dtype_precisions.get(dtype0, (0, 0))[1], dtype_precisions.get(dtype1, (0, 0))[1])
    return (rtol, atol)

def op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    if False:
        return 10
    assert orig.dtype == decomp.dtype, f'{i} Operation:  {op}'
    if orig.numel() == 0 or decomp.numel() == 0:
        assert orig.numel() == decomp.numel()
        return
    assert orig.shape == decomp.shape, f'{i} Operation:  {op}'
    tol_table = {(torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 0.001, (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 0.02, (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-05, (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.default): 1e-05, (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-05, (torch.float16, torch.ops.aten._native_batch_norm_legit.default): 1e-05, (torch.float16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-05, (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 0.0001, (torch.float16, torch.ops.aten.linalg_vector_norm.default): 0.0001, (torch.bfloat16, torch.ops.aten.var_mean.correction): 5e-07, (torch.float16, torch.ops.aten.var_mean.correction): 5e-07, (torch.bfloat16, torch.ops.aten.var_mean.dim): 5e-07, (torch.float16, torch.ops.aten.var_mean.dim): 5e-07, (torch.float16, torch.ops.aten.nll_loss_forward.default): 0.01, (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 0.1, (torch.float16, torch.ops.aten.nll_loss2d_forward.default): 0.01, (torch.bfloat16, torch.ops.aten.nll_loss2d_forward.default): 0.2, (torch.float16, torch.ops.aten.mv.default): 1e-05}
    if ref.is_floating_point():
        orig_diff = (orig - ref).abs().max()
        decomp_diff = (decomp - ref).abs().max()
        atol = tol_table.get((test_dtype, op), 1e-07)
        if decomp_diff > orig_diff + atol:
            raise RuntimeError(f'Difference from float64 is larger with decomposition {op.__name__} than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\natol = {atol}\nargs = {args}\nkwargs = {kwargs}')
    else:
        test_case.assertEqual(orig, decomp, msg=f'{op.__name__}\nargs = {args}\nkwargs = {kwargs}')

def op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    if False:
        print('Hello World!')
    test_case.assertEqual(orig.dtype, decomp.dtype, f'Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}')
    tol_table = {(torch.float32, torch.ops.aten.native_layer_norm.default): (0.001, 0.001), (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (0.001, 0.001), (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-06, 1e-06), (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-06, 3e-05), (torch.float32, torch.ops.aten.mv.default): (1e-05, 3e-05), (torch.complex64, torch.ops.aten.mv.default): (5e-05, 5e-05), (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-05, 0.0005), (torch.float64, torch.ops.aten.upsample_bicubic2d.default): (1e-05, 0.0005), (torch.int8, torch.ops.aten.linspace.default): (0, 1), (torch.uint8, torch.ops.aten.linspace.default): (0, 1), (torch.int16, torch.ops.aten.linspace.default): (0, 1), (torch.int32, torch.ops.aten.linspace.default): (0, 1), (torch.int64, torch.ops.aten.linspace.default): (0, 1), (torch.int8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1), (torch.uint8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1), (torch.int16, torch.ops.aten.linspace.Tensor_Tensor): (0, 1), (torch.int32, torch.ops.aten.linspace.Tensor_Tensor): (0, 1), (torch.int64, torch.ops.aten.linspace.Tensor_Tensor): (0, 1), (torch.int8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1), (torch.uint8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1), (torch.int16, torch.ops.aten.linspace.Tensor_Scalar): (0, 1), (torch.int32, torch.ops.aten.linspace.Tensor_Scalar): (0, 1), (torch.int64, torch.ops.aten.linspace.Tensor_Scalar): (0, 1), (torch.int8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1), (torch.uint8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1), (torch.int16, torch.ops.aten.linspace.Scalar_Tensor): (0, 1), (torch.int32, torch.ops.aten.linspace.Scalar_Tensor): (0, 1), (torch.int64, torch.ops.aten.linspace.Scalar_Tensor): (0, 1)}
    if (decomp.dtype, op) in tol_table:
        (rtol, atol) = tol_table[decomp.dtype, op]
    else:
        (rtol, atol) = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    test_case.assertEqual(orig, decomp, rtol=rtol, atol=atol, msg=f'{op.__name__}\nargs = {args}\nkwargs = {kwargs}')

def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    if False:
        while True:
            i = 10
    (flat_args, args_spec) = tree_flatten(args)
    diff_argnums = tuple((i for (i, arg) in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad)))
    assert len(diff_argnums) > 0
    primals = tuple((flat_args[i] for i in diff_argnums))

    @functools.wraps(f)
    def wrapped(*primals):
        if False:
            for i in range(10):
                print('nop')
        _args = list(flat_args)
        for (num, arg) in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple((r for r in result if isinstance(r, Tensor) and (r.is_floating_point() or r.is_complex())))
            assert len(result) > 0
        return result
    return (wrapped, primals)

def upcast_tensor(x, dtype=torch.float32):
    if False:
        while True:
            i = 10
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        return x.to(dtype=dtype)
    elif isinstance(x, torch.dtype) and x in [torch.float16, torch.bfloat16, torch.float]:
        return dtype
    else:
        return x

def normalize_op_input_output(f, sample, requires_grad=True):
    if False:
        print('Hello World!')
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad)
CROSS_REF_EXCLUDE_SET = {('cuda', torch.bfloat16, 'nn.functional.bilinear'), (None, None, 'special.ndtr'), (None, None, 'new_empty'), (None, None, 'empty_like'), (None, None, 'empty'), (None, None, 'item'), (None, None, 'zero_'), (None, torch.float32, 'masked.logsumexp'), (None, torch.float64, 'masked.logsumexp'), (torch.cpu, torch.float16, 'signal.windows.exponential'), (torch.cpu, torch.float16, 'signal.windows.gaussian'), (torch.cpu, torch.float16, 'signal.windows.cosine'), (None, None, 'nn.functional.relu6'), (None, None, 'nn.functional.rrelu'), (None, None, 'meshgrid'), (None, None, 'nn.functional.hardshrink'), (None, None, 'nn.functional.softshrink'), (None, None, 'diag'), ('cpu', torch.bfloat16, '_softmax_backward_data'), (None, None, 'norm'), (None, None, 'native_batch_norm'), (None, None, '_upsample_bilinear2d_aa'), (None, None, 'empty_strided')}
CROSS_REF_BACKWARD_EXCLUDE_SET = {('cpu', torch.bfloat16, 'nn.functional.hardswish'), ('cuda', torch.float16, 'nn.functional.cross_entropy')}
all_decomposed = set()
all_called = defaultdict(int)
'\nimport atexit\ndef check_coverage():\n    print("missing coverage:")\n    print("\n".join(map(str, decomposition_table.keys() - all_decomposed)))\natexit.register(check_coverage)\n'
"\nimport atexit\ndef dump_ops():\n    with open('run_ops.txt', 'w') as f, open('count_ops.txt', 'w') as g:\n        for op, count in sorted(all_called.items(), key=lambda x: x[0].__name__):\n            f.write(f'{op.__name__}\n')\n            g.write(f'{count}\n')\n    with open('run_decompositions.txt', 'w') as f:\n        for op in sorted([i.__name__ for i in all_decomposed]):\n            f.write(f'{op}\n')\n\natexit.register(dump_ops)\n"

def any_unsupported(args, kwargs):
    if False:
        return 10

    def test_unsupported(t):
        if False:
            for i in range(10):
                print('nop')
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            return any([t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized, t.is_nested, torch._is_functional_tensor(t)])
        elif torch.overrides.is_tensor_like(t):
            return True
        else:
            return False
    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    return any((test_unsupported(x) for x in flat_args))
core_backward_failures = {skip('_softmax_backward_data'), xfail('addcdiv'), skip('addcmul'), skip('deg2rad'), skip('diag_embed'), skip('frac'), skip('grid_sampler_2d'), xfail('lerp'), skip('logaddexp'), skip('native_dropout_backward'), xfail('nn.functional.binary_cross_entropy_with_logits'), skip('nn.functional.glu'), xfail('nn.functional.hardshrink'), xfail('nn.functional.softshrink'), skip('nn.functional.unfold'), xfail('norm'), xfail('norm', 'fro'), xfail('norm', 'inf'), xfail('norm', 'nuc'), skip('rad2deg'), skip('renorm'), skip('rot90'), skip('rsub'), skip('sgn'), skip('special.xlog1py'), xfail('stack'), skip('tril'), skip('triu'), skip('unfold_copy'), skip('xlogy'), xfail('zero_')}
if not TEST_WITH_SLOW:
    core_backward_failures.update({skip('addr'), skip('baddbmm'), skip('clamp_min'), skip('clamp_max'), skip('logit'), skip('nn.functional.hardswish'), skip('std_mean'), skip('split', variant_name='list_args'), skip('transpose'), skip('unbind'), skip('unsafe_split')})

class TestDecomp(TestCase):
    longMessage = True

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    def test_quick(self, device, dtype, op):
        if False:
            return 10
        self.do_cross_ref(device, dtype, op, run_all=False)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @skipOps('TestDecomp', 'test_quick_core_backward', core_backward_failures)
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops_core_autograd, allowed_dtypes=(torch.float64,))
    def test_quick_core_backward(self, device, dtype, op):
        if False:
            print('Hello World!')
        for sample_input in op.sample_inputs(device, dtype, requires_grad=True):
            aten_name = op.decomp_aten_name or op.aten_name
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            func = partial(op.get_op(), **kwargs)
            with self.DecompCrossRefMode(self, self.precision, self.rel_tol, dtype, run_all=False) as mode, enable_python_dispatcher():
                torch.autograd.gradcheck(func, args)
            self.check_decomposed(aten_name, mode)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        if False:
            while True:
                i = 10
        self.do_cross_ref(device, dtype, op, run_all=True)

    def test_uniform(self, device):
        if False:
            print('Hello World!')
        size = (2, 3, 4, 5)
        dtype = torch.float32
        x = make_tensor(size, dtype=dtype, device=device)
        low = 0.3
        high = 0.9
        torch.manual_seed(123)
        ref = torch.ops.aten.uniform(x, low, high)
        torch.manual_seed(123)
        res = torch._decomp.decompositions.uniform(x, low=low, high=high)
        self.assertEqual(ref, res)

    def test_rrelu_with_noise(self, device):
        if False:
            i = 10
            return i + 15
        dtype = torch.float64
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
        lower = 1.0
        upper = 4.0
        training = False
        torch.manual_seed(123)
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)
        torch.manual_seed(123)
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)
        res = torch._decomp.decompositions.rrelu_with_noise(x, noise_res, lower, upper, training)
        self.assertEqual(ref, res)
        self.assertEqual(noise_ref, noise_res)
        training = True
        torch.manual_seed(123)
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)
        torch.manual_seed(123)
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)
        res = torch._decomp.decompositions.rrelu_with_noise(x, noise_res, lower, upper, training)
        self.assertEqual(ref, res)
        self.assertEqual(noise_ref, noise_res)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @suppress_warnings
    @tf32_off()
    @modules(filter(lambda m: m.module_cls in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU), module_db))
    def test_rnn_decomp_module(self, device, dtype, module_info, training):
        if False:
            while True:
                i = 10
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype, requires_grad=True, training=training)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue
            (args, kwargs) = (module_input.constructor_input.args, module_input.constructor_input.kwargs)
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            (args, kwargs) = (module_input.forward_input.args, module_input.forward_input.kwargs)
            with self.DecompCrossRefMode(self, self.precision, self.rel_tol, dtype, run_all=True), enable_python_dispatcher():
                decomp_out = m(*args, **kwargs)
            non_decomp_out = m(*args, **kwargs)
            self.assertEqual(decomp_out, non_decomp_out)

    def test_batch_norm_unflatten_weight_bias(self, device):
        if False:
            i = 10
            return i + 15
        shape = (1, 3, 2, 2)
        input = torch.randn(shape, device=device)
        weight = torch.randn((3, 1, 1, 1), device=device)
        bias = torch.randn(3, device=device)
        mean = torch.randn(3, device=device)
        var = torch.randn(3, device=device)
        res = torch._decomp.decompositions.native_batch_norm(input, weight, bias, mean, var, False, 1, 1e-05)
        self.assertEqual(shape, res[0].shape)

    class DecompCrossRefMode(TorchDispatchMode):

        def __init__(self, test_case, saved_precision, saved_rel_tol, dtype, run_all):
            if False:
                return 10
            self.test_case = test_case
            self.saved_precision = saved_precision
            self.saved_rel_tol = saved_rel_tol
            self.test_dtype = dtype
            self.run_all = run_all
            self.called = set()
            self.decomposed = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if False:
                for i in range(10):
                    print('nop')
            self.test_case.precision = self.saved_precision
            self.test_case.rel_tol = self.saved_rel_tol
            self.called.add(func)
            all_called[func] += 1
            in_place = func.name()[-1] == '_'
            ignored_ops = [torch.ops.aten.detach.default, torch.ops.aten.empty.memory_format, torch.ops.aten.empty_like.default, torch.ops.aten.new_empty.default, torch.ops.aten.empty_strided.default, torch.ops.aten.new_empty_strided.default, torch.ops.aten.randn.default, torch.ops.aten.native_dropout.default]
            if func not in decomposition_table or func in ignored_ops or torch.Tag.nondeterministic_seeded in func.tags or any_unsupported(args, kwargs) or in_place:
                return func(*args, **kwargs)
            self.decomposed.add(func)
            all_decomposed.add(func)
            decomposition = decomposition_table[func]
            do_relative_check = self.test_dtype in [torch.float16, torch.bfloat16]
            if self.run_all:
                with self:
                    decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))
            else:
                decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))
            real_out_unflat = func(*args, **kwargs)
            real_out = pytree.tree_leaves(real_out_unflat)
            assert len(real_out) == len(decomp_out)
            if do_relative_check:
                upcast = partial(upcast_tensor, dtype=torch.float64)
                (real_out_double, _) = tree_flatten(func(*tree_map(upcast, args), **tree_map(upcast, kwargs)))
                for (i, (orig, decomp, ref)) in enumerate(zip(real_out, decomp_out, real_out_double)):
                    if not isinstance(orig, torch.Tensor):
                        assert type(orig) == type(decomp)
                        assert orig == decomp
                        continue
                    op_assert_ref(self.test_case, func, self.test_dtype, i, orig, decomp, ref, args, kwargs)
            else:
                for (orig, decomp) in zip(real_out, decomp_out):
                    if not isinstance(orig, torch.Tensor):
                        assert type(orig) == type(decomp)
                        assert orig == decomp
                        continue
                    op_assert_equal(self.test_case, func, self.test_dtype, orig, decomp, args, kwargs)
            return real_out_unflat

    def check_decomposed(self, aten_name, mode):
        if False:
            i = 10
            return i + 15
        self.assertTrue(any((overload_to_aten_name(c) == aten_name for c in mode.decomposed)), msg=f"aten.{aten_name} was not decomposed, saw calls for: {', '.join(map(str, list(mode.called)))}. If your op is  CompositeImplicitAutograd you should skip this test by updating CROSS_REF_EXCLUDE_SET.")

    @skipIfTorchDynamo('Test does not work with TorchDynamo')
    def do_cross_ref(self, device, dtype, op, *, run_all):
        if False:
            i = 10
            return i + 15
        test_keys = [(torch.device(device).type, dtype, op.name), (None, dtype, op.name), (None, None, op.name)]
        if any((key in CROSS_REF_EXCLUDE_SET for key in test_keys)):
            self.skipTest(f'{op.name} in {dtype} not supported')
        skip_decomp_vjp = any((key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys))
        requires_grad = op.supports_autograd and dtype in op.supported_backward_dtypes(torch.device(device).type) and (not dtype == torch.complex32)
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)
        aten_name = op.decomp_aten_name or op.aten_name
        func = op.get_op()
        for sample_input in samples:
            if requires_grad:
                (fn, primals) = normalize_op_input_output(func, sample_input)
                primals = tree_map(lambda x: x if isinstance(x, torch.Tensor) else x, primals)
                with self.DecompCrossRefMode(self, self.precision, self.rel_tol, dtype, run_all) as mode, enable_python_dispatcher():
                    (decomp_out, decomp_vjp_fn) = ref_vjp_no_create(fn, *primals)
                if aten_name in decomposition_names:
                    self.check_decomposed(aten_name, mode)
                if not skip_decomp_vjp and (op.aten_backward_name in decomposition_names or run_all):
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)
                    with self.DecompCrossRefMode(self, self.precision, self.rel_tol, dtype, run_all) as mode, enable_python_dispatcher():
                        decomp_vjp_fn(cotangents)
                    if not run_all:
                        self.check_decomposed(op.aten_backward_name, mode)
            elif aten_name in decomposition_names or run_all:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                with self.DecompCrossRefMode(self, self.precision, self.rel_tol, dtype, run_all) as mode, enable_python_dispatcher():
                    func(*args, **kwargs)
                if not run_all:
                    self.check_decomposed(aten_name, mode)
            else:
                assert op.supports_autograd
                self.skipTest("only backwards is decomposed, but dtype doesn't support AD")
instantiate_device_type_tests(TestDecomp, globals())

class DecompOneOffTests(TestCase):

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_softmax(self, device):
        if False:
            print('Hello World!')
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)
        ref = torch.ops.aten._softmax(x, -1, False)
        res = torch._decomp.decompositions._softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_log_softmax(self, device):
        if False:
            return 10
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)
        ref = torch.ops.aten._log_softmax(x, -1, False)
        res = torch._decomp.decompositions._log_softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @skipIfCrossRef
    @onlyCUDA
    def test_amp_batch_norm_backward(self):
        if False:
            print('Hello World!')
        device = 'cuda'
        grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        weight = torch.randn((2,), dtype=torch.float32, device=device)
        rmean = torch.randn((2,), dtype=torch.float32, device=device)
        rvar = torch.randn((2,), dtype=torch.float32, device=device)
        mean = torch.randn((0,), dtype=torch.float32, device=device)
        ref = torch.ops.aten.native_batch_norm_backward(grad_out, x, weight, rmean, rvar, mean, mean, False, 1e-05, [True, True, True])
        res = torch._decomp.decompositions.native_batch_norm_backward(grad_out, x, weight, rmean, rvar, mean, mean, False, 1e-05, [True, True, True])
        for (a, b) in zip(ref, res):
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.dtype, b.dtype)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_elu_backward(self, device):
        if False:
            i = 10
            return i + 15
        size = (2, 4, 3, 3)
        dtype = torch.float32
        grad_out = torch.randn(size, dtype=dtype, device=device)
        out = torch.randn(size, dtype=dtype, device=device)
        ref = torch.ops.aten.elu_backward(grad_out, 1.0, 1, 1, True, out)
        res = torch._decomp.decompositions.elu_backward(grad_out, 1.0, 1, 1, True, out)
        self.assertEqual(ref, res)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_threshold_backward_dtype(self, device):
        if False:
            i = 10
            return i + 15
        grad = torch.randint(10, (4,), device=device)
        input_tensor = torch.randint(10, (4,), device=device)
        ref = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        res = torch._decomp.decompositions.threshold_backward(grad, input_tensor, 1)
        self.assertEqual(ref.dtype, res.dtype)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_weight_norm_interface(self, device):
        if False:
            i = 10
            return i + 15
        g = torch.randn((3, 10, 10), device=device)
        v = torch.randn((1, 1, 10), device=device)
        ref = torch.ops.aten._weight_norm_interface(g, v, 2)
        res = torch._decomp.decompositions._weight_norm_interface(g, v, 2)
        self.assertTrue(torch.allclose(ref[0], res[0]))
        self.assertTrue(torch.allclose(ref[1], res[1]))

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_sdpa(self, device):
        if False:
            print('Hello World!')
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._decomp import get_decompositions
        from torch.nn import functional as F

        class ScaledDotProductAttention(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()

            def forward(self, query_layer, key_layer, value_layer):
                if False:
                    print('Hello World!')
                attn_output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, None, dropout_p=0.0, is_causal=True)
                return attn_output
        query_layer = torch.randn(1, 128, 100, 64, device=device)
        key_layer = torch.randn(1, 128, 100, 64, device=device)
        value_layer = torch.randn(1, 128, 100, 64, device=device)
        attention = ScaledDotProductAttention()
        fx_g = make_fx(attention, decomposition_table=get_decompositions([torch.ops.aten._scaled_dot_product_flash_attention.default]))(query_layer, key_layer, value_layer)
        compiled_res = fx_g(query_layer, key_layer, value_layer)
        eager_res = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, None, dropout_p=0.0, is_causal=True)
        self.assertTrue(torch.allclose(compiled_res, eager_res, atol=1e-06, rtol=1e-05))
instantiate_device_type_tests(DecompOneOffTests, globals())

class HasDecompTest(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.maxDiff = None

    @staticmethod
    def _can_appear_in_trace(op: torch._ops.OpOverload) -> bool:
        if False:
            for i in range(10):
                print('nop')
        has_tensor_arg = any(('Tensor' in str(a.type) for a in itertools.chain(op._schema.arguments, op._schema.returns)))
        if not has_tensor_arg:
            return False
        try:
            return not op.has_kernel_for_dispatch_key(DispatchKey.CompositeImplicitAutograd)
        except RuntimeError as e:
            if 'does not exist' in str(e):
                return False
            raise

    def test_has_decomposition(self):
        if False:
            i = 10
            return i + 15

        def all_aten_overloads():
            if False:
                return 10
            for name in torch._C._dispatch_get_all_op_names():
                if not name.startswith('aten::'):
                    continue
                name = name[6:]
                if '.' in name:
                    (packet_name, overload_name) = name.split('.')
                else:
                    (packet_name, overload_name) = (name, 'default')
                packet = getattr(aten, packet_name)
                assert isinstance(packet, torch._ops.OpOverloadPacket)
                op = getattr(packet, overload_name)
                yield op
        allow_list = {aten.get_gradients.default}
        overloads_wanting_decomp = {op for op in all_aten_overloads() if self._can_appear_in_trace(op)}
        ops_missing_decomp = overloads_wanting_decomp - decomposition_table.keys()
        ops_missing_decomp -= allow_list
        self.assertExpected(''.join(sorted((op.name() + '\n' for op in ops_missing_decomp))))

    def test_aten_core_operators(self):
        if False:
            for i in range(10):
                print('nop')
        useful_decomps = {op for op in decomposition_table.keys() if isinstance(op, torch._ops.OpOverload) and self._can_appear_in_trace(op)}
        core_decomps = torch._decomp.core_aten_decompositions().keys()
        core_aten_ops = useful_decomps - core_decomps
        self.assertExpected(''.join(sorted((op.name() + '\n' for op in core_aten_ops))))
if __name__ == '__main__':
    run_tests()