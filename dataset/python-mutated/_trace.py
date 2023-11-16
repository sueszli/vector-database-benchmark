"""Tracing.

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import _qualified_name, get_callable_argument_names, is_scripting
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
R = TypeVar('R', covariant=True)
P = ParamSpec('P')

def _create_interpreter_name_lookup_fn(frames_up=1):
    if False:
        return 10

    def _get_interpreter_name_for_var(var):
        if False:
            for i in range(10):
                print('nop')
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError('failed to inspect frame')
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError('failed to get frame')
            i += 1
        f_locals = frame.f_locals
        f_globals = frame.f_globals
        for (k, v) in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        return ''
    return _get_interpreter_name_for_var

def _unique_state_dict(module, keep_vars=False):
    if False:
        for i in range(10):
            print('nop')
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids: Set[int] = set()
    for (k, v) in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.detach()
    return filtered_dict

class ONNXTracedModule(torch.nn.Module):

    def __init__(self, inner, strict=True, force_outplace=False, return_inputs=False, return_inputs_states=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.inner = inner
        self.strict = strict
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states

    def forward(self, *args: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        (in_vars, in_desc) = _flatten(args)
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        ret_inputs = []
        inputs_states = []
        outs = []

        def wrapper(*args):
            if False:
                for i in range(10):
                    print('nop')
            in_args: List[torch.Tensor] = []
            for i in range(len(in_vars)):
                if not isinstance(args[i], torch.Tensor):
                    raise RuntimeError('Expected Tensor argument')
                in_args.append(args[i])
            trace_inputs = _unflatten(in_args, in_desc)
            if self._return_inputs:
                ret_inputs.append(tuple((x.clone(memory_format=torch.preserve_format) for x in args)))
            if self._return_inputs_states:
                inputs_states.append(_unflatten(in_args, in_desc))
            outs.append(self.inner(*trace_inputs))
            if self._return_inputs_states:
                inputs_states[0] = (inputs_states[0], trace_inputs)
            (out_vars, _) = _flatten(outs)
            if len(out_vars) == 1:
                return out_vars[0]
            else:
                return tuple(out_vars)
        (graph, out) = torch._C._create_graph_by_tracing(wrapper, in_vars + module_state, _create_interpreter_name_lookup_fn(), self.strict, self._force_outplace)
        if self._return_inputs:
            return (graph, outs[0], ret_inputs[0])
        if self._return_inputs_states:
            return (graph, outs[0], inputs_states[0])
        else:
            return (graph, outs[0])

def _clone_inputs(args):
    if False:
        return 10

    def clone_input(a):
        if False:
            print('Hello World!')
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            v = a.detach().clone(memory_format=None if a.is_mkldnn else torch.preserve_format).requires_grad_(a.requires_grad)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone(memory_format=torch.preserve_format)
    return function._nested_map(lambda x: isinstance(x, torch.Tensor), clone_input, condition_msg='tensors')(args)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)
_JIT_STATS = os.environ.get('PYTORCH_JIT_STATS', False)

@contextlib.contextmanager
def _time(trace_name, name, time=True):
    if False:
        i = 10
        return i + 15
    if not _JIT_TIME and (not time) or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        yield
    finally:
        stream.record_event(end)
        end.synchronize()
        print(f'{trace_name} {name} time: {start.elapsed_time(end)} ms')

def verify(model, args, loss_fn=torch.sum, devices=None):
    if False:
        while True:
            i = 10
    "\n    Verify that a JIT compiled model has the same behavior as its uncompiled version along with its backwards pass.\n\n    If your model returns multiple outputs,\n    you must also specify a `loss_fn` to produce a loss for which\n    the backwards will be computed.\n\n    This function has side-effects (e.g., it executes your model / saves and loads\n    parameters), so don't expect the model to come out exactly the same as what\n    you passed in.\n\n    Args:\n        model (compiled torch.nn.Module or function): the module/function to be\n            verified.  The module/function definition MUST have been decorated with\n            `@torch.jit.compile`.\n        args (tuple or Tensor): the positional arguments to pass to the\n            compiled function/module to be verified.  A non-tuple is assumed to\n            be a single positional argument to be passed to the model.\n        loss_fn (function, optional): the loss function to be applied to\n            the output of the model, before backwards is invoked.  By default,\n            we assume that a model returns a single result, and we :func:`torch.sum`\n            before calling backwards; if this is inappropriate, you can pass your\n            own loss function.  Note that if a model returns a tuple of results,\n            these are passed as separate positional arguments to `loss_fn`.\n        devices (iterable of device IDs, optional): the GPU devices which the\n            compiled module will be run on.  This determines the RNG state we\n            must save when running both compiled and uncompiled versions of the model.\n    "
    if not isinstance(model, torch._C.CompiledFunction):
        raise TypeError('Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it')
    is_module = isinstance(model, Module)
    if not isinstance(args, tuple):
        args = (args,)
    saved_args = _clone_inputs(args)
    if is_module:
        saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        if False:
            i = 10
            return i + 15
        params = list(model.parameters()) if is_module else []
        (in_vars, _) = _flatten((args, params))
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        if assert_compiled:
            hits = compiled_fn.hits
        out = model(*args)
        if assert_compiled and compiled_fn.hits == hits:
            raise RuntimeError('failed to use the compiled function')
        if not isinstance(out, tuple):
            out = (out,)
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(f'Model returns {len(out)} outputs, but default loss function (torch.sum) can only handle a single output')
        (out_vars, _) = _flatten(out)
        saved_outs = [v.detach().clone(memory_format=torch.preserve_format) for v in out_vars]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        saved_grads = [v.detach().clone(memory_format=torch.preserve_format) for v in grads]
        return (saved_outs, saved_grads)
    with torch.random.fork_rng(devices, _caller='torch.jit.verify'):
        (uncompiled_outs, uncompiled_grads) = run_fwd_bwd(args, force_trace=True)
        assert model.has_trace_for(*args)
    if is_module:
        model.load_state_dict(saved_state)
    (compiled_outs, compiled_grads) = run_fwd_bwd(args, assert_compiled=True)
    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)

def _verify_equal(xs, ys):
    if False:
        while True:
            i = 10
    for (x, y) in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-06:
            raise RuntimeError('JIT and real computation mismatch')

def indent(s):
    if False:
        while True:
            i = 10
    return '\n'.join(['\t' + line for line in s.splitlines()])

class TracingCheckError(Exception):

    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        if False:
            return 10
        self.message = 'Tracing failed sanity checks!\n'
        if extra_msg is not None:
            self.message += extra_msg + '\n'
        if graph_diff_error is not None:
            self.message += 'ERROR: Graphs differed across invocations!\n'
            self.message += indent(graph_diff_error) + '\n'
        if tensor_compare_error is not None:
            self.message += 'ERROR: Tensor-valued Constant nodes differed in value across invocations. This often indicates that the tracer has encountered untraceable code.\n'
            self.message += indent(tensor_compare_error) + '\n'
        super().__init__(self.message)

@torch.no_grad()
def _check_trace(check_inputs, func, traced_func, check_tolerance, strict, force_outplace, is_trace_module, _module_class, example_inputs_is_kwarg=False):
    if False:
        print('Hello World!')
    for inputs in check_inputs:
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        if is_trace_module:
            copied_dict = {}
            for (name, data) in inputs.items():
                copied_dict[name] = _clone_inputs(data)
            check_mod = torch.jit.trace_module(getattr(func, '__self__', func), copied_dict, check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, _compilation_unit=torch._C.CompilationUnit(), example_inputs_is_kwarg=example_inputs_is_kwarg, _store_inputs=False)
            check_mod_func = check_mod._c._get_method(traced_func.name)
            inputs = inputs[traced_func.name]
            if isinstance(inputs, torch.Tensor) or (isinstance(inputs, dict) and (not example_inputs_is_kwarg)):
                inputs = (inputs,)
        else:
            if example_inputs_is_kwarg:
                check_mod = torch.jit.trace(func, check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, example_kwarg_inputs=_clone_inputs(inputs), _store_inputs=False)
            else:
                check_mod = torch.jit.trace(func, _clone_inputs(inputs), check_trace=False, strict=strict, _force_outplace=force_outplace, _module_class=_module_class, _store_inputs=False)
            check_mod_func = check_mod

        def graph_diagnostic_info():
            if False:
                while True:
                    i = 10
            mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
            torch._C._jit_pass_inline(mod_canonicalized)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            mod_str = str(mod_canonicalized)
            mod_str = re.sub('___torch_mangle_[0-9]+\\.', '', mod_str)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
            torch._C._jit_pass_inline(check_canonicalized)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)
            check_str = str(check_canonicalized)
            check_str = re.sub('___torch_mangle_[0-9]+\\.', '', check_str)
            graph_diff_errors = None
            if mod_str != check_str:
                import difflib
                graph_diff = difflib.ndiff(mod_str.splitlines(True), check_str.splitlines(True))
                graph_diff_errors = 'Graph diff:\n' + indent(''.join(graph_diff)) + '\n'
                for (n_mod, n_check) in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += 'First diverging operator:\n'
                        node_diff = difflib.ndiff(str(n_mod).splitlines(True), str(n_check).splitlines(True))
                        source_printout = 'Node diff:\n' + indent(''.join(node_diff)) + '\n'
                        mod_stack = n_mod.sourceRange()
                        if mod_stack:
                            source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                        check_stack = n_check.sourceRange()
                        if check_stack:
                            source_printout += 'Check source location:\n' + indent(check_stack) + '\n'
                        graph_diff_errors += source_printout
                        break
            tensor_compare_errors = None
            for (n_mod, n_check) in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                if n_mod.kind() != n_check.kind():
                    break
                if n_mod.kind() == 'prim::Constant' and (not (n_mod.mustBeNone() or n_check.mustBeNone())):
                    if not n_mod.hasAttribute('value'):
                        continue
                    if n_mod.kindOf('value') != 't' or n_check.kindOf('value') != 't':
                        continue
                    mod_tensor_val = n_mod.t('value')
                    check_tensor_val = n_check.t('value')
                    try:
                        torch.testing.assert_close(mod_tensor_val, check_tensor_val, equal_nan=True)
                    except (RuntimeError, AssertionError) as e:
                        if tensor_compare_errors is None:
                            tensor_compare_errors = ''
                        tensor_compare_errors += 'Node:\n' + indent(str(n_mod)) + '\n'
                        compare_stack = n_mod.sourceRange()
                        if compare_stack:
                            tensor_compare_errors += 'Source Location:\n' + indent(compare_stack) + '\n'
                        tensor_compare_errors += 'Comparison exception: ' + indent(str(e))
                        break
            return (graph_diff_errors, tensor_compare_errors)

        def wrap_retval(x):
            if False:
                i = 10
                return i + 15
            return x if isinstance(x, tuple) else (x,)

        def run_mod_and_filter_tensor_outputs(mod, inputs, running_what):
            if False:
                return 10
            try:
                if isinstance(inputs, dict) and example_inputs_is_kwarg:
                    outs = wrap_retval(mod(**inputs))
                else:
                    outs = wrap_retval(mod(*_clone_inputs(inputs)))
                outs = [out for out in outs if isinstance(out, torch.Tensor)]
                return outs
            except Exception as e:
                (graph_diff_errors, tensor_compare_errors) = graph_diagnostic_info()
                msg = f'encountered an exception while running the {running_what} with test inputs.\nException:\n{indent(str(e))}'
                raise TracingCheckError(graph_diff_errors, tensor_compare_errors, extra_msg=msg) from e
        has_warned = [False]

        def maybe_warn_nondeterministic():
            if False:
                return 10
            if has_warned[0]:
                return
            has_warned[0] = True
            nondeterm_ops = [op for op in traced_func.graph.nodes() if op.isNondeterministic()]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = 'Trace had nondeterministic nodes. '
                nondeterministic_ops_warning += 'Did you forget call .eval() on your model? Nodes:\n'
                nondeterministic_ops_warning += '\n'.join([indent(str(op)) for op in nondeterm_ops][:20])
                nondeterministic_ops_warning += '\nThis may cause errors in trace checking. To disable trace checking, pass check_trace=False to torch.jit.trace()'
                warnings.warn(nondeterministic_ops_warning, category=TracerWarning, stacklevel=5)

        def compare_outputs(original, reference, match_what):
            if False:
                while True:
                    i = 10
            all_ok = True
            for (i, (orig, ref)) in enumerate(zip(original, reference)):
                try:
                    if orig.is_quantized:
                        orig = orig.dequantize()
                    if ref.is_quantized:
                        ref = ref.dequantize()
                    if orig.is_mkldnn:
                        orig = orig.to_dense()
                    if ref.is_mkldnn:
                        ref = ref.to_dense()
                    if ref.is_complex() or orig.is_complex():
                        torch.testing.assert_close(orig.to(torch.cdouble), ref.to(torch.cdouble), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                    elif orig.is_mps or ref.is_mps:
                        torch.testing.assert_close(orig.float(), ref.float(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                    else:
                        torch.testing.assert_close(orig.double(), ref.double(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
                except AssertionError as e:
                    maybe_warn_nondeterministic()
                    warnings.warn('Output nr ' + str(i + 1) + '. of the traced function does not match the corresponding output of the ' + match_what + '. Detailed error:\n' + str(e), category=TracerWarning, stacklevel=4)
                    all_ok = False
            return all_ok
        traced_outs = run_mod_and_filter_tensor_outputs(traced_func, inputs, 'trace')
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, 'Python function')
        if compare_outputs(traced_outs, fn_outs, 'Python function'):
            check_outs = run_mod_and_filter_tensor_outputs(check_mod_func, inputs, 'repeated trace')
            compare_outputs(traced_outs, check_outs, 'repeated trace')
        diag_info = graph_diagnostic_info()
        if any((info is not None for info in diag_info)):
            raise TracingCheckError(*diag_info)

class TracerWarning(Warning):

    @staticmethod
    def ignore_lib_warnings():
        if False:
            return 10
        warnings.filterwarnings('ignore', category=TracerWarning, module='torch.(?!jit)')
        warnings.filterwarnings('ignore', 'torch::jit::fuser::cuda')
TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()

def make_tuple(example_inputs):
    if False:
        return 10
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs

def make_module(mod, _module_class, _compilation_unit):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(mod, ScriptModule):
        return mod
    elif torch._jit_internal.module_has_exports(mod):
        infer_methods_stubs_fn = torch.jit._recursive.make_stubs_from_exported_methods
        return torch.jit._recursive.create_script_module(mod, infer_methods_stubs_fn, share_types=False, is_tracing=True)
    else:
        if _module_class is None:
            _module_class = TopLevelTracedModule
        return _module_class(mod, _compilation_unit=_compilation_unit)

def wrap_check_inputs(check_inputs):
    if False:
        i = 10
        return i + 15
    if check_inputs is None:
        return None
    return [{'forward': c} for c in check_inputs]

def trace(func, example_inputs=None, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu, example_kwarg_inputs=None, _store_inputs=True):
    if False:
        print('Hello World!')
    "\n    Trace a function and return an executable  or :class:`ScriptFunction` that will be optimized using just-in-time compilation.\n\n    Tracing is ideal for code that operates only on\n    ``Tensor``\\\\s and lists, dictionaries, and\n    tuples of ``Tensor``\\\\s.\n\n    Using `torch.jit.trace` and `torch.jit.trace_module`, you can turn an\n    existing module or Python function into a TorchScript\n    :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example\n    inputs, and we run the function, recording the operations performed on all\n    the tensors.\n\n    * The resulting recording of a standalone function produces `ScriptFunction`.\n    * The resulting recording of `nn.Module.forward` or `nn.Module` produces\n      `ScriptModule`.\n\n    This module also contains any parameters that the original\n    module had as well.\n\n    Warning:\n        Tracing only correctly records functions and modules which are not data\n        dependent (e.g., do not have conditionals on data in tensors) and do not have\n        any untracked external dependencies (e.g., perform input/output or\n        access global variables). Tracing only records operations done when the given\n        function is run on the given tensors. Therefore, the returned\n        `ScriptModule` will always run the same traced graph on any input. This\n        has some important implications when your module is expected to run\n        different sets of operations, depending on the input and/or the module\n        state. For example,\n\n        * Tracing will not record any control-flow like if-statements or loops.\n          When this control-flow is constant across your module, this is fine\n          and it often inlines the control-flow decisions. But sometimes the\n          control-flow is actually part of the model itself. For instance, a\n          recurrent network is a loop over the (possibly dynamic) length of an\n          input sequence.\n        * In the returned :class:`ScriptModule`, operations that have different\n          behaviors in ``training`` and ``eval`` modes will always behave as if\n          it is in the mode it was in during tracing, no matter which mode the\n          `ScriptModule` is in.\n\n        In cases like these, tracing would not be appropriate and\n        :func:`scripting <torch.jit.script>` is a better choice. If you trace\n        such models, you may silently get incorrect results on subsequent\n        invocations of the model. The tracer will try to emit warnings when\n        doing something that may cause an incorrect trace to be produced.\n\n    Args:\n        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`\n            that will be run with `example_inputs`. `func` arguments and return\n            values  must be tensors or (possibly nested) tuples that contain\n            tensors. When a module is passed `torch.jit.trace`, only the\n            ``forward`` method is run and traced (see :func:`torch.jit.trace\n            <torch.jit.trace_module>` for details).\n\n    Keyword arguments:\n        example_inputs (tuple or torch.Tensor or None, optional): A tuple of example\n            inputs that will be passed to the function while tracing.\n            Default: ``None``. Either this argument or ``example_kwarg_inputs``\n            should be specified. The resulting trace can be run with inputs of\n            different types and shapes assuming the traced operations support those\n            types and shapes. `example_inputs` may also be a single Tensor in which\n            case it is automatically wrapped in a tuple. When the value is None,\n            ``example_kwarg_inputs`` should be specified.\n\n        check_trace (``bool``, optional): Check if the same inputs run through\n            traced code produce the same outputs. Default: ``True``. You might want\n            to disable this if, for example, your network contains non-\n            deterministic ops or if you are sure that the network is correct despite\n            a checker failure.\n\n        check_inputs (list of tuples, optional): A list of tuples of input\n            arguments that should be used to check the trace against what is\n            expected. Each tuple is equivalent to a set of input arguments that\n            would be specified in ``example_inputs``. For best results, pass in\n            a set of checking inputs representative of the space of shapes and\n            types of inputs you expect the network to see.  If not specified,\n            the original ``example_inputs`` are used for checking\n        check_tolerance (float, optional): Floating-point comparison tolerance\n            to use in the checker procedure.  This can be used to relax the\n            checker strictness in the event that results diverge numerically\n            for a known reason, such as operator fusion.\n        strict (``bool``, optional): run the tracer in a strict mode or not\n            (default: ``True``). Only turn this off when you want the tracer to\n            record your mutable container types (currently ``list``/``dict``)\n            and you are sure that the container you are using in your\n            problem is a ``constant`` structure and does not get used as\n            control flow (if, for) conditions.\n        example_kwarg_inputs (dict, optional): This parameter is a pack of keyword\n            arguments of example inputs that will be passed to the function while\n            tracing. Default: ``None``. Either this argument or ``example_inputs``\n            should be specified. The dict will be unpacking by the arguments name\n            of the traced function. If the keys of the dict don't not match with\n            the traced function's arguments name, a runtime exception will be raised.\n\n    Returns:\n        If `func` is `nn.Module` or ``forward`` of `nn.Module`, `trace` returns\n        a :class:`ScriptModule` object with a single ``forward`` method\n        containing the traced code.  The returned `ScriptModule` will\n        have the same set of sub-modules and parameters as the original\n        ``nn.Module``.  If ``func`` is a standalone function, ``trace``\n        returns `ScriptFunction`.\n\n    Example (tracing a function):\n\n    .. testcode::\n\n        import torch\n\n        def foo(x, y):\n            return 2 * x + y\n\n        # Run `foo` with the provided inputs and record the tensor operations\n        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))\n\n        # `traced_foo` can now be run with the TorchScript interpreter or saved\n        # and loaded in a Python-free environment\n\n    Example (tracing an existing module)::\n\n        import torch\n        import torch.nn as nn\n\n        class Net(nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.conv = nn.Conv2d(1, 1, 3)\n\n            def forward(self, x):\n                return self.conv(x)\n\n        n = Net()\n        example_weight = torch.rand(1, 1, 3, 3)\n        example_forward_input = torch.rand(1, 1, 3, 3)\n\n        # Trace a specific method and construct `ScriptModule` with\n        # a single `forward` method\n        module = torch.jit.trace(n.forward, example_forward_input)\n\n        # Trace a module (implicitly traces `forward`) and construct a\n        # `ScriptModule` with a single `forward` method\n        module = torch.jit.trace(n, example_forward_input)\n\n    "
    if not _enabled:
        return func
    if optimize is not None:
        warnings.warn('`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead')
    if isinstance(func, torch.jit.ScriptModule):
        warnings.warn('The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.')
        return func
    if isinstance(func, torch.nn.Module):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError('example_kwarg_inputs should be a dict')
        return trace_module(func, {'forward': example_inputs}, None, check_trace, wrap_check_inputs(check_inputs), check_tolerance, strict, _force_outplace, _module_class, example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict), _store_inputs=_store_inputs)
    if hasattr(func, '__self__') and isinstance(func.__self__, torch.nn.Module) and (func.__name__ == 'forward'):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError('example_kwarg_inputs should be a dict')
        return trace_module(func.__self__, {'forward': example_inputs}, None, check_trace, wrap_check_inputs(check_inputs), check_tolerance, strict, _force_outplace, _module_class, example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict), _store_inputs=_store_inputs)
    if isinstance(example_inputs, (torch.Tensor, dict)) and example_kwarg_inputs is None:
        example_inputs = (example_inputs,)
    elif example_kwarg_inputs is None and (not isinstance(example_inputs, tuple)):
        example_inputs = tuple(example_inputs)
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)
    if hasattr(func, '__self__') and isinstance(func.__self__, torch.nn.Module):
        raise AttributeError("trace doesn't support compiling individual module's functions.\nPlease use trace_module")
    name = _qualified_name(func)
    if isinstance(example_kwarg_inputs, dict):
        example_inputs = example_kwarg_inputs
        traced = torch._C._create_function_from_trace_with_dict(name, func, example_kwarg_inputs, var_lookup_fn, strict, _force_outplace, get_callable_argument_names(func))
    else:
        traced = torch._C._create_function_from_trace(name, func, example_inputs, var_lookup_fn, strict, _force_outplace, get_callable_argument_names(func))
    if check_trace:
        if check_inputs is not None:
            _check_trace(check_inputs, func, traced, check_tolerance, strict, _force_outplace, False, _module_class, example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict))
        else:
            _check_trace([example_inputs], func, traced, check_tolerance, strict, _force_outplace, False, _module_class, example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict))
    traced._torchdynamo_inline = func
    return traced
_trace_module_map: Optional[Dict[Any, Any]] = None

def trace_module(mod, inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu, example_inputs_is_kwarg=False, _store_inputs=True):
    if False:
        while True:
            i = 10
    "\n    Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation.\n\n    When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only\n    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of\n    method names to example inputs to trace (see the ``inputs``) argument below.\n\n    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.\n\n    Args:\n        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are\n                                specified in ``inputs``. The given methods will be compiled\n                                as a part of a single `ScriptModule`.\n        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.\n                                The inputs will be passed to methods whose names correspond to inputs'\n                                keys while tracing.\n                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``\n    Keyword arguments:\n        check_trace (``bool``, optional): Check if the same inputs run through\n                                      traced code produce the same outputs. Default: ``True``. You might want\n                                      to disable this if, for example, your network contains non-\n                                      deterministic ops or if you are sure that the network is correct despite\n                                      a checker failure.\n\n        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used\n                                                 to check the trace against what is expected. Each tuple\n                                                 is equivalent to a set of input arguments that would\n                                                 be specified in ``inputs``. For best results, pass in a\n                                                 set of checking inputs representative of the space of\n                                                 shapes and types of inputs you expect the network to see.\n                                                 If not specified, the original ``inputs`` are used for checking\n        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.\n                                           This can be used to relax the checker strictness in the event that\n                                           results diverge numerically for a known reason, such as operator fusion.\n        example_inputs_is_kwarg (``bool``, optional): This parameter indicate whether the example inputs is a pack\n                                           pack of keyword arguments. Default: ``False``.\n\n    Returns:\n        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.\n        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of\n        sub-modules and parameters as ``func``.\n\n    Example (tracing a module with multiple methods)::\n\n        import torch\n        import torch.nn as nn\n\n        class Net(nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.conv = nn.Conv2d(1, 1, 3)\n\n            def forward(self, x):\n                return self.conv(x)\n\n            def weighted_kernel_sum(self, weight):\n                return weight * self.conv.weight\n\n\n        n = Net()\n        example_weight = torch.rand(1, 1, 3, 3)\n        example_forward_input = torch.rand(1, 1, 3, 3)\n\n        # Trace a specific method and construct `ScriptModule` with\n        # a single `forward` method\n        module = torch.jit.trace(n.forward, example_forward_input)\n\n        # Trace a module (implicitly traces `forward`) and construct a\n        # `ScriptModule` with a single `forward` method\n        module = torch.jit.trace(n, example_forward_input)\n\n        # Trace specific methods on a module (specified in `inputs`), constructs\n        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods\n        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}\n        module = torch.jit.trace_module(n, inputs)\n\n    "
    if not _enabled:
        return mod
    if optimize is not None:
        warnings.warn('`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead')
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)
    if not isinstance(mod, torch.nn.Module):
        raise AttributeError('expected torch.nn.Module as the first argument')
    if not isinstance(inputs, dict):
        raise AttributeError('expected a dictionary of (method_name, input) pairs')
    old_module_map = torch.jit._trace._trace_module_map
    try:
        trace_module_map: Dict[Any, Any] = {}

        def register_submods(mod, prefix):
            if False:
                return 10
            for (name, child) in mod.named_children():
                submod_qualname = prefix + '.' + name
                trace_module_map[child] = submod_qualname
                register_submods(child, submod_qualname)
        trace_module_map['__module'] = mod
        torch.jit._trace._trace_module_map = trace_module_map
        register_submods(mod, '__module')
        module = make_module(mod, _module_class, _compilation_unit)
        for (method_name, example_inputs) in inputs.items():
            if method_name == 'forward':
                func = mod
                forward_method = getattr(mod, method_name)
                argument_names = get_callable_argument_names(forward_method)
            else:
                func = getattr(mod, method_name)
                argument_names = get_callable_argument_names(func)
            if isinstance(example_inputs, dict) and example_inputs_is_kwarg:
                for key in example_inputs:
                    if key not in argument_names:
                        valid_arguments = '[' + ','.join(argument_names) + ']'
                        raise NameError(f"'{key}' is not in forward() method's arguments,\n                         valid arguments name are {valid_arguments}")
                module._c._create_method_from_trace_with_dict(method_name, func, example_inputs, var_lookup_fn, strict, _force_outplace, argument_names, _store_inputs)
            else:
                example_inputs = make_tuple(example_inputs)
                module._c._create_method_from_trace(method_name, func, example_inputs, var_lookup_fn, strict, _force_outplace, argument_names, _store_inputs)
            check_trace_method = module._c._get_method(method_name)
            if check_trace:
                if check_inputs is not None:
                    _check_trace(check_inputs, func, check_trace_method, check_tolerance, strict, _force_outplace, True, _module_class, example_inputs_is_kwarg=example_inputs_is_kwarg)
                else:
                    _check_trace([inputs], func, check_trace_method, check_tolerance, strict, _force_outplace, True, _module_class, example_inputs_is_kwarg=example_inputs_is_kwarg)
    finally:
        torch.jit._trace._trace_module_map = old_module_map
    return module

def is_tracing():
    if False:
        while True:
            i = 10
    'Return a boolean value.\n\n    Returns ``True`` in tracing (if a function is called during the\n    tracing of code with ``torch.jit.trace``) and ``False`` otherwise.\n    '
    if is_scripting():
        return False
    return torch._C._is_tracing()

class TracedModule(ScriptModule):
    _disable_script_meta = True

    def __init__(self, orig, id_set=None, _compilation_unit=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert isinstance(orig, torch.nn.Module)
        id_set = set()

        class QualnameWrapper(torch.nn.Module):
            pass
        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(type(orig))
        tmp_module = QualnameWrapper()

        def check_unique(param):
            if False:
                for i in range(10):
                    print('nop')
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)
        tmp_module.training = orig.training
        for (name, param) in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        for (name, buf) in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        for (name, val) in orig.__dict__.items():
            if torch._C._jit_is_script_object(val) and name not in orig._parameters and (name not in orig._buffers):
                setattr(tmp_module, name, val)
        if orig._backward_hooks:
            raise ValueError("Modules that have backward hooks assigned can't be compiled: " + str(orig))
        for (name, submodule) in orig._modules.items():
            if submodule is None:
                continue
            tmp_module._modules[name] = make_module(submodule, TracedModule, _compilation_unit=None)
        script_module = torch.jit._recursive.create_script_module(tmp_module, lambda module: (), share_types=False, is_tracing=True)
        self.__dict__['_name'] = type(orig).__name__
        self.__dict__['_actual_script_module'] = script_module
        for name in ('_parameters', '_buffers', '_modules', 'training'):
            delattr(self, name)

    def forward(self, *args, **kwargs):
        if False:
            return 10
        raise RuntimeError('Trace submodules cannot be called.')

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if '_actual_script_module' not in self.__dict__:
            return super().__getattr__(attr)
        return getattr(self._actual_script_module, attr)

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        if '_actual_script_module' not in self.__dict__:
            return super().__setattr__(attr, value)
        setattr(self._actual_script_module, attr, value)

    def _get_name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def extra_repr(self):
        if False:
            return 10
        return f'original_name={self._name}'

class TopLevelTracedModule(TracedModule):
    forward: Callable[..., Any] = _CachedForward()

    def _reconstruct(self, cpp_module):
        if False:
            i = 10
            return i + 15
        '\n        Re-construct an instance of TopLevelTracedModule using an instance of a C++ module.\n\n        Args:\n            cpp_module: The C++ module that this TopLevelTracedModule will be rebuilt around.\n        '
        self.__dict__['_actual_script_module']._reconstruct(cpp_module)

def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]:
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if False:
            return 10
        if not is_tracing():
            return fn(*args, **kwargs)
        compiled_fn: Callable[P, R] = script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)
    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True
    return wrapper

def _get_trace_graph(f, args=(), kwargs=None, strict=True, _force_outplace=False, return_inputs=False, _return_inputs_states=False):
    if False:
        i = 10
        return i + 15
    'Return a tuple on tracing a function or model.\n\n    .. warning::\n        This function is internal-only and should only be used by the ONNX\n        exporter. If you are trying to get a graph through tracing, please go\n        through the public API instead::\n\n            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))\n            trace_graph = trace.graph\n\n    Trace a function or model, returning a tuple consisting of the both the\n    *trace* of an execution, as well as the original return value. If return_inputs,\n    also returns the trace inputs as part of the tuple\n\n    Tracing is guaranteed not to change the semantics of the function/module\n    that is traced.\n\n    Args:\n        f (torch.nn.Module or function): the function or module\n            to be traced.\n        args (tuple or Tensor): the positional arguments to pass to the\n            function/module to be traced.  A non-tuple is assumed to\n            be a single positional argument to be passed to the model.\n        kwargs (dict): the keyword arguments to pass to the function/module\n            to be traced.\n\n    Example (trace a cell):\n\n    .. testcode::\n\n        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))\n    '
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
    return outs