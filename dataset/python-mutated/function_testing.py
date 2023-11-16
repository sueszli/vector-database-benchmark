import copy
import time
from typing import Union, List, Optional
import numpy as np
import types
import importlib
import inspect
from collections import OrderedDict
from .globals import mod_backend
try:
    import tensorflow as tf
except ImportError:
    tf = types.SimpleNamespace()
    tf.TensorShape = None
from .pipeline_helper import BackendHandler, BackendHandlerMode, get_frontend_config
import ivy
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
import ivy_tests.test_ivy.helpers.globals as t_globals
from ivy.functional.ivy.data_type import _get_function_list, _get_functions_from_string
from ivy_tests.test_ivy.test_frontends import NativeClass
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
from ivy_tests.test_ivy.helpers.testing_helpers import _create_transpile_report
from .assertions import value_test, assert_same_type, check_unsupported_dtype

def traced_if_required(backend: str, fn, test_trace=False, args=None, kwargs=None):
    if False:
        return 10
    with BackendHandler.update_backend(backend) as ivy_backend:
        if test_trace:
            fn = ivy_backend.trace_graph(fn, args=args, kwargs=kwargs)
    return fn

def _find_instance_in_args(backend: str, args, array_indices, mask):
    if False:
        return 10
    '\n    Find the first element in the arguments that is considered to be an instance of\n    Array or Container class.\n\n    Parameters\n    ----------\n    args\n        Arguments to iterate over\n    array_indices\n        Indices of arrays that exists in the args\n    mask\n        Boolean mask for whether the corresponding element in (args) has a\n        generated test_flags.native_array as False or test_flags.container as\n        true\n\n    Returns\n    -------\n        First found instance in the arguments and the updates arguments not\n        including the instance\n    '
    i = 0
    for (i, a) in enumerate(mask):
        if a:
            break
    instance_idx = array_indices[i]
    with BackendHandler.update_backend(backend) as ivy_backend:
        instance = ivy_backend.index_nest(args, instance_idx)
        new_args = ivy_backend.copy_nest(args, to_mutable=False)
        ivy_backend.prune_nest_at_index(new_args, instance_idx)
    return (instance, new_args)

def _get_frontend_submodules(fn_tree: str, gt_fn_tree: str):
    if False:
        i = 10
        return i + 15
    split_index = fn_tree.rfind('.')
    (frontend_submods, fn_name) = (fn_tree[:split_index], fn_tree[split_index + 1:])
    if gt_fn_tree is not None:
        split_index = gt_fn_tree.rfind('.')
        (gt_frontend_submods, gt_fn_name) = (gt_fn_tree[:split_index], gt_fn_tree[split_index + 1:])
    else:
        (gt_frontend_submods, gt_fn_name) = (fn_tree[25:fn_tree.rfind('.')], fn_name)
    return (frontend_submods, fn_name, gt_frontend_submods, gt_fn_name)

def test_function_backend_computation(fw, test_flags, all_as_kwargs_np, input_dtypes, on_device, fn_name):
    if False:
        for i in range(10):
            print('nop')
    (args_np, kwargs_np) = kwargs_to_args_n_kwargs(num_positional_args=test_flags.num_positional_args, kwargs=all_as_kwargs_np)
    (arg_np_arrays, arrays_args_indices, n_args_arrays) = _get_nested_np_arrays(args_np)
    (kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays) = _get_nested_np_arrays(kwargs_np)
    total_num_arrays = n_args_arrays + n_kwargs_arrays
    if len(input_dtypes) < total_num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(total_num_arrays)]
    if len(test_flags.as_variable) < total_num_arrays:
        test_flags.as_variable = [test_flags.as_variable[0] for _ in range(total_num_arrays)]
    if len(test_flags.native_arrays) < total_num_arrays:
        test_flags.native_arrays = [test_flags.native_arrays[0] for _ in range(total_num_arrays)]
    if len(test_flags.container) < total_num_arrays:
        test_flags.container = [test_flags.container[0] for _ in range(total_num_arrays)]
    with BackendHandler.update_backend(fw) as ivy_backend:
        test_flags.as_variable = [v if ivy_backend.is_float_dtype(d) and (not test_flags.with_out) else False for (v, d) in zip(test_flags.as_variable, input_dtypes)]
    instance_method = test_flags.instance_method and (not test_flags.native_arrays[0] or test_flags.container[0])
    (args, kwargs) = create_args_kwargs(backend=fw, args_np=args_np, arg_np_vals=arg_np_arrays, args_idxs=arrays_args_indices, kwargs_np=kwargs_np, kwarg_np_vals=kwarg_np_arrays, kwargs_idxs=arrays_kwargs_indices, input_dtypes=input_dtypes, test_flags=test_flags, on_device=on_device)
    if ('out' in kwargs or test_flags.with_out) and 'out' not in inspect.signature(getattr(ivy, fn_name)).parameters:
        raise Exception(f'Function {fn_name} does not have an out parameter')
    with BackendHandler.update_backend(fw) as ivy_backend:
        instance = None
        if instance_method:
            array_or_container_mask = [not native_flag or container_flag for (native_flag, container_flag) in zip(test_flags.native_arrays, test_flags.container)]
            args_instance_mask = array_or_container_mask[:test_flags.num_positional_args]
            kwargs_instance_mask = array_or_container_mask[test_flags.num_positional_args:]
            if any(args_instance_mask):
                (instance, args) = _find_instance_in_args(fw, args, arrays_args_indices, args_instance_mask)
            else:
                (instance, kwargs) = _find_instance_in_args(fw, kwargs, arrays_kwargs_indices, kwargs_instance_mask)
            if test_flags.test_trace:

                def target_fn(instance, *args, **kwargs):
                    if False:
                        return 10
                    return instance.__getattribute__(fn_name)(*args, **kwargs)
                args = [instance, *args]
            else:
                target_fn = instance.__getattribute__(fn_name)
        else:
            target_fn = ivy_backend.__dict__[fn_name]
        copy_kwargs = copy.deepcopy(kwargs)
        copy_args = copy.deepcopy(args)
        (ret_from_target, ret_np_flat_from_target) = get_ret_and_flattened_np_array(fw, target_fn, *copy_args, test_trace=test_flags.test_trace, **copy_kwargs)
        assert ivy_backend.nested_map(lambda x: ivy_backend.is_ivy_array(x) if ivy_backend.is_array(x) else True, ret_from_target), f'Ivy function returned non-ivy arrays: {ret_from_target}'
        if test_flags.with_out and (not test_flags.test_trace):
            test_ret = ret_from_target[getattr(ivy_backend.__dict__[fn_name], 'out_index')] if hasattr(ivy_backend.__dict__[fn_name], 'out_index') else ret_from_target
            out = ivy_backend.nested_map(ivy_backend.zeros_like, test_ret, to_mutable=True, include_derived=True)
            if instance_method:
                (ret_from_target, ret_np_flat_from_target) = get_ret_and_flattened_np_array(fw, instance.__getattribute__(fn_name), *args, **kwargs, out=out)
            else:
                (ret_from_target, ret_np_flat_from_target) = get_ret_and_flattened_np_array(fw, ivy_backend.__dict__[fn_name], *args, **kwargs, out=out)
            test_ret = ret_from_target[getattr(ivy_backend.__dict__[fn_name], 'out_index')] if hasattr(ivy_backend.__dict__[fn_name], 'out_index') else ret_from_target
            assert not ivy_backend.nested_any(ivy_backend.nested_multi_map(lambda x, _: x[0] is x[1], [test_ret, out]), lambda x: not x), 'the array in out argument does not contain same value as the returned'
            if not max(test_flags.container) and ivy_backend.native_inplace_support:
                assert not ivy_backend.nested_any(ivy_backend.nested_multi_map(lambda x, _: x[0].data is x[1].data, [test_ret, out]), lambda x: not x), 'the array in out argument does not contain same value as the returned'
        if test_flags.with_copy:
            array_fn = ivy_backend.is_array
            if 'copy' in list(inspect.signature(target_fn).parameters.keys()):
                kwargs['copy'] = True
            if instance_method:
                first_array = instance
            else:
                first_array = ivy_backend.func_wrapper._get_first_array(*args, array_fn=array_fn, **kwargs)
            (ret_, ret_np_flat_) = get_ret_and_flattened_np_array(fw, target_fn, *args, test_trace=test_flags.test_trace, precision_mode=test_flags.precision_mode, **kwargs)
            assert not np.may_share_memory(first_array, ret_)
    ret_device = None
    if isinstance(ret_from_target, ivy_backend.Array):
        ret_device = ivy_backend.dev(ret_from_target)
    return (ret_from_target, ret_np_flat_from_target, ret_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, test_flags, input_dtypes)

def test_function_ground_truth_computation(ground_truth_backend, on_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, input_dtypes, test_flags, fn_name):
    if False:
        return 10
    with BackendHandler.update_backend(ground_truth_backend) as gt_backend:
        gt_backend.set_default_device(on_device)
        (args, kwargs) = create_args_kwargs(backend=test_flags.ground_truth_backend, args_np=args_np, arg_np_vals=arg_np_arrays, args_idxs=arrays_args_indices, kwargs_np=kwargs_np, kwargs_idxs=arrays_kwargs_indices, kwarg_np_vals=kwarg_np_arrays, input_dtypes=input_dtypes, test_flags=test_flags, on_device=on_device)
        (ret_from_gt, ret_np_from_gt_flat) = get_ret_and_flattened_np_array(test_flags.ground_truth_backend, gt_backend.__dict__[fn_name], *args, test_trace=test_flags.test_trace, precision_mode=test_flags.precision_mode, **kwargs)
        assert gt_backend.nested_map(lambda x: gt_backend.is_ivy_array(x) if gt_backend.is_array(x) else True, ret_from_gt), f'Ground-truth function returned non-ivy arrays: {ret_from_gt}'
        if test_flags.with_out and (not test_flags.test_trace):
            test_ret_from_gt = ret_from_gt[getattr(gt_backend.__dict__[fn_name], 'out_index')] if hasattr(gt_backend.__dict__[fn_name], 'out_index') else ret_from_gt
            out_from_gt = gt_backend.nested_map(gt_backend.zeros_like, test_ret_from_gt, to_mutable=True, include_derived=True)
            (ret_from_gt, ret_np_from_gt_flat) = get_ret_and_flattened_np_array(test_flags.ground_truth_backend, gt_backend.__dict__[fn_name], *args, test_trace=test_flags.test_trace, precision_mode=test_flags.precision_mode, **kwargs, out=out_from_gt)
        fw_list = gradient_unsupported_dtypes(fn=gt_backend.__dict__[fn_name])
        ret_from_gt_device = None
        if isinstance(ret_from_gt, gt_backend.Array):
            ret_from_gt_device = gt_backend.dev(ret_from_gt)
    return (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, test_flags, fw_list)

def test_function(*, input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]], test_flags: FunctionTestFlags, fn_name: str, rtol_: Optional[float]=None, atol_: float=1e-06, tolerance_dict: Optional[dict]=None, test_values: bool=True, xs_grad_idxs=None, ret_grad_idxs=None, backend_to_test: str, on_device: str, return_flat_np_arrays: bool=False, **all_as_kwargs_np):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a function that consumes (or returns) arrays for the current backend by\n    comparing the result with numpy.\n\n    Parameters\n    ----------\n    input_dtypes\n        data types of the input arguments in order.\n    test_flags\n        FunctionTestFlags object that stores all testing flags, including:\n        num_positional_args, with_out, instance_method, as_variable,\n        native_arrays, container, gradient\n    fw\n        current backend (framework).\n    fn_name\n        name of the function to test.\n    rtol_\n        relative tolerance value.\n    atol_\n        absolute tolerance value.\n    test_values\n        if True, test for the correctness of the resulting values.\n    xs_grad_idxs\n        Indices of the input arrays to compute gradients with respect to. If None,\n        gradients are returned with respect to all input arrays. (Default value = None)\n    ret_grad_idxs\n        Indices of the returned arrays for which to return computed gradients. If None,\n        gradients are returned for all returned arrays. (Default value = None)\n    on_device\n        The device on which to create arrays\n    return_flat_np_arrays\n        If test_values is False, this flag dictates whether the original returns are\n        returned, or whether the flattened numpy arrays are returned.\n    all_as_kwargs_np\n        input arguments to the function as keyword arguments.\n\n    Returns\n    -------\n    ret\n        optional, return value from the function\n    ret_gt\n        optional, return value from the Ground Truth function\n\n    Examples\n    --------\n    >>> input_dtypes = \'float64\'\n    >>> as_variable_flags = False\n    >>> with_out = False\n    >>> num_positional_args = 0\n    >>> native_array_flags = False\n    >>> container_flags = False\n    >>> instance_method = False\n    >>> test_flags = FunctionTestFlags(num_positional_args, with_out,\n        instance_method,\n        as_variable,\n        native_arrays,\n        container_flags,\n        none)\n    >>> fw = "torch"\n    >>> fn_name = "abs"\n    >>> x = np.array([-1])\n    >>> test_function(input_dtypes, test_flags, fw, fn_name, x=x)\n\n    >>> input_dtypes = [\'float64\', \'float32\']\n    >>> as_variable_flags = [False, True]\n    >>> with_out = False\n    >>> num_positional_args = 1\n    >>> native_array_flags = [True, False]\n    >>> container_flags = [False, False]\n    >>> instance_method = False\n    >>> test_flags = FunctionTestFlags(num_positional_args, with_out,\n        instance_method,\n        as_variable,\n        native_arrays,\n        container_flags,\n        none)\n    >>> fw = "numpy"\n    >>> fn_name = "add"\n    >>> x1 = np.array([1, 3, 4])\n    >>> x2 = np.array([-3, 15, 24])\n    >>> test_function(input_dtypes, test_flags, fw, fn_name, x1=x1, x2=x2)\n    '
    _switch_backend_context(test_flags.test_trace or test_flags.transpile)
    ground_truth_backend = test_flags.ground_truth_backend
    if test_flags.container[0]:
        test_flags.with_copy = False
    if test_flags.with_copy is True:
        test_flags.with_out = False
    if mod_backend[backend_to_test]:
        (proc, input_queue, output_queue) = mod_backend[backend_to_test]
        input_queue.put(('function_backend_computation', backend_to_test, test_flags, all_as_kwargs_np, input_dtypes, on_device, fn_name))
        (ret_from_target, ret_np_flat_from_target, ret_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, test_flags, input_dtypes) = output_queue.get()
    else:
        (ret_from_target, ret_np_flat_from_target, ret_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, test_flags, input_dtypes) = test_function_backend_computation(backend_to_test, test_flags, all_as_kwargs_np, input_dtypes, on_device, fn_name)
    if mod_backend[ground_truth_backend]:
        (proc, input_queue, output_queue) = mod_backend[ground_truth_backend]
        input_queue.put(('function_ground_truth_computation', ground_truth_backend, on_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, input_dtypes, test_flags, fn_name))
        (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, test_flags, fw_list) = output_queue.get()
    else:
        (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, test_flags, fw_list) = test_function_ground_truth_computation(ground_truth_backend, on_device, args_np, arg_np_arrays, arrays_args_indices, kwargs_np, arrays_kwargs_indices, kwarg_np_arrays, input_dtypes, test_flags, fn_name)
    if test_flags.transpile:
        if mod_backend[backend_to_test]:
            (proc, input_queue, output_queue) = mod_backend[backend_to_test]
            input_queue.put(('transpile_if_required_backend', backend_to_test, fn_name, args_np, kwargs_np))
        else:
            _transpile_if_required_backend(backend_to_test, fn_name, args=args_np, kwargs=kwargs_np)
    if test_flags.test_gradients and (not test_flags.instance_method) and ('bool' not in input_dtypes) and (not any((d in ['complex64', 'complex128'] for d in input_dtypes))):
        if backend_to_test not in fw_list or not ivy.nested_argwhere(all_as_kwargs_np, lambda x: x.dtype in fw_list[backend_to_test] if isinstance(x, np.ndarray) else None):
            gradient_test(fn=fn_name, all_as_kwargs_np=all_as_kwargs_np, args_np=args_np, kwargs_np=kwargs_np, input_dtypes=input_dtypes, test_flags=test_flags, rtol_=rtol_, atol_=atol_, xs_grad_idxs=xs_grad_idxs, ret_grad_idxs=ret_grad_idxs, ground_truth_backend=ground_truth_backend, backend_to_test=backend_to_test, on_device=on_device)
    if not test_values:
        if return_flat_np_arrays:
            return (ret_np_flat_from_target, ret_np_from_gt_flat)
        return (ret_from_target, ret_from_gt)
    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, backend_to_test)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, backend_to_test)
    value_test(ret_np_flat=ret_np_flat_from_target, ret_np_from_gt_flat=ret_np_from_gt_flat, rtol=rtol_, atol=atol_, specific_tolerance_dict=tolerance_dict, backend=backend_to_test, ground_truth_backend=test_flags.ground_truth_backend)
    assert_same_type(ret_from_target, ret_from_gt, backend_to_test, test_flags.ground_truth_backend)
    assert ret_device == ret_from_gt_device, f'ground truth backend ({test_flags.ground_truth_backend}) returned array on device {ret_from_gt_device} but target backend ({backend_to_test}) returned array on device {ret_device}'
    if ret_device is not None:
        assert ret_device == on_device, f'device is set to {on_device}, but ground truth produced array on {ret_device}'

def _assert_frontend_ret(ret, for_fn=True):
    if False:
        return 10
    fn_or_method = 'function' if for_fn else 'method'
    if not inspect.isclass(ret):
        is_ret_tuple = issubclass(ret.__class__, tuple)
    else:
        is_ret_tuple = issubclass(ret, tuple)
    if is_ret_tuple:
        non_frontend_idxs = ivy.nested_argwhere(ret, lambda _x: not _is_frontend_array(_x) if ivy.is_array(_x) else False)
        assert not non_frontend_idxs, f'Frontend {fn_or_method} return contains non-frontend arrays at positions {non_frontend_idxs} (zero-based): {ivy.multi_index_nest(ret, non_frontend_idxs)}'
    elif ivy.is_array(ret):
        assert _is_frontend_array(ret), f'Frontend {fn_or_method} returned non-frontend array: {ret}'

def _transpile_if_required_backend(backend: str, fn_name: str, args=None, kwargs=None):
    if False:
        i = 10
        return i + 15
    iterations = 1
    with BackendHandler.update_backend(backend) as ivy_backend:
        (args, kwargs) = ivy_backend.args_to_ivy(*args, **kwargs)
        backend_fn = ivy.__dict__[fn_name]
    backend_traced_fn = traced_if_required(backend, backend_fn, test_trace=True, args=args, kwargs=kwargs)
    func_timings = []
    for i in range(0, iterations):
        start = time.time()
        backend_traced_fn(*args, **kwargs)
        end = time.time()
        func_timings.append(end - start)
    func_time = np.mean(func_timings).item()
    backend_nodes = len(backend_traced_fn._functions)
    data = {'fn_name': fn_name, 'args': str(args), 'kwargs': str(kwargs), 'time': func_time, 'nodes': backend_nodes}
    _create_transpile_report(data, backend, 'report.json', True)

def test_frontend_function(*, input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]], test_flags: pf.frontend_function_flags, backend_to_test: str, on_device='cpu', frontend: str, fn_tree: str, gt_fn_tree: Optional[str]=None, rtol: Optional[float]=None, atol: float=1e-06, tolerance_dict: Optional[dict]=None, test_values: bool=True, **all_as_kwargs_np):
    if False:
        print('Hello World!')
    '\n    Test a frontend function for the current backend by comparing the result with the\n    function in the associated framework.\n\n    Parameters\n    ----------\n    input_dtypes\n        data types of the input arguments in order.\n    test_flags\n        FunctionTestFlags object that stores all testing flags, including:\n        num_positional_args, with_out, instance_method, as_variable,\n        native_arrays, container, gradient, precision_mode\n    frontend\n        current frontend (framework).\n    fn_tree\n        Path to function in frontend framework namespace.\n    gt_fn_tree\n        Path to function in ground truth framework namespace.\n    rtol\n        relative tolerance value.\n    atol\n        absolute tolerance value.\n    tolerance_dict\n        dictionary of tolerance values for specific dtypes.\n    test_values\n        if True, test for the correctness of the resulting values.\n    all_as_kwargs_np\n        input arguments to the function as keyword arguments.\n\n    Returns\n    -------\n    ret\n        optional, return value from the function\n    ret_np\n        optional, return value from the Numpy function\n    '
    _switch_backend_context(test_flags.test_trace or test_flags.transpile)
    assert not test_flags.with_out or not test_flags.inplace, 'only one of with_out or with_inplace can be set as True'
    if test_flags.with_copy is True:
        test_flags.with_out = False
        test_flags.inplace = False
    (args_np, kwargs_np) = kwargs_to_args_n_kwargs(num_positional_args=test_flags.num_positional_args, kwargs=all_as_kwargs_np)
    (arg_np_vals, args_idxs, c_arg_vals) = _get_nested_np_arrays(args_np)
    (kwarg_np_vals, kwargs_idxs, c_kwarg_vals) = _get_nested_np_arrays(kwargs_np)
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(test_flags.as_variable) < num_arrays:
        test_flags.as_variable = [test_flags.as_variable[0] for _ in range(num_arrays)]
    if len(test_flags.native_arrays) < num_arrays:
        test_flags.native_arrays = [test_flags.native_arrays[0] for _ in range(num_arrays)]
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        test_flags.as_variable = [v if ivy_backend.is_float_dtype(d) and (not test_flags.with_out) else False for (v, d) in zip(test_flags.as_variable, input_dtypes)]
        local_importer = ivy_backend.utils.dynamic_import
        if frontend == 'jax':
            local_importer.import_module('ivy.functional.frontends.jax').config.update('jax_enable_x64', True)
        (frontend_submods, fn_name, gt_frontend_submods, gt_fn_name) = _get_frontend_submodules(fn_tree, gt_fn_tree)
        function_module = local_importer.import_module(frontend_submods)
        frontend_fn = getattr(function_module, fn_name)
        (args, kwargs) = create_args_kwargs(backend=backend_to_test, args_np=args_np, arg_np_vals=arg_np_vals, args_idxs=args_idxs, kwargs_np=kwargs_np, kwarg_np_vals=kwarg_np_vals, kwargs_idxs=kwargs_idxs, input_dtypes=input_dtypes, test_flags=test_flags, on_device=on_device)
        copy_kwargs = copy.deepcopy(kwargs)
        copy_args = copy.deepcopy(args)
        create_frontend_array = local_importer.import_module(f'ivy.functional.frontends.{frontend}')._frontend_array
        if test_flags.generate_frontend_arrays:
            (args_for_test, kwargs_for_test) = args_to_frontend(backend_to_test, *args, frontend_array_fn=create_frontend_array, **kwargs)
            (copy_args, copy_kwargs) = args_to_frontend(backend_to_test, *args, frontend_array_fn=create_frontend_array, **kwargs)
        else:
            args_for_test = copy.deepcopy(args)
            kwargs_for_test = copy.deepcopy(kwargs)
        ret = get_frontend_ret(backend_to_test, frontend_fn, *args_for_test, test_trace=test_flags.test_trace, frontend_array_function=create_frontend_array if test_flags.test_trace else None, precision_mode=test_flags.precision_mode, **kwargs_for_test)
        _assert_frontend_ret(ret)
        if test_flags.with_out and 'out' in list(inspect.signature(frontend_fn).parameters.keys()):
            if not inspect.isclass(ret):
                is_ret_tuple = issubclass(ret.__class__, tuple)
            else:
                is_ret_tuple = issubclass(ret, tuple)
            out = ret
            if is_ret_tuple:
                flatten_ret = flatten_frontend(ret=ret, backend=backend_to_test, frontend_array_fn=create_frontend_array)
                flatten_out = flatten_frontend(ret=out, backend=backend_to_test, frontend_array_fn=create_frontend_array)
                for (ret_array, out_array) in zip(flatten_ret, flatten_out):
                    if ivy_backend.native_inplace_support and (not any((ivy_backend.isscalar(ret), ivy_backend.isscalar(out)))):
                        assert ret_array.ivy_array.data is out_array.ivy_array.data
                    assert ret_array is out_array
            else:
                if ivy_backend.native_inplace_support and (not any((ivy_backend.isscalar(ret), ivy_backend.isscalar(out)))):
                    assert ret.ivy_array.data is out.ivy_array.data
                assert ret is out
        elif test_flags.with_copy:
            assert _is_frontend_array(ret)
            if 'copy' in list(inspect.signature(frontend_fn).parameters.keys()):
                copy_kwargs['copy'] = True
            first_array = ivy_backend.func_wrapper._get_first_array(*copy_args, array_fn=_is_frontend_array if test_flags.generate_frontend_arrays else ivy_backend.is_array, **copy_kwargs)
            ret_ = get_frontend_ret(backend_to_test, frontend_fn, *copy_args, test_trace=test_flags.test_trace, frontend_array_function=create_frontend_array if test_flags.test_trace else None, precision_mode=test_flags.precision_mode, **copy_kwargs)
            if test_flags.generate_frontend_arrays:
                first_array = first_array.ivy_array
            ret_ = ret_.ivy_array
            if 'bfloat16' in str(ret_.dtype):
                ret_ = ivy_backend.astype(ret_, ivy_backend.float64)
            if 'bfloat16' in str(first_array.dtype):
                first_array = ivy_backend.astype(first_array, ivy_backend.float64)
            if not ivy_backend.is_native_array(first_array):
                first_array = first_array.data
            ret_ = ret_.data
            if hasattr(first_array, 'requires_grad'):
                first_array.requires_grad = False
            assert not np.may_share_memory(first_array, ret_)
        elif test_flags.inplace:
            assert _is_frontend_array(ret)
            if 'inplace' in list(inspect.signature(frontend_fn).parameters.keys()):
                copy_kwargs['inplace'] = True
            first_array = ivy_backend.func_wrapper._get_first_array(*copy_args, array_fn=_is_frontend_array if test_flags.generate_frontend_arrays else ivy_backend.is_array, **copy_kwargs)
            ret_ = get_frontend_ret(backend_to_test, frontend_fn, *copy_args, test_trace=test_flags.test_trace, frontend_array_function=create_frontend_array if test_flags.test_trace else None, precision_mode=test_flags.precision_mode, **copy_kwargs)
            if test_flags.generate_frontend_arrays:
                assert first_array is ret_
            elif ivy_backend.is_native_array(first_array) and ivy_backend.inplace_arrays_supported():
                assert first_array is ret_.ivy_array.data
            elif ivy_backend.is_ivy_array(first_array):
                assert first_array.data is ret_.ivy_array.data
        if test_values:
            ret_np_flat = flatten_frontend_to_np(ret=ret, backend=backend_to_test)
        if not test_values:
            ret = ivy_backend.nested_map(_frontend_array_to_ivy, ret, include_derived={'tuple': True})
    frontend_config = get_frontend_config(frontend)
    args_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else frontend_config.as_native_dtype(x) if isinstance(x, frontend_config.Dtype) else x, args_np, shallow=False)
    kwargs_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else x, kwargs_np, shallow=False)
    if 'dtype' in kwargs_frontend and kwargs_frontend['dtype'] is not None:
        kwargs_frontend['dtype'] = frontend_config.as_native_dtype(kwargs_frontend['dtype'])
    if 'device' in kwargs_frontend:
        kwargs_frontend['device'] = frontend_config.as_native_device(kwargs_frontend['device'])
    frontend_fw = importlib.import_module(gt_frontend_submods)
    frontend_fw_fn = frontend_fw.__dict__[gt_fn_name]
    frontend_ret = frontend_fw_fn(*args_frontend, **kwargs_frontend)
    if test_flags.transpile:
        _get_transpiled_data_if_required(frontend_fn, frontend_fw_fn, frontend, backend_to_test, fn_name=f'{gt_frontend_submods}.{gt_fn_name}', generate_frontend_arrays=test_flags.generate_frontend_arrays, args_for_test=args_for_test, kwargs_for_test=kwargs_for_test, frontend_fw_args=args_frontend, frontend_fw_kwargs=kwargs_frontend)
    if test_values:
        frontend_ret_np_flat = flatten_frontend_fw_to_np(frontend_ret, frontend_config.isscalar, frontend_config.is_native_array, frontend_config.to_numpy)
    if not test_values:
        return (ret, frontend_ret)
    if isinstance(rtol, dict):
        rtol = _get_framework_rtol(rtol, t_globals.CURRENT_BACKEND)
    if isinstance(atol, dict):
        atol = _get_framework_atol(atol, t_globals.CURRENT_BACKEND)
    value_test(ret_np_flat=ret_np_flat, ret_np_from_gt_flat=frontend_ret_np_flat, rtol=rtol, atol=atol, specific_tolerance_dict=tolerance_dict, backend=backend_to_test, ground_truth_backend=frontend)

def test_gradient_backend_computation(backend_to_test, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, kwargs_idxs, input_dtypes, test_flags, on_device, fn, test_trace, xs_grad_idxs, ret_grad_idxs):
    if False:
        print('Hello World!')
    (args, kwargs) = create_args_kwargs(backend=backend_to_test, args_np=args_np, arg_np_vals=arg_np_vals, args_idxs=args_idxs, kwargs_np=kwargs_np, kwarg_np_vals=kwarg_np_vals, kwargs_idxs=kwargs_idxs, input_dtypes=input_dtypes, test_flags=test_flags, on_device=on_device)
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:

        def _grad_fn(all_args):
            if False:
                i = 10
                return i + 15
            (args, kwargs, i) = all_args
            call_fn = ivy_backend.__dict__[fn] if isinstance(fn, str) else fn[i]
            ret = traced_if_required(backend_to_test, call_fn, test_trace=test_trace, args=args, kwargs=kwargs)(*args, **kwargs)
            return ivy_backend.nested_map(ivy_backend.mean, ret, include_derived=True)
        with ivy_backend.PreciseMode(test_flags.precision_mode):
            (_, grads) = ivy_backend.execute_with_gradients(_grad_fn, [args, kwargs, 0], xs_grad_idxs=xs_grad_idxs, ret_grad_idxs=ret_grad_idxs)
    grads_np_flat = flatten_and_to_np(backend=backend_to_test, ret=grads)
    return grads_np_flat

def test_gradient_ground_truth_computation(ground_truth_backend, on_device, fn, input_dtypes, all_as_kwargs_np, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, test_flags, kwargs_idxs, test_trace, xs_grad_idxs, ret_grad_idxs):
    if False:
        return 10
    with BackendHandler.update_backend(ground_truth_backend) as gt_backend:
        gt_backend.set_default_device(on_device)
        if check_unsupported_dtype(fn=gt_backend.__dict__[fn] if isinstance(fn, str) else fn[1], input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np):
            return
        (args, kwargs) = create_args_kwargs(backend=ground_truth_backend, args_np=args_np, arg_np_vals=arg_np_vals, args_idxs=args_idxs, kwargs_np=kwargs_np, kwarg_np_vals=kwarg_np_vals, kwargs_idxs=kwargs_idxs, input_dtypes=input_dtypes, test_flags=test_flags, on_device=on_device)

        def _gt_grad_fn(all_args):
            if False:
                return 10
            (args, kwargs, i) = all_args
            call_fn = gt_backend.__dict__[fn] if isinstance(fn, str) else fn[i]
            ret = traced_if_required(ground_truth_backend, call_fn, test_trace=test_trace, args=args, kwargs=kwargs)(*args, **kwargs)
            return gt_backend.nested_map(gt_backend.mean, ret, include_derived=True)
        with gt_backend.PreciseMode(test_flags.precision_mode):
            (_, grads_from_gt) = gt_backend.execute_with_gradients(_gt_grad_fn, [args, kwargs, 1], xs_grad_idxs=xs_grad_idxs, ret_grad_idxs=ret_grad_idxs)
        grads_np_from_gt_flat = flatten_and_to_np(backend=ground_truth_backend, ret=grads_from_gt)
    return grads_np_from_gt_flat

def gradient_test(*, fn, all_as_kwargs_np, args_np, kwargs_np, input_dtypes, test_flags, test_trace: bool=False, rtol_: Optional[float]=None, atol_: float=1e-06, tolerance_dict=None, xs_grad_idxs=None, ret_grad_idxs=None, backend_to_test: str, ground_truth_backend: str, on_device: str):
    if False:
        print('Hello World!')
    (arg_np_vals, args_idxs, _) = _get_nested_np_arrays(args_np)
    (kwarg_np_vals, kwargs_idxs, _) = _get_nested_np_arrays(kwargs_np)
    if mod_backend[backend_to_test]:
        (proc, input_queue, output_queue) = mod_backend[backend_to_test]
        input_queue.put(('gradient_backend_computation', backend_to_test, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, kwargs_idxs, input_dtypes, test_flags, on_device, fn, test_trace, xs_grad_idxs, ret_grad_idxs))
        grads_np_flat = output_queue.get()
    else:
        grads_np_flat = test_gradient_backend_computation(backend_to_test, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, kwargs_idxs, input_dtypes, test_flags, on_device, fn, test_trace, xs_grad_idxs, ret_grad_idxs)
    if mod_backend[ground_truth_backend]:
        (proc, input_queue, output_queue) = mod_backend[ground_truth_backend]
        input_queue.put(('gradient_ground_truth_computation', ground_truth_backend, on_device, fn, input_dtypes, all_as_kwargs_np, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, test_flags, kwargs_idxs, test_trace, xs_grad_idxs, ret_grad_idxs))
        grads_np_from_gt_flat = output_queue.get()
    else:
        grads_np_from_gt_flat = test_gradient_ground_truth_computation(ground_truth_backend, on_device, fn, input_dtypes, all_as_kwargs_np, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, test_flags, kwargs_idxs, test_trace, xs_grad_idxs, ret_grad_idxs)
    assert len(grads_np_flat) == len(grads_np_from_gt_flat), f'result length mismatch: {grads_np_flat} ({len(grads_np_flat)}) != {grads_np_from_gt_flat} ({len(grads_np_from_gt_flat)})'
    value_test(ret_np_flat=grads_np_flat, ret_np_from_gt_flat=grads_np_from_gt_flat, rtol=rtol_, atol=atol_, specific_tolerance_dict=tolerance_dict, backend=backend_to_test, ground_truth_backend=ground_truth_backend)

def test_method_backend_computation(init_input_dtypes, init_flags, backend_to_test, init_all_as_kwargs_np, on_device, method_input_dtypes, method_flags, method_all_as_kwargs_np, class_name, method_name, init_with_v, test_trace, method_with_v):
    if False:
        return 10
    init_input_dtypes = ivy.default(init_input_dtypes, [])
    init_all_as_kwargs_np = ivy.default(init_all_as_kwargs_np, {})
    (args_np_constructor, kwargs_np_constructor) = kwargs_to_args_n_kwargs(num_positional_args=init_flags.num_positional_args, kwargs=init_all_as_kwargs_np)
    (con_arg_np_vals, con_args_idxs, con_c_arg_vals) = _get_nested_np_arrays(args_np_constructor)
    (con_kwarg_np_vals, con_kwargs_idxs, con_c_kwarg_vals) = _get_nested_np_arrays(kwargs_np_constructor)
    num_arrays_constructor = con_c_arg_vals + con_c_kwarg_vals
    if len(init_input_dtypes) < num_arrays_constructor:
        init_input_dtypes = [init_input_dtypes[0] for _ in range(num_arrays_constructor)]
    if len(init_flags.as_variable) < num_arrays_constructor:
        init_flags.as_variable = [init_flags.as_variable[0] for _ in range(num_arrays_constructor)]
    if len(init_flags.native_arrays) < num_arrays_constructor:
        init_flags.native_arrays = [init_flags.native_arrays[0] for _ in range(num_arrays_constructor)]
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        init_flags.as_variable = [v if ivy_backend.is_float_dtype(d) else False for (v, d) in zip(init_flags.as_variable, init_input_dtypes)]
    constructor_data = OrderedDict(args_np=args_np_constructor, arg_np_vals=con_arg_np_vals, args_idxs=con_args_idxs, kwargs_np=kwargs_np_constructor, kwarg_np_vals=con_kwarg_np_vals, kwargs_idxs=con_kwargs_idxs, input_dtypes=init_input_dtypes, test_flags=init_flags, on_device=on_device)
    org_con_data = copy.deepcopy(constructor_data)
    (args_constructor, kwargs_constructor) = create_args_kwargs(backend=backend_to_test, **constructor_data)
    method_input_dtypes = ivy.default(method_input_dtypes, [])
    (args_np_method, kwargs_np_method) = kwargs_to_args_n_kwargs(num_positional_args=method_flags.num_positional_args, kwargs=method_all_as_kwargs_np)
    (met_arg_np_vals, met_args_idxs, met_c_arg_vals) = _get_nested_np_arrays(args_np_method)
    (met_kwarg_np_vals, met_kwargs_idxs, met_c_kwarg_vals) = _get_nested_np_arrays(kwargs_np_method)
    num_arrays_method = met_c_arg_vals + met_c_kwarg_vals
    if len(method_input_dtypes) < num_arrays_method:
        method_input_dtypes = [method_input_dtypes[0] for _ in range(num_arrays_method)]
    if len(method_flags.as_variable) < num_arrays_method:
        method_flags.as_variable = [method_flags.as_variable[0] for _ in range(num_arrays_method)]
    if len(method_flags.native_arrays) < num_arrays_method:
        method_flags.native_arrays = [method_flags.native_arrays[0] for _ in range(num_arrays_method)]
    if len(method_flags.container) < num_arrays_method:
        method_flags.container = [method_flags.container[0] for _ in range(num_arrays_method)]
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        method_flags.as_variable = [v if ivy_backend.is_float_dtype(d) else False for (v, d) in zip(method_flags.as_variable, method_input_dtypes)]
    (args_method, kwargs_method) = create_args_kwargs(backend=backend_to_test, args_np=args_np_method, arg_np_vals=met_arg_np_vals, args_idxs=met_args_idxs, kwargs_np=kwargs_np_method, kwarg_np_vals=met_kwarg_np_vals, kwargs_idxs=met_kwargs_idxs, input_dtypes=method_input_dtypes, test_flags=method_flags, on_device=on_device)
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        ins = ivy_backend.__dict__[class_name](*args_constructor, **kwargs_constructor)
        if any((dtype in ivy_backend.function_unsupported_dtypes(ins.__getattribute__(method_name)) for dtype in method_input_dtypes)):
            return
        v_np = None
        if isinstance(ins, ivy_backend.Module):
            if init_with_v:
                v = ivy_backend.Container(ins._create_variables(device=on_device, dtype=method_input_dtypes[0]))
                ins = ivy_backend.__dict__[class_name](*args_constructor, **kwargs_constructor, v=v)
            v = ins.__getattribute__('v')
            v_np = v.cont_map(lambda x, kc: ivy_backend.to_numpy(x) if ivy_backend.is_array(x) else x)
            if method_with_v:
                kwargs_method = dict(**kwargs_method, v=v)
        (ret, ret_np_flat) = get_ret_and_flattened_np_array(backend_to_test, ins.__getattribute__(method_name), *args_method, test_trace=test_trace, precision_mode=method_flags.precision_mode, **kwargs_method)
        if isinstance(ret, ivy_backend.Array):
            ret_device = ivy_backend.dev(ret)
        else:
            ret_device = None
    fw_list = gradient_unsupported_dtypes(fn=ins.__getattribute__(method_name))
    return (ret, ret_np_flat, ret_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, v_np, fw_list)

def test_method_ground_truth_computation(ground_truth_backend, on_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, method_input_dtypes, method_flags, class_name, method_name, test_trace, v_np):
    if False:
        print('Hello World!')
    with BackendHandler.update_backend(ground_truth_backend) as gt_backend:
        gt_backend.set_default_device(on_device)
        (args_gt_constructor, kwargs_gt_constructor) = create_args_kwargs(backend=ground_truth_backend, **org_con_data)
        (args_gt_method, kwargs_gt_method) = create_args_kwargs(backend=ground_truth_backend, args_np=args_np_method, arg_np_vals=met_arg_np_vals, args_idxs=met_args_idxs, kwargs_np=kwargs_np_method, kwarg_np_vals=met_kwarg_np_vals, kwargs_idxs=met_kwargs_idxs, input_dtypes=method_input_dtypes, test_flags=method_flags, on_device=on_device)
        ins_gt = gt_backend.__dict__[class_name](*args_gt_constructor, **kwargs_gt_constructor)
        if any((dtype in gt_backend.function_unsupported_dtypes(ins_gt.__getattribute__(method_name)) for dtype in method_input_dtypes)):
            return
        if isinstance(ins_gt, gt_backend.Module):
            v_gt = v_np.cont_map(lambda x, kc: gt_backend.asarray(x) if isinstance(x, np.ndarray) else x)
            kwargs_gt_method = dict(**kwargs_gt_method, v=v_gt)
        (ret_from_gt, ret_np_from_gt_flat) = get_ret_and_flattened_np_array(ground_truth_backend, ins_gt.__getattribute__(method_name), *args_gt_method, test_trace=test_trace, precision_mode=method_flags.precision_mode, **kwargs_gt_method)
        assert gt_backend.nested_map(lambda x: gt_backend.is_ivy_array(x) if gt_backend.is_array(x) else True, ret_from_gt), f'Ground-truth method returned non-ivy arrays: {ret_from_gt}'
        fw_list2 = gradient_unsupported_dtypes(fn=ins_gt.__getattribute__(method_name))
        if isinstance(ret_from_gt, gt_backend.Array):
            ret_from_gt_device = gt_backend.dev(ret_from_gt)
        else:
            ret_from_gt_device = None
    return (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, fw_list2)

def test_method(*, init_input_dtypes: Optional[List[ivy.Dtype]]=None, method_input_dtypes: Optional[List[ivy.Dtype]]=None, init_all_as_kwargs_np: Optional[dict]=None, method_all_as_kwargs_np: Optional[dict]=None, init_flags: pf.MethodTestFlags, method_flags: pf.MethodTestFlags, class_name: str, method_name: str='__call__', init_with_v: bool=False, method_with_v: bool=False, rtol_: Optional[float]=None, atol_: float=1e-06, tolerance_dict=None, test_values: Union[bool, str]=True, test_gradients: bool=False, xs_grad_idxs=None, ret_grad_idxs=None, test_trace: bool=False, backend_to_test: str, ground_truth_backend: str, on_device: str, return_flat_np_arrays: bool=False):
    if False:
        i = 10
        return i + 15
    '\n    Test a class-method that consumes (or returns) arrays for the current backend by\n    comparing the result with numpy.\n\n    Parameters\n    ----------\n    init_input_dtypes\n        data types of the input arguments to the constructor in order.\n    init_as_variable_flags\n        dictates whether the corresponding input argument passed to the constructor\n        should be treated as an ivy.Array.\n    init_num_positional_args\n        number of input arguments that must be passed as positional arguments to the\n        constructor.\n    init_native_array_flags\n        dictates whether the corresponding input argument passed to the constructor\n        should be treated as a native array.\n    init_all_as_kwargs_np:\n        input arguments to the constructor as keyword arguments.\n    method_input_dtypes\n        data types of the input arguments to the method in order.\n    method_as_variable_flags\n        dictates whether the corresponding input argument passed to the method should\n        be treated as an ivy.Array.\n    method_num_positional_args\n        number of input arguments that must be passed as positional arguments to the\n        method.\n    method_native_array_flags\n        dictates whether the corresponding input argument passed to the method should\n        be treated as a native array.\n    method_container_flags\n        dictates whether the corresponding input argument passed to the method should\n        be treated as an ivy Container.\n    method_all_as_kwargs_np:\n        input arguments to the method as keyword arguments.\n    class_name\n        name of the class to test.\n    method_name\n        name of the method to test.\n    init_with_v\n        if the class being tested is an ivy.Module, then setting this flag as True will\n        call the constructor with the variables v passed explicitly.\n    method_with_v\n        if the class being tested is an ivy.Module, then setting this flag as True will\n        call the method with the variables v passed explicitly.\n    rtol_\n        relative tolerance value.\n    atol_\n        absolute tolerance value.\n    test_values\n        can be a bool or a string to indicate whether correctness of values should be\n        tested. If the value is `with_v`, shapes are tested but not values.\n    test_gradients\n        if True, test for the correctness of gradients.\n    xs_grad_idxs\n        Indices of the input arrays to compute gradients with respect to. If None,\n        gradients are returned with respect to all input arrays. (Default value = None)\n    ret_grad_idxs\n        Indices of the returned arrays for which to return computed gradients. If None,\n        gradients are returned for all returned arrays. (Default value = None)\n    test_trace\n        If True, test for the correctness of tracing.\n    ground_truth_backend\n        Ground Truth Backend to compare the result-values.\n    device_\n        The device on which to create arrays.\n    return_flat_np_arrays\n        If test_values is False, this flag dictates whether the original returns are\n        returned, or whether the flattened numpy arrays are returned.\n\n    Returns\n    -------\n    ret\n        optional, return value from the function\n    ret_gt\n        optional, return value from the Ground Truth function\n    '
    if mod_backend[backend_to_test]:
        (proc, input_queue, output_queue) = mod_backend[backend_to_test]
        input_queue.put(('method_backend_computation', init_input_dtypes, init_flags, backend_to_test, init_all_as_kwargs_np, on_device, method_input_dtypes, method_flags, method_all_as_kwargs_np, class_name, method_name, init_with_v, test_trace, method_with_v))
        (ret, ret_np_flat, ret_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, v_np, fw_list) = output_queue.get()
    else:
        (ret, ret_np_flat, ret_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, v_np, fw_list) = test_method_backend_computation(init_input_dtypes, init_flags, backend_to_test, init_all_as_kwargs_np, on_device, method_input_dtypes, method_flags, method_all_as_kwargs_np, class_name, method_name, init_with_v, test_trace, method_with_v)
    if mod_backend[ground_truth_backend]:
        (proc, input_queue, output_queue) = mod_backend[ground_truth_backend]
        input_queue.put(('method_ground_truth_computation', ground_truth_backend, on_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, method_input_dtypes, method_flags, class_name, method_name, test_trace, v_np))
        (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, fw_list2) = output_queue.get()
    else:
        (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, fw_list2) = test_method_ground_truth_computation(ground_truth_backend, on_device, org_con_data, args_np_method, met_arg_np_vals, met_args_idxs, kwargs_np_method, met_kwarg_np_vals, met_kwargs_idxs, method_input_dtypes, method_flags, class_name, method_name, test_trace, v_np)
    for (k, v) in fw_list2.items():
        if k not in fw_list:
            fw_list[k] = []
        fw_list[k].extend(v)
    assert ret_device == ret_from_gt_device, f'ground truth backend ({ground_truth_backend}) returned array on device {ret_from_gt_device} but target backend ({backend_to_test}) returned array on device {ret_device}'
    if ret_device is not None:
        assert ret_device == on_device, f'device is set to {on_device}, but ground truth produced array on {ret_device}'
    if not test_values:
        if return_flat_np_arrays:
            return (ret_np_flat, ret_np_from_gt_flat)
        return (ret, ret_from_gt)
    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, backend_to_test)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, backend_to_test)
    value_test(backend=backend_to_test, ground_truth_backend=ground_truth_backend, ret_np_flat=ret_np_flat, ret_np_from_gt_flat=ret_np_from_gt_flat, rtol=rtol_, atol=atol_, specific_tolerance_dict=tolerance_dict)

def test_frontend_method(*, init_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]]=None, method_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]], init_flags, method_flags, init_all_as_kwargs_np: Optional[dict]=None, method_all_as_kwargs_np: dict, frontend: str, frontend_method_data: FrontendMethodData, backend_to_test: str, on_device, rtol_: Optional[float]=None, atol_: float=1e-06, tolerance_dict: Optional[dict]=None, test_values: Union[bool, str]=True):
    if False:
        while True:
            i = 10
    '\n    Test a class-method that consumes (or returns) arrays for the current backend by\n    comparing the result with numpy.\n\n    Parameters\n    ----------\n    init_input_dtypes\n        data types of the input arguments to the constructor in order.\n    init_as_variable_flags\n        dictates whether the corresponding input argument passed to the constructor\n        should be treated as an ivy.Variable.\n    init_num_positional_args\n        number of input arguments that must be passed as positional arguments to the\n        constructor.\n    init_native_array_flags\n        dictates whether the corresponding input argument passed to the constructor\n        should be treated as a native array.\n    init_all_as_kwargs_np:\n        input arguments to the constructor as keyword arguments.\n    method_input_dtypes\n        data types of the input arguments to the method in order.\n    method_all_as_kwargs_np:\n        input arguments to the method as keyword arguments.\n    frontend\n        current frontend (framework).\n    rtol_\n        relative tolerance value.\n    atol_\n        absolute tolerance value.\n    tolerance_dict\n        dictionary of tolerance values for specific dtypes.\n    test_values\n        can be a bool or a string to indicate whether correctness of values should be\n        tested. If the value is `with_v`, shapes are tested but not values.\n\n    Returns\n    -------\n    ret\n        optional, return value from the function\n    ret_gt\n        optional, return value from the Ground Truth function\n    '
    _switch_backend_context(method_flags.test_trace)
    (args_np_constructor, kwargs_np_constructor) = kwargs_to_args_n_kwargs(num_positional_args=init_flags.num_positional_args, kwargs=init_all_as_kwargs_np)
    (con_arg_np_vals, con_args_idxs, con_c_arg_vals) = _get_nested_np_arrays(args_np_constructor)
    (con_kwarg_np_vals, con_kwargs_idxs, con_c_kwarg_vals) = _get_nested_np_arrays(kwargs_np_constructor)
    num_arrays_constructor = con_c_arg_vals + con_c_kwarg_vals
    if len(init_input_dtypes) < num_arrays_constructor:
        init_input_dtypes = [init_input_dtypes[0] for _ in range(num_arrays_constructor)]
    if len(init_flags.as_variable) < num_arrays_constructor:
        init_flags.as_variable = [init_flags.as_variable[0] for _ in range(num_arrays_constructor)]
    if len(init_flags.native_arrays) < num_arrays_constructor:
        init_flags.native_arrays = [init_flags.native_arrays[0] for _ in range(num_arrays_constructor)]
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        init_flags.as_variable = [v if ivy_backend.is_float_dtype(d) else False for (v, d) in zip(init_flags.as_variable, init_input_dtypes)]
    (args_constructor, kwargs_constructor) = create_args_kwargs(backend=backend_to_test, args_np=args_np_constructor, arg_np_vals=con_arg_np_vals, args_idxs=con_args_idxs, kwargs_np=kwargs_np_constructor, kwarg_np_vals=con_kwarg_np_vals, kwargs_idxs=con_kwargs_idxs, input_dtypes=init_input_dtypes, test_flags=init_flags, on_device=on_device)
    (args_np_method, kwargs_np_method) = kwargs_to_args_n_kwargs(num_positional_args=method_flags.num_positional_args, kwargs=method_all_as_kwargs_np)
    (met_arg_np_vals, met_args_idxs, met_c_arg_vals) = _get_nested_np_arrays(args_np_method)
    (met_kwarg_np_vals, met_kwargs_idxs, met_c_kwarg_vals) = _get_nested_np_arrays(kwargs_np_method)
    num_arrays_method = met_c_arg_vals + met_c_kwarg_vals
    if len(method_input_dtypes) < num_arrays_method:
        method_input_dtypes = [method_input_dtypes[0] for _ in range(num_arrays_method)]
    if len(method_flags.as_variable) < num_arrays_method:
        method_flags.as_variable = [method_flags.as_variable[0] for _ in range(num_arrays_method)]
    if len(method_flags.native_arrays) < num_arrays_method:
        method_flags.native_arrays = [method_flags.native_arrays[0] for _ in range(num_arrays_method)]
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        if frontend == 'jax':
            importlib.import_module('ivy.functional.frontends.jax').config.update('jax_enable_x64', True)
        method_flags.as_variable = [v if ivy_backend.is_float_dtype(d) else False for (v, d) in zip(method_flags.as_variable, method_input_dtypes)]
        (args_method, kwargs_method) = create_args_kwargs(backend=backend_to_test, args_np=args_np_method, arg_np_vals=met_arg_np_vals, args_idxs=met_args_idxs, kwargs_np=kwargs_np_method, kwarg_np_vals=met_kwarg_np_vals, kwargs_idxs=met_kwargs_idxs, input_dtypes=method_input_dtypes, test_flags=method_flags, on_device=on_device)
        local_importer = ivy_backend.utils.dynamic_import
        create_frontend_array = local_importer.import_module(f'ivy.functional.frontends.{frontend}')._frontend_array
        (args_constructor_ivy, kwargs_constructor_ivy) = ivy_backend.args_to_ivy(*args_constructor, **kwargs_constructor)
        (args_method_ivy, kwargs_method_ivy) = ivy_backend.args_to_ivy(*args_method, **kwargs_method)
        args_constructor_np = ivy_backend.nested_map(lambda x: ivy_backend.to_numpy(x._data) if isinstance(x, ivy_backend.Array) else x, args_constructor_ivy, shallow=False)
        kwargs_constructor_np = ivy_backend.nested_map(lambda x: ivy_backend.to_numpy(x._data) if isinstance(x, ivy_backend.Array) else x, kwargs_constructor_ivy, shallow=False)
        args_method_np = ivy_backend.nested_map(lambda x: ivy_backend.to_numpy(x._data) if isinstance(x, ivy_backend.Array) else x, args_method_ivy, shallow=False)
        kwargs_method_np = ivy_backend.nested_map(lambda x: ivy_backend.to_numpy(x._data) if isinstance(x, ivy_backend.Array) else x, kwargs_method_ivy, shallow=False)
        frontend_fw_module = ivy_backend.utils.dynamic_import.import_module(frontend_method_data.ivy_init_module)
        ivy_frontend_creation_fn = getattr(frontend_fw_module, frontend_method_data.init_name)
        ins = ivy_frontend_creation_fn(*args_constructor, **kwargs_constructor)
        if method_flags.inplace:
            copy_args_method = copy.deepcopy(args_method)
            copy_kwargs_method = copy.deepcopy(kwargs_method)
            copy_ins = ivy_frontend_creation_fn(*args_constructor, **kwargs_constructor)
            frontend_ret_ins = copy_ins.__getattribute__(frontend_method_data.method_name)(*copy_args_method, **copy_kwargs_method)
            assert frontend_ret_ins is copy_ins, f'Inplace method did not return the same instance of the frontend array, expected {copy_ins}, got {frontend_ret_ins}'
        ret = get_frontend_ret(backend_to_test, ins.__getattribute__(frontend_method_data.method_name), *args_method_ivy, frontend_array_function=create_frontend_array if method_flags.test_trace else None, test_trace=method_flags.test_trace, precision_mode=method_flags.precision_mode, **kwargs_method_ivy)
        _assert_frontend_ret(ret, for_fn=False)
        ret_np_flat = flatten_frontend_to_np(ret=ret, backend=backend_to_test)
    frontend_config = get_frontend_config(frontend)
    args_constructor_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else x, args_constructor_np, shallow=False)
    kwargs_constructor_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else x, kwargs_constructor_np, shallow=False)
    args_method_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else frontend_config.as_native_dtype(x) if isinstance(x, frontend_config.Dtype) else frontend_config.as_native_device(x) if isinstance(x, frontend_config.Device) else x, args_method_np, shallow=False)
    kwargs_method_frontend = ivy.nested_map(lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else x, kwargs_method_np, shallow=False)
    if 'dtype' in kwargs_method_frontend:
        kwargs_method_frontend['dtype'] = frontend_config.as_native_dtype(kwargs_method_frontend['dtype'])
    if 'device' in kwargs_method_frontend:
        kwargs_method_frontend['device'] = frontend_config.as_native_device(kwargs_method_frontend['device'])
    frontend_creation_fn = getattr(importlib.import_module(frontend_method_data.framework_init_module), frontend_method_data.init_name)
    ins_gt = frontend_creation_fn(*args_constructor_frontend, **kwargs_constructor_frontend)
    frontend_ret = ins_gt.__getattribute__(frontend_method_data.method_name)(*args_method_frontend, **kwargs_method_frontend)
    if frontend == 'tensorflow' and isinstance(frontend_ret, tf.TensorShape):
        frontend_ret_np_flat = [np.asarray(frontend_ret, dtype=np.int32)]
    else:
        frontend_ret_np_flat = flatten_frontend_fw_to_np(frontend_ret, frontend_config.isscalar, frontend_config.is_native_array, frontend_config.to_numpy)
    if not test_values:
        return (ret, frontend_ret)
    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, backend_to_test)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, backend_to_test)
    value_test(ret_np_flat=ret_np_flat, ret_np_from_gt_flat=frontend_ret_np_flat, rtol=rtol_, atol=atol_, specific_tolerance_dict=tolerance_dict, backend=backend_to_test, ground_truth_backend=frontend)
DEFAULT_RTOL = None
DEFAULT_ATOL = 1e-06

def _get_framework_rtol(rtols: dict, current_fw: str):
    if False:
        print('Hello World!')
    if current_fw in rtols:
        return rtols[current_fw]
    return DEFAULT_RTOL

def _get_framework_atol(atols: dict, current_fw: str):
    if False:
        return 10
    if current_fw in atols:
        return atols[current_fw]
    return DEFAULT_ATOL

def _get_nested_np_arrays(nest):
    if False:
        for i in range(10):
            print('nop')
    '\n    Search for a NumPy arrays in a nest.\n\n    Parameters\n    ----------\n    nest\n        nest to search in.\n\n    Returns\n    -------\n         Items found, indices, and total number of arrays found\n    '
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))
    ret = ivy.multi_index_nest(nest, indices)
    return (ret, indices, len(ret))

def create_args_kwargs(*, backend: str, args_np, arg_np_vals, args_idxs, kwargs_np, kwarg_np_vals, kwargs_idxs, input_dtypes, test_flags: Union[pf.FunctionTestFlags, pf.MethodTestFlags], on_device):
    if False:
        while True:
            i = 10
    '\n    Create arguments and keyword-arguments for the function to test.\n\n    Parameters\n    ----------\n    args_np\n        A dictionary of arguments in Numpy.\n    kwargs_np\n        A dictionary of keyword-arguments in Numpy.\n    input_dtypes\n        data-types of the input arguments and keyword-arguments.\n\n    Returns\n    -------\n    Backend specific arguments, keyword-arguments\n    '
    with BackendHandler.update_backend(backend) as ivy_backend:
        args = ivy_backend.copy_nest(args_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(args, args_idxs, test_flags.apply_flags(arg_np_vals, input_dtypes, 0, backend=backend, on_device=on_device))
        kwargs = ivy_backend.copy_nest(kwargs_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(kwargs, kwargs_idxs, test_flags.apply_flags(kwarg_np_vals, input_dtypes, len(arg_np_vals), backend=backend, on_device=on_device))
    return (args, kwargs)

def convtrue(argument):
    if False:
        while True:
            i = 10
    'Convert NativeClass in argument to true framework counter part.'
    if isinstance(argument, NativeClass):
        return argument._native_class
    return argument

def wrap_frontend_function_args(argument):
    if False:
        return 10
    'Wrap frontend function arguments to return native arrays.'
    with BackendHandler.update_backend(t_globals.CURRENT_FRONTEND_STR) as ivy_frontend:
        if ivy_frontend.nested_any(argument, lambda x: hasattr(x, '__module__') and x.__module__.startswith('ivy.functional.frontends')):
            return ivy_frontend.output_to_native_arrays(ivy_frontend.frontend_outputs_to_ivy_arrays(argument))
    if ivy_frontend.nested_any(argument, lambda x: isinstance(x, ivy_frontend.Shape)):
        return argument.shape
    return argument

def kwargs_to_args_n_kwargs(*, num_positional_args, kwargs):
    if False:
        while True:
            i = 10
    '\n    Split the kwargs into args and kwargs.\n\n    The first num_positional_args ported to args.\n    '
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return (args, kwargs)

def flatten(*, backend: str, ret):
    if False:
        i = 10
        return i + 15
    'Return a flattened numpy version of the arrays in ret.'
    if not isinstance(ret, tuple):
        ret = (ret,)
    with BackendHandler.update_backend(backend) as ivy_backend:
        ret_idxs = ivy_backend.nested_argwhere(ret, ivy_backend.is_ivy_array)
        if len(ret_idxs) == 0:
            ret_idxs = ivy_backend.nested_argwhere(ret, ivy_backend.isscalar)
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
            ret_flat = [ivy_backend.asarray(x, dtype=ivy_backend.Dtype(str(np.asarray(x).dtype))) for x in ret_flat]
        else:
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
    return ret_flat

def flatten_frontend(*, ret, backend: str, frontend_array_fn=None):
    if False:
        i = 10
        return i + 15
    'Return a flattened version of the frontend arrays in ret.'
    if not isinstance(ret, tuple):
        ret = (ret,)
    with BackendHandler.update_backend(backend) as ivy_backend:
        ret_idxs = ivy_backend.nested_argwhere(ret, _is_frontend_array)
        if len(ret_idxs) == 0:
            ret_idxs = ivy_backend.nested_argwhere(ret, ivy_backend.isscalar)
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
            ret_flat = [frontend_array_fn(x) for x in ret_flat]
        else:
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
    return ret_flat

def flatten_frontend_fw_to_np(frontend_ret, isscalar_func, is_native_array_func, to_numpy_func):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(frontend_ret, tuple):
        frontend_ret = (frontend_ret,)
    frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, is_native_array_func)
    if len(frontend_ret_idxs) == 0:
        frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, isscalar_func)
        frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
    else:
        frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
    return [to_numpy_func(x) for x in frontend_ret_flat]

def flatten_and_to_np(*, backend: str, ret):
    if False:
        i = 10
        return i + 15
    ret_flat = flatten(backend=backend, ret=ret)
    with BackendHandler.update_backend(backend) as ivy_backend:
        ret = [ivy_backend.to_numpy(x) for x in ret_flat]
    return ret

def flatten_frontend_to_np(*, backend: str, ret):
    if False:
        i = 10
        return i + 15
    if not isinstance(ret, tuple):
        ret = (ret,)
    with BackendHandler.update_backend(backend) as ivy_backend:
        ret_idxs = ivy_backend.nested_argwhere(ret, _is_frontend_array)
        if len(ret_idxs) == 0:
            ret_idxs = ivy_backend.nested_argwhere(ret, ivy_backend.isscalar)
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
            return [ivy_backend.to_numpy(x) for x in ret_flat]
        else:
            ret_flat = ivy_backend.multi_index_nest(ret, ret_idxs)
            return [ivy_backend.to_numpy(x.ivy_array) for x in ret_flat]

def get_ret_and_flattened_np_array(backend_to_test: str, fn, *args, test_trace=False, precision_mode=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run func with args and kwargs.\n\n    Return the result along with its flattened version.\n    '
    fn = traced_if_required(backend_to_test, fn, test_trace=test_trace, args=args, kwargs=kwargs)
    with BackendHandler.update_backend(backend_to_test) as ivy_backend:
        with ivy_backend.PreciseMode(precision_mode):
            ret = fn(*args, **kwargs)

        def map_fn(x):
            if False:
                print('Hello World!')
            if _is_frontend_array(x):
                return x.ivy_array
            elif ivy_backend.is_native_array(x) or isinstance(x, np.ndarray):
                return ivy_backend.to_ivy(x)
            return x
        ret = ivy_backend.nested_map(map_fn, ret, include_derived={'tuple': True})
        return (ret, flatten_and_to_np(backend=backend_to_test, ret=ret))

def get_frontend_ret(backend, frontend_fn, *args, frontend_array_function=None, precision_mode=False, test_trace: bool=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    frontend_fn = traced_if_required(backend, frontend_fn, test_trace=test_trace, args=args, kwargs=kwargs)
    with BackendHandler.update_backend(backend) as ivy_backend:
        if test_trace:
            (args, kwargs) = ivy_backend.nested_map(_frontend_array_to_ivy, (args, kwargs), include_derived={'tuple': True})
        with ivy_backend.PreciseMode(precision_mode):
            ret = frontend_fn(*args, **kwargs)
        if test_trace:
            assert frontend_array_function is not None
            ret = ivy_backend.nested_map(arrays_to_frontend(backend, frontend_array_function), ret, include_derived={'tuple': True})
    return ret

def _get_transpiled_data_if_required(frontend_fn, frontend_fw_fn, frontend, backend, fn_name, generate_frontend_arrays, args_for_test, kwargs_for_test, frontend_fw_args, frontend_fw_kwargs):
    if False:
        i = 10
        return i + 15
    iterations = 1
    with BackendHandler.update_backend(backend) as ivy_backend:
        if generate_frontend_arrays:
            (args_for_test, kwargs_for_test) = ivy.nested_map(_frontend_array_to_ivy, (args_for_test, kwargs_for_test), include_derived={'tuple': True})
        else:
            (args_for_test, kwargs_for_test) = ivy_backend.args_to_ivy(*args_for_test, **kwargs_for_test)
    traced_fn = traced_if_required(backend, frontend_fn, test_trace=True, args=args_for_test, kwargs=kwargs_for_test)
    frontend_timings = []
    frontend_fw_timings = []
    for i in range(0, iterations):
        start = time.time()
        traced_fn(*args_for_test, **kwargs_for_test)
        end = time.time()
        frontend_timings.append(end - start)
        start = time.time()
        frontend_fw_fn(*frontend_fw_args, **frontend_fw_kwargs)
        end = time.time()
        frontend_fw_timings.append(end - start)
    with BackendHandler.update_backend(backend) as ivy_backend:
        traced_fn_to_ivy = ivy_backend.trace_graph(frontend_fn, to='ivy', args=args_for_test, kwargs=kwargs_for_test)
    frontend_time = np.mean(frontend_timings).item()
    frontend_fw_time = np.mean(frontend_fw_timings).item()
    backend_nodes = len(traced_fn._functions)
    ivy_nodes = len(traced_fn_to_ivy._functions)
    data = {'frontend': frontend, 'frontend_func': fn_name, 'args': str(args_for_test), 'kwargs': str(kwargs_for_test), 'time': frontend_time, 'fw_time': frontend_fw_time, 'nodes': backend_nodes, 'ivy_nodes': ivy_nodes}
    _create_transpile_report(data, backend, 'report.json')

def args_to_container(array_args):
    if False:
        while True:
            i = 10
    array_args_container = ivy.Container({str(k): v for (k, v) in enumerate(array_args)})
    return array_args_container

def as_lists(*args):
    if False:
        print('Hello World!')
    'Change the elements in args to be of type list.'
    return (a if isinstance(a, list) else [a] for a in args)

def gradient_incompatible_function(*, fn):
    if False:
        print('Hello World!')
    return not ivy.supports_gradients and hasattr(fn, 'computes_gradients') and fn.computes_gradients

def gradient_unsupported_dtypes(*, fn):
    if False:
        while True:
            i = 10
    visited = set()
    to_visit = [fn]
    (out, res) = ({}, {})
    while to_visit:
        fn = to_visit.pop()
        if fn in visited:
            continue
        visited.add(fn)
        unsupported_grads = fn.unsupported_gradients if hasattr(fn, 'unsupported_gradients') else {}
        for (k, v) in unsupported_grads.items():
            if k not in out:
                out[k] = []
            out[k].extend(v)
        if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            continue
        fl = _get_function_list(fn)
        res = _get_functions_from_string(fl, __import__(fn.__module__))
        to_visit.extend(res)
    return out

def _is_frontend_array(x):
    if False:
        i = 10
        return i + 15
    return hasattr(x, 'ivy_array')

def _frontend_array_to_ivy(x):
    if False:
        i = 10
        return i + 15
    if _is_frontend_array(x):
        return x.ivy_array
    else:
        return x

def args_to_frontend(backend: str, *args, frontend_array_fn=None, include_derived=None, **kwargs):
    if False:
        print('Hello World!')
    with BackendHandler.update_backend(backend) as ivy_backend:
        frontend_args = ivy_backend.nested_map(arrays_to_frontend(backend=backend, frontend_array_fn=frontend_array_fn), args, include_derived, shallow=False)
        frontend_kwargs = ivy_backend.nested_map(arrays_to_frontend(backend=backend, frontend_array_fn=frontend_array_fn), kwargs, include_derived, shallow=False)
        return (frontend_args, frontend_kwargs)

def arrays_to_frontend(backend: str, frontend_array_fn):
    if False:
        print('Hello World!')
    with BackendHandler.update_backend(backend) as ivy_backend:

        def _new_fn(x):
            if False:
                print('Hello World!')
            if _is_frontend_array(x):
                return x
            elif ivy_backend.is_array(x):
                if tuple(x.shape) == ():
                    try:
                        ret = frontend_array_fn(x, dtype=ivy_backend.Dtype(str(x.dtype)))
                    except ivy_backend.utils.exceptions.IvyException:
                        ret = frontend_array_fn(x, dtype=ivy_backend.array(x).dtype)
                else:
                    ret = frontend_array_fn(x)
                return ret
            return x
    return _new_fn

def _switch_backend_context(trace: bool):
    if False:
        i = 10
        return i + 15
    if trace:
        BackendHandler._update_context(BackendHandlerMode.SetBackend)
    else:
        BackendHandler._update_context(BackendHandlerMode.WithBackend) if BackendHandler._ctx_flag else None