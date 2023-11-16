import json
import os
import pytest
import importlib
import inspect
import functools
from typing import List, Optional
from hypothesis import given, strategies as st
import ivy.functional.frontends.numpy as np_frontend
from .hypothesis_helpers import number_helpers as nh
from .globals import TestData
from . import test_parameter_flags as pf
from . import test_globals as t_globals
from .pipeline_helper import BackendHandler
from ivy_tests.test_ivy.helpers.test_parameter_flags import DynamicFlag, BuiltInstanceStrategy, BuiltAsVariableStrategy, BuiltNativeArrayStrategy, BuiltGradientStrategy, BuiltContainerStrategy, BuiltWithOutStrategy, BuiltWithCopyStrategy, BuiltInplaceStrategy, BuiltTraceStrategy, BuiltFrontendArrayStrategy, BuiltTranspileStrategy, BuiltPrecisionModeStrategy
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import _dtype_kind_keys, _get_type_dict
from .globals import mod_backend
cmd_line_args = ('with_out', 'instance_method', 'test_gradients', 'test_trace', 'precision_mode')
cmd_line_args_lists = ('as_variable', 'native_array', 'container')

def _get_runtime_flag_value(flag):
    if False:
        i = 10
        return i + 15
    return flag.strategy if isinstance(flag, DynamicFlag) else flag

@st.composite
def num_positional_args_method(draw, *, method):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draws an integers randomly from the minimum and maximum number of positional\n    arguments a given method can take.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    method\n        callable method\n\n    Returns\n    -------\n    A strategy that can be used in the @given hypothesis decorator.\n    '
    (total, num_positional_only, num_keyword_only) = (0, 0, 0)
    for param in inspect.signature(method).parameters.values():
        if param.name == 'self':
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind in [param.KEYWORD_ONLY, param.VAR_KEYWORD]:
            num_keyword_only += 1
    return draw(nh.ints(min_value=num_positional_only, max_value=total - num_keyword_only))

@st.composite
def num_positional_args(draw, *, fn_name: Optional[str]=None):
    if False:
        print('Hello World!')
    '\n    Draws an integers randomly from the minimum and maximum number of positional\n    arguments a given function can take.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a\n        given data-set (ex. list).\n    fn_name\n        name of the function.\n\n    Returns\n    -------\n    A strategy that can be used in the @given hypothesis decorator.\n\n    Examples\n    --------\n    @given(\n        num_positional_args=num_positional_args(fn_name="floor_divide")\n    )\n    @given(\n        num_positional_args=num_positional_args(fn_name="add")\n    )\n    '
    if mod_backend[t_globals.CURRENT_BACKEND]:
        (proc, input_queue, output_queue) = mod_backend[t_globals.CURRENT_BACKEND]
        input_queue.put(('num_positional_args_helper', fn_name, t_globals.CURRENT_BACKEND))
        (num_positional_only, total, num_keyword_only) = output_queue.get()
    else:
        (num_positional_only, total, num_keyword_only) = num_positional_args_helper(fn_name, t_globals.CURRENT_BACKEND)
    return draw(nh.ints(min_value=num_positional_only, max_value=total - num_keyword_only))

def num_positional_args_helper(fn_name, backend):
    if False:
        print('Hello World!')
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    with BackendHandler.update_backend(backend) as ivy_backend:
        ivy_backend.utils.dynamic_import.import_module(fn_name.rpartition('.')[0])
        for (i, fn_name_key) in enumerate(fn_name.split('.')):
            if i == 0:
                fn = ivy_backend.__dict__[fn_name_key]
            else:
                fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == 'self':
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind in [param.KEYWORD_ONLY, param.VAR_KEYWORD]:
            num_keyword_only += 1
    return (num_positional_only, total, num_keyword_only)

def _import_fn(fn_tree: str):
    if False:
        while True:
            i = 10
    '\n    Import a function from function tree string.\n\n    Parameters\n    ----------\n    fn_tree\n        Full function tree without "ivy" root\n        example: "functional.backends.jax.creation.arange".\n\n    Returns\n    -------\n    Returns fn_name, imported module, callable function\n    '
    split_index = fn_tree.rfind('.')
    fn_name = fn_tree[split_index + 1:]
    module_to_import = fn_tree[:split_index]
    mod = importlib.import_module(module_to_import)
    try:
        callable_fn = mod.__dict__[fn_name]
    except KeyError:
        raise ImportError(f"Error: The function '{fn_name}' could not be found within the module '{module_to_import}'.\nPlease double-check the function name and its associated path.\nIf this function is a new feature you'd like to see, we'd love to hear from you! You can contribute to our project. For more details, please visit:\nhttps://lets-unify.ai/ivy/contributing/open_tasks.html\n")
    return (callable_fn, fn_name, module_to_import)

def _get_method_supported_devices_dtypes_helper(method_name: str, class_module: str, class_name: str, backend_str: str):
    if False:
        for i in range(10):
            print('nop')
    with BackendHandler.update_backend(backend_str) as backend:
        _fn = getattr(class_module.__dict__[class_name], method_name)
        devices_and_dtypes = backend.function_supported_devices_and_dtypes(_fn)
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(backend_str, devices_and_dtypes[device])
    return organized_dtypes

def _get_method_supported_devices_dtypes(method_name: str, class_module: str, class_name: str):
    if False:
        return 10
    '\n    Get supported devices and data types for a method in Ivy API.\n\n    Parameters\n    ----------\n    method_name\n        Name of the method in the class\n\n    class_module\n        Name of the class module\n\n    class_name\n        Name of the class\n\n    Returns\n    -------\n    Returns a dictionary containing supported device types and its supported data types\n    for the method\n    '
    supported_device_dtypes = {}
    for backend_str in available_frameworks:
        if mod_backend[backend_str]:
            (proc, input_queue, output_queue) = mod_backend[backend_str]
            input_queue.put(('method supported dtypes', method_name, class_module.__name__, class_name, backend_str))
            supported_device_dtypes[backend_str] = output_queue.get()
        else:
            supported_device_dtypes[backend_str] = _get_method_supported_devices_dtypes_helper(method_name, class_module, class_name, backend_str)
    return supported_device_dtypes

def _get_supported_devices_dtypes_helper(backend_str: str, fn_module: str, fn_name: str):
    if False:
        while True:
            i = 10
    with BackendHandler.update_backend(backend_str) as backend:
        _tmp_mod = importlib.import_module(fn_module)
        _fn = _tmp_mod.__dict__[fn_name]
        devices_and_dtypes = backend.function_supported_devices_and_dtypes(_fn)
        try:
            if 'bfloat16' in devices_and_dtypes['gpu']:
                tmp = list(devices_and_dtypes['gpu'])
                tmp.remove('bfloat16')
                devices_and_dtypes['gpu'] = tuple(tmp)
        except KeyError:
            pass
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(backend_str, devices_and_dtypes[device])
    return organized_dtypes

def _get_supported_devices_dtypes(fn_name: str, fn_module: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get supported devices and data types for a function in Ivy API.\n\n    Parameters\n    ----------\n    fn_name\n        Name of the function\n\n    fn_module\n        Full import path of the function module\n\n    Returns\n    -------\n    Returns a dictionary containing supported device types and its supported data types\n    for the function\n    '
    supported_device_dtypes = {}
    if fn_module == 'ivy.functional.frontends.numpy':
        fn_module_ = np_frontend
        if isinstance(getattr(fn_module_, fn_name), fn_module_.ufunc):
            fn_name = f'_{fn_name}'
    for backend_str in available_frameworks:
        if mod_backend[backend_str]:
            (proc, input_queue, output_queue) = mod_backend[backend_str]
            input_queue.put(('supported dtypes', fn_module, fn_name, backend_str))
            supported_device_dtypes[backend_str] = output_queue.get()
        else:
            supported_device_dtypes[backend_str] = _get_supported_devices_dtypes_helper(backend_str, fn_module, fn_name)
    return supported_device_dtypes

def _partition_dtypes_into_kinds(framework: str, dtypes):
    if False:
        i = 10
        return i + 15
    partitioned_dtypes = {}
    for kind in _dtype_kind_keys:
        partitioned_dtypes[kind] = set(_get_type_dict(framework, kind)).intersection(dtypes)
    return partitioned_dtypes

def handle_test(*, fn_tree: Optional[str]=None, ground_truth_backend: str='tensorflow', number_positional_args=None, test_instance_method=BuiltInstanceStrategy, test_with_out=BuiltWithOutStrategy, test_with_copy=BuiltWithCopyStrategy, test_gradients=BuiltGradientStrategy, test_trace=BuiltTraceStrategy, transpile=BuiltTranspileStrategy, precision_mode=BuiltPrecisionModeStrategy, as_variable_flags=BuiltAsVariableStrategy, native_array_flags=BuiltNativeArrayStrategy, container_flags=BuiltContainerStrategy, **_given_kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test wrapper for Ivy functions.\n\n    The wrapper sets the required test globals and creates test flags strategies.\n\n    Parameters\n    ----------\n    fn_tree\n        Full function import path\n\n    ground_truth_backend\n        The framework to assert test results are equal to\n\n    number_positional_args\n        A search strategy for determining the number of positional arguments to be\n        passed to the function\n\n    test_instance_method\n        A search strategy that generates a boolean to test instance methods\n\n    test_with_out\n        A search strategy that generates a boolean to test the function with an `out`\n        parameter\n\n    test_with_copy\n        A search strategy that generates a boolean to test the function with an `copy`\n        parameter\n\n    test_gradients\n        A search strategy that generates a boolean to test the function with arrays as\n        gradients\n\n    test_trace\n        A search strategy that generates a boolean to trace and test the\n        function\n\n    precision_mode\n        A search strategy that generates a boolean to switch between two different\n        precision modes supported by numpy and (torch, jax) and test the function\n\n    as_variable_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a Variable array\n\n    native_array_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a native array\n\n    container_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a Container\n    '
    is_fn_tree_provided = fn_tree is not None
    if is_fn_tree_provided:
        fn_tree = f'ivy.{fn_tree}'
    is_hypothesis_test = len(_given_kwargs) != 0
    possible_arguments = {}
    if is_hypothesis_test and is_fn_tree_provided:
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        possible_arguments['test_flags'] = pf.function_flags(ground_truth_backend=st.just(ground_truth_backend), num_positional_args=number_positional_args, instance_method=_get_runtime_flag_value(test_instance_method), with_out=_get_runtime_flag_value(test_with_out), with_copy=_get_runtime_flag_value(test_with_copy), test_gradients=_get_runtime_flag_value(test_gradients), test_trace=_get_runtime_flag_value(test_trace), transpile=_get_runtime_flag_value(transpile), as_variable=_get_runtime_flag_value(as_variable_flags), native_arrays=_get_runtime_flag_value(native_array_flags), container_flags=_get_runtime_flag_value(container_flags), precision_mode=_get_runtime_flag_value(precision_mode))

    def test_wrapper(test_fn):
        if False:
            print('Hello World!')
        if is_fn_tree_provided:
            (callable_fn, fn_name, fn_mod) = _import_fn(fn_tree)
            supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
            possible_arguments['fn_name'] = st.just(fn_name)
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    if e.__class__.__qualname__ == 'IvyNotImplementedException':
                        pytest.skip('Function not implemented in backend.')
                    else:
                        raise e
        else:
            wrapped_test = test_fn
        if is_fn_tree_provided:
            wrapped_test.test_data = TestData(test_fn=wrapped_test, fn_tree=fn_tree, fn_name=fn_name, supported_device_dtypes=supported_device_dtypes)
        wrapped_test._ivy_test = True
        wrapped_test.ground_truth_backend = ground_truth_backend
        return wrapped_test
    return test_wrapper

def handle_frontend_test(*, fn_tree: str, gt_fn_tree: Optional[str]=None, aliases: Optional[List[str]]=None, number_positional_args=None, test_with_out=BuiltWithOutStrategy, test_with_copy=BuiltWithCopyStrategy, test_inplace=BuiltInplaceStrategy, as_variable_flags=BuiltAsVariableStrategy, native_array_flags=BuiltNativeArrayStrategy, test_trace=BuiltTraceStrategy, generate_frontend_arrays=BuiltFrontendArrayStrategy, transpile=BuiltTranspileStrategy, precision_mode=BuiltPrecisionModeStrategy, **_given_kwargs):
    if False:
        print('Hello World!')
    '\n    Test wrapper for Ivy frontend functions.\n\n    The wrapper sets the required test globals and creates test flags strategies.\n\n    Parameters\n    ----------\n    fn_tree\n        Full function import path\n    gt_fn_tree\n        Full function import path for the ground truth function, by default will be\n        the same as fn_tree\n    number_positional_args\n        A search strategy for determining the number of positional arguments to be\n        passed to the function\n\n    test_inplace\n        A search strategy that generates a boolean to test the method with `inplace`\n        update\n\n    test_with_out\n        A search strategy that generates a boolean to test the function with an `out`\n        parameter\n\n    test_with_copy\n        A search strategy that generates a boolean to test the function with an `copy`\n        parameter\n\n    precision_mode\n        A search strategy that generates a boolean to switch between two different\n        precision modes supported by numpy and (torch, jax) and test the function\n\n    as_variable_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a Variable array\n\n    native_array_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a native array\n\n    test_trace\n        A search strategy that generates a boolean to trace and test the\n        function\n\n    generate_frontend_arrays\n        A search strategy that generates a list of boolean flags for array inputs to\n        be frontend array\n    '
    fn_tree = f'ivy.functional.frontends.{fn_tree}'
    if aliases is not None:
        for i in range(len(aliases)):
            aliases[i] = f'ivy.functional.frontends.{aliases[i]}'
    is_hypothesis_test = len(_given_kwargs) != 0
    if is_hypothesis_test:
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        test_flags = pf.frontend_function_flags(num_positional_args=number_positional_args, with_out=_get_runtime_flag_value(test_with_out), with_copy=_get_runtime_flag_value(test_with_copy), inplace=_get_runtime_flag_value(test_inplace), as_variable=_get_runtime_flag_value(as_variable_flags), native_arrays=_get_runtime_flag_value(native_array_flags), test_trace=_get_runtime_flag_value(test_trace), generate_frontend_arrays=_get_runtime_flag_value(generate_frontend_arrays), transpile=_get_runtime_flag_value(transpile), precision_mode=_get_runtime_flag_value(precision_mode))

    def test_wrapper(test_fn):
        if False:
            print('Hello World!')
        (callable_fn, fn_name, fn_mod) = _import_fn(fn_tree)
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            possible_arguments = {'test_flags': test_flags, 'fn_tree': st.sampled_from([fn_tree] + aliases) if aliases is not None else st.just(fn_tree), 'gt_fn_tree': st.just(gt_fn_tree)}
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    if e.__class__.__qualname__ == 'IvyNotImplementedException':
                        pytest.skip('Function not implemented in backend.')
                    else:
                        raise e
        else:
            wrapped_test = test_fn
        wrapped_test.test_data = TestData(test_fn=wrapped_test, fn_tree=fn_tree, fn_name=fn_name, supported_device_dtypes=supported_device_dtypes)
        return wrapped_test
    return test_wrapper

def _import_method(method_tree: str):
    if False:
        for i in range(10):
            print('nop')
    split_index = method_tree.rfind('.')
    (class_tree, method_name) = (method_tree[:split_index], method_tree[split_index + 1:])
    split_index = class_tree.rfind('.')
    (mod_to_import, class_name) = (class_tree[:split_index], class_tree[split_index + 1:])
    _mod = importlib.import_module(mod_to_import)
    _class = _mod.__getattribute__(class_name)
    _method = getattr(_class, method_name)
    return (_method, method_name, _class, class_name, _mod)

def handle_method(*, init_tree: str='', method_tree: Optional[str]=None, ground_truth_backend: str='tensorflow', test_gradients=BuiltGradientStrategy, test_trace=BuiltTraceStrategy, precision_mode=BuiltPrecisionModeStrategy, init_num_positional_args=None, init_native_arrays=BuiltNativeArrayStrategy, init_as_variable_flags=BuiltAsVariableStrategy, method_num_positional_args=None, method_native_arrays=BuiltNativeArrayStrategy, method_as_variable_flags=BuiltAsVariableStrategy, method_container_flags=BuiltContainerStrategy, **_given_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Test wrapper for Ivy methods.\n\n    The wrapper sets the required test globals and creates test flags strategies.\n\n    Parameters\n    ----------\n    method_tree\n        Full method import path\n\n    ground_truth_backend\n        The framework to assert test results are equal to\n    '
    is_method_tree_provided = method_tree is not None
    if is_method_tree_provided:
        method_tree = f'ivy.{method_tree}'
    is_hypothesis_test = len(_given_kwargs) != 0
    possible_arguments = {'ground_truth_backend': st.just(ground_truth_backend), 'test_gradients': _get_runtime_flag_value(test_gradients), 'test_trace': _get_runtime_flag_value(test_trace), 'precision_mode': _get_runtime_flag_value(precision_mode)}
    if is_hypothesis_test and is_method_tree_provided:
        (callable_method, method_name, _, class_name, method_mod) = _import_method(method_tree)
        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(fn_name=init_tree)
        possible_arguments['init_flags'] = pf.init_method_flags(num_positional_args=init_num_positional_args, as_variable=_get_runtime_flag_value(init_as_variable_flags), native_arrays=_get_runtime_flag_value(init_native_arrays), precision_mode=_get_runtime_flag_value(precision_mode))
        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(method=callable_method)
        possible_arguments['method_flags'] = pf.method_flags(num_positional_args=method_num_positional_args, as_variable=_get_runtime_flag_value(method_as_variable_flags), native_arrays=_get_runtime_flag_value(method_native_arrays), container_flags=_get_runtime_flag_value(method_container_flags), precision_mode=_get_runtime_flag_value(precision_mode))

    def test_wrapper(test_fn):
        if False:
            while True:
                i = 10
        if is_method_tree_provided:
            supported_device_dtypes = _get_method_supported_devices_dtypes(method_name, method_mod, class_name)
            possible_arguments['class_name'] = st.just(class_name)
            possible_arguments['method_name'] = st.just(method_name)
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                if False:
                    return 10
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    if e.__class__.__qualname__ == 'IvyNotImplementedException':
                        pytest.skip('Function not implemented in backend.')
                    else:
                        raise e
        else:
            wrapped_test = test_fn
        wrapped_test.test_data = TestData(test_fn=wrapped_test, fn_tree=method_tree, fn_name=method_name, supported_device_dtypes=supported_device_dtypes, is_method=True)
        wrapped_test.ground_truth_backend = ground_truth_backend
        wrapped_test._ivy_test = True
        return wrapped_test
    return test_wrapper

def handle_frontend_method(*, class_tree: str, init_tree: str, method_name: str, init_num_positional_args=None, init_native_arrays=BuiltNativeArrayStrategy, init_as_variable_flags=BuiltAsVariableStrategy, test_trace=BuiltTraceStrategy, precision_mode=BuiltPrecisionModeStrategy, method_num_positional_args=None, method_native_arrays=BuiltNativeArrayStrategy, method_as_variable_flags=BuiltAsVariableStrategy, test_inplace=BuiltInplaceStrategy, generate_frontend_arrays=BuiltFrontendArrayStrategy, **_given_kwargs):
    if False:
        print('Hello World!')
    '\n    Test wrapper for Ivy frontends methods.\n\n    The wrapper sets the required test globals and creates\n    test flags strategies.\n\n    Parameters\n    ----------\n    class_tree\n        Full class import path\n\n    init_tree\n        Full import path for the function used to create the class\n\n    method_name\n        Name of the method\n\n    init_num_positional_args\n        A search strategy that generates a number of positional arguments\n        to be passed during instantiation of the class\n\n    init_native_arrays\n        A search strategy that generates a boolean to test the method with native\n        arrays\n\n    init_as_variable_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a Variable array\n\n    test_compile\n        A search strategy that generates a boolean to graph compile and test the\n        function\n\n    precision_mode\n        A search strategy that generates a boolean to switch between two different\n        precision modes supported by numpy and (torch, jax) and test the function\n\n    method_num_positional_args\n        A search strategy that generates a number of positional arguments\n        to be passed during call of the class method\n\n    method_native_arrays\n        A search strategy that generates a boolean to test the method with native\n        arrays\n\n    method_as_variable_flags\n        A search strategy that generates a list of boolean flags for array inputs to be\n        passed as a Variable array\n\n    test_inplace\n        A search strategy that generates a boolean to test the method with `inplace`\n        update\n    '
    split_index = init_tree.rfind('.')
    framework_init_module = init_tree[:split_index]
    ivy_init_module = f'ivy.functional.frontends.{init_tree[:split_index]}'
    init_name = init_tree[split_index + 1:]
    init_tree = f'ivy.functional.frontends.{init_tree}'
    is_hypothesis_test = len(_given_kwargs) != 0
    split_index = class_tree.rfind('.')
    (class_module_path, class_name) = (class_tree[:split_index], class_tree[split_index + 1:])
    class_module = importlib.import_module(class_module_path)
    method_class = getattr(class_module, class_name)
    if is_hypothesis_test:
        callable_method = getattr(method_class, method_name)
        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(fn_name=init_tree)
        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(method=callable_method)

    def test_wrapper(test_fn):
        if False:
            print('Hello World!')
        supported_device_dtypes = _get_method_supported_devices_dtypes(method_name, class_module, class_name)
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            init_flags = pf.frontend_init_flags(num_positional_args=init_num_positional_args, as_variable=_get_runtime_flag_value(init_as_variable_flags), native_arrays=_get_runtime_flag_value(init_native_arrays))
            method_flags = pf.frontend_method_flags(num_positional_args=method_num_positional_args, inplace=_get_runtime_flag_value(test_inplace), as_variable=_get_runtime_flag_value(method_as_variable_flags), native_arrays=_get_runtime_flag_value(method_native_arrays), test_trace=_get_runtime_flag_value(test_trace), precision_mode=_get_runtime_flag_value(precision_mode), generate_frontend_arrays=_get_runtime_flag_value(generate_frontend_arrays))
            ivy_init_modules = str(ivy_init_module)
            framework_init_modules = str(framework_init_module)
            frontend_helper_data = FrontendMethodData(ivy_init_module=ivy_init_modules, framework_init_module=framework_init_modules, init_name=init_name, method_name=method_name)
            possible_arguments = {'init_flags': init_flags, 'method_flags': method_flags, 'frontend_method_data': st.just(frontend_helper_data)}
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                if False:
                    print('Hello World!')
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    if e.__class__.__qualname__ == 'IvyNotImplementedException':
                        pytest.skip('Function not implemented in backend.')
                    else:
                        raise e
        else:
            wrapped_test = test_fn
        wrapped_test.test_data = TestData(test_fn=wrapped_test, fn_tree=f'{init_tree}.{method_name}', fn_name=method_name, supported_device_dtypes=supported_device_dtypes, is_method=[method_name, class_tree, split_index])
        return wrapped_test
    return test_wrapper

@st.composite
def seed(draw):
    if False:
        while True:
            i = 10
    return draw(st.integers(min_value=0, max_value=2 ** 8 - 1))

def _create_transpile_report(data: dict, backend: str, file_name: str, is_backend: bool=False):
    if False:
        while True:
            i = 10
    backend_specific_data = ['nodes', 'time', 'args', 'kwargs']
    if os.path.isfile(file_name):
        with open(file_name, 'r') as outfile:
            file_data = json.load(outfile)
            if file_data['nodes'].get(backend, 0) > data['nodes']:
                return
            for key in backend_specific_data:
                file_data[key][backend] = data[key]
            if not is_backend:
                for key in ['ivy_nodes', 'fw_time']:
                    file_data[key] = data[key]
            json_object = json.dumps(file_data, indent=6)
            with open(file_name, 'w') as outfile:
                outfile.write(json_object)
            return
    for key in backend_specific_data:
        data[key] = {backend: data[key]}
    json_object = json.dumps(data, indent=6)
    with open(file_name, 'w') as outfile:
        outfile.write(json_object)