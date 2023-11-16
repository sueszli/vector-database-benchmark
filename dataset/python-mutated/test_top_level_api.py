from inspect import getmembers, isfunction, ismodule
import sys
import pytest
import ray

def test_api_functions():
    if False:
        i = 10
        return i + 15
    PYTHON_API = ['init', 'is_initialized', 'remote', 'get', 'wait', 'put', 'kill', 'cancel', 'get_actor', 'get_gpu_ids', 'shutdown', 'method', 'nodes', 'timeline', 'cluster_resources', 'available_resources', 'java_function', 'java_actor_class', 'client']
    OTHER_ALLOWED_FUNCTIONS = ['get_runtime_context']
    IMPL_FUNCTIONS = ['__getattr__']
    functions = getmembers(ray, isfunction)
    function_names = [f[0] for f in functions]
    assert set(function_names) == set(PYTHON_API + OTHER_ALLOWED_FUNCTIONS + IMPL_FUNCTIONS)

def test_non_ray_modules():
    if False:
        return 10
    modules = getmembers(ray, ismodule)
    for (name, mod) in modules:
        assert 'ray' in str(mod), f'Module {mod} should not be reachable via ray.{name}'

def test_dynamic_subpackage_import():
    if False:
        print('Hello World!')
    assert 'ray.data' not in sys.modules
    ray.data
    assert 'ray.data' in sys.modules
    assert 'ray.workflow' not in sys.modules
    ray.workflow
    assert 'ray.workflow' in sys.modules
    assert 'ray.autoscaler' not in sys.modules
    ray.autoscaler
    assert 'ray.autoscaler' in sys.modules

def test_dynamic_subpackage_missing():
    if False:
        print('Hello World!')
    with pytest.raises(AttributeError):
        ray.foo
    with pytest.raises(ImportError):
        from ray.foo import bar

def test_dynamic_subpackage_fallback_only():
    if False:
        for i in range(10):
            print('nop')
    assert 'ray._raylet' in sys.modules
    assert ray.__getattribute__('_raylet') is ray._raylet
    with pytest.raises(AttributeError):
        ray.__getattr__('_raylet')

def test_for_strings():
    if False:
        print('Hello World!')
    strings = getmembers(ray, lambda obj: isinstance(obj, str))
    for (string, _) in strings:
        assert string.startswith('__'), f'Invalid string: {string} found in '
        'top level Ray. Please delete at the end of __init__.py.'
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))