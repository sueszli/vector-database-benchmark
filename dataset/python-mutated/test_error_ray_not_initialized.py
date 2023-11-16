import sys
import pytest
import ray

def test_errors_before_initializing_ray(set_enable_auto_connect):
    if False:
        while True:
            i = 10

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        pass

    @ray.remote
    class Foo:
        pass
    api_methods = [f.remote, Foo.remote, lambda : ray.cancel(None), lambda : ray.get([]), lambda : ray.get_actor('name'), ray.get_gpu_ids, lambda : ray.kill(None), ray.nodes, lambda : ray.put(1), lambda : ray.wait([])]

    def test_exceptions_raised():
        if False:
            for i in range(10):
                print('nop')
        for api_method in api_methods:
            print(api_method)
            with pytest.raises(ray.exceptions.RaySystemError, match='Ray has not been started yet.'):
                api_method()
    test_exceptions_raised()
    ray.init(num_cpus=0)
    ray.shutdown()
    test_exceptions_raised()
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))