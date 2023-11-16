import pytest
import sys
import ray
from ray._private.test_utils import run_string_as_driver

def test_list_named_actors_ray_kill(ray_start_regular):
    if False:
        while True:
            i = 10
    'Verify that names are returned even while actors are restarting.'

    @ray.remote(max_restarts=-1)
    class A:

        def __init__(self):
            if False:
                return 10
            pass
    a = A.options(name='hi').remote()
    assert ray.util.list_named_actors() == ['hi']
    ray.kill(a, no_restart=False)
    assert ray.util.list_named_actors() == ['hi']
    ray.kill(a, no_restart=True)
    assert not ray.util.list_named_actors()

def test_list_named_actors_detached(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    'Verify that names are returned for detached actors until killed.'
    address = ray_start_regular['address']
    driver_script = '\nimport ray\nray.init(address="{}", namespace="default_test_namespace")\n\n@ray.remote\nclass A:\n    pass\n\nA.options(name="hi", lifetime="detached").remote()\na = A.options(name="sad").remote()\n\nassert len(ray.util.list_named_actors()) == 2\nassert "hi" in ray.util.list_named_actors()\nassert "sad" in ray.util.list_named_actors()\n'.format(address)
    run_string_as_driver(driver_script)
    assert ray.util.list_named_actors() == ['hi']
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))