import pytest
import sys
import ray
from ray._private.test_utils import run_string_as_driver

def test_list_named_actors_namespace(ray_start_regular):
    if False:
        while True:
            i = 10
    'Verify that actor names are filtered on namespace by default.'
    address = ray_start_regular['address']
    driver_script_1 = '\nimport ray\nray.init(address="{}", namespace="test")\n\n@ray.remote\nclass A:\n    pass\n\nA.options(name="hi", lifetime="detached").remote()\n\nassert len(ray.util.list_named_actors()) == 1\nassert ray.util.list_named_actors() == ["hi"]\nassert ray.util.list_named_actors(all_namespaces=True) ==     [dict(name="hi", namespace="test")]\n'.format(address)
    run_string_as_driver(driver_script_1)
    assert not ray.util.list_named_actors()
    assert ray.util.list_named_actors(all_namespaces=True) == [{'name': 'hi', 'namespace': 'test'}]
    driver_script_2 = '\nimport ray\nray.init(address="{}", namespace="test")\n\nassert ray.util.list_named_actors() == ["hi"]\nassert ray.util.list_named_actors(all_namespaces=True) ==     [dict(name="hi", namespace="test")]\nray.kill(ray.get_actor("hi"), no_restart=True)\nassert not ray.util.list_named_actors()\n'.format(address)
    run_string_as_driver(driver_script_2)
    assert not ray.util.list_named_actors()
    assert not ray.util.list_named_actors(all_namespaces=True)
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))