import ray
from ray.tests.conftest import *
from ray.tune.utils.resource_updater import _ResourceUpdater, _Resources
from unittest import mock

def test_resources_numerical_error():
    if False:
        for i in range(10):
            print('nop')
    resource = _Resources(cpu=0.99, gpu=0.99, custom_resources={'a': 0.99})
    small_resource = _Resources(cpu=0.33, gpu=0.33, custom_resources={'a': 0.33})
    for i in range(3):
        resource = _Resources.subtract(resource, small_resource)
    assert resource.is_nonnegative()

def test_resources_subtraction():
    if False:
        for i in range(10):
            print('nop')
    resource_1 = _Resources(1, 0, 0, 1, custom_resources={'a': 1, 'b': 2}, extra_custom_resources={'a': 1, 'b': 1})
    resource_2 = _Resources(1, 0, 0, 1, custom_resources={'a': 1, 'b': 2}, extra_custom_resources={'a': 1, 'b': 1})
    new_res = _Resources.subtract(resource_1, resource_2)
    assert new_res.cpu == 0
    assert new_res.gpu == 0
    assert new_res.extra_cpu == 0
    assert new_res.extra_gpu == 0
    assert all((k == 0 for k in new_res.custom_resources.values()))
    assert all((k == 0 for k in new_res.extra_custom_resources.values()))

def test_resources_different():
    if False:
        print('Hello World!')
    resource_1 = _Resources(1, 0, 0, 1, custom_resources={'a': 1, 'b': 2})
    resource_2 = _Resources(1, 0, 0, 1, custom_resources={'a': 1, 'c': 2})
    new_res = _Resources.subtract(resource_1, resource_2)
    assert 'c' in new_res.custom_resources
    assert 'b' in new_res.custom_resources
    assert new_res.cpu == 0
    assert new_res.gpu == 0
    assert new_res.extra_cpu == 0
    assert new_res.extra_gpu == 0
    assert new_res.get('a') == 0

def test_resource_updater(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    resource_updater = _ResourceUpdater(refresh_period=100)
    assert resource_updater.get_num_cpus() == 0
    assert resource_updater.get_num_gpus() == 0
    cluster.add_node(num_cpus=1, num_gpus=2)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    assert resource_updater.get_num_cpus() == 1
    assert resource_updater.get_num_gpus() == 2
    cluster.add_node(num_cpus=1, num_gpus=1)
    cluster.wait_for_nodes()
    assert resource_updater.get_num_cpus() == 1
    assert resource_updater.get_num_gpus() == 2
    resource_updater = _ResourceUpdater(refresh_period=0)
    assert resource_updater.get_num_cpus() == 2
    assert resource_updater.get_num_gpus() == 3
    cluster.add_node(num_cpus=1, num_gpus=1)
    cluster.wait_for_nodes()
    assert resource_updater.get_num_cpus() == 3
    assert resource_updater.get_num_gpus() == 4

def test_resource_updater_automatic():
    if False:
        print('Hello World!')
    "Test that resources are automatically updated when they get out of sync.\n\n    We instantiate a resource updater. When the reported resources are less than\n    what is available, we don't force an update.\n    However, if any of the resources (cpu, gpu, or custom) are higher than what\n    the updater currently think is available, we force an update from the\n    Ray cluster.\n    "
    resource_updater = _ResourceUpdater()
    resource_updater._avail_resources = _Resources(cpu=2, gpu=1, memory=1, object_store_memory=1, custom_resources={'a': 4})
    resource_updater._last_resource_refresh = 2
    with mock.patch.object(_ResourceUpdater, 'update_avail_resources', wraps=resource_updater.update_avail_resources) as upd:
        assert '2/2 CPUs' in resource_updater.debug_string(total_allocated_resources={'CPU': 2, 'GPU': 1, 'a': 4})
        assert upd.call_count == 0
        assert '4/2 CPUs' in resource_updater.debug_string(total_allocated_resources={'CPU': 4, 'GPU': 1, 'a': 0})
        assert upd.call_count == 1
        assert '8/1 GPUs' in resource_updater.debug_string(total_allocated_resources={'CPU': 2, 'GPU': 8, 'a': 0})
        assert upd.call_count == 2
        assert '6/4 a' in resource_updater.debug_string(total_allocated_resources={'CPU': 2, 'GPU': 1, 'a': 6})
        assert upd.call_count == 3
        assert '2/2 CPUs' in resource_updater.debug_string(total_allocated_resources={'CPU': 2, 'GPU': 1, 'a': 4})
        assert upd.call_count == 3
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))