import sys
import time
import numpy as np
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import get_error_message
from ray.cluster_utils import Cluster, cluster_not_supported

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
@pytest.fixture(params=[1, 4])
def ray_start_reconstruction(request):
    if False:
        print('Hello World!')
    num_nodes = request.param
    plasma_store_memory = int(0.5 * 10 ** 9)
    cluster = Cluster(initialize_head=True, head_node_args={'num_cpus': 1, 'object_store_memory': plasma_store_memory // num_nodes, 'redis_max_memory': 10 ** 8, '_system_config': {'object_timeout_milliseconds': 200}})
    for i in range(num_nodes - 1):
        cluster.add_node(num_cpus=1, object_store_memory=plasma_store_memory // num_nodes)
    ray.init(address=cluster.address)
    yield (plasma_store_memory, num_nodes, cluster)
    ray.shutdown()
    cluster.shutdown()

@pytest.mark.skip(reason='Failing with new GCS API on Linux.')
def test_simple(ray_start_reconstruction):
    if False:
        while True:
            i = 10
    (plasma_store_memory, num_nodes, cluster) = ray_start_reconstruction
    num_objects = 100
    size = int(plasma_store_memory * 1.5 / (num_objects * 8))

    @ray.remote
    def foo(i, size):
        if False:
            for i in range(10):
                print('nop')
        array = np.zeros(size)
        array[0] = i
        return array
    args = []
    for i in range(num_objects):
        args.append(foo.remote(i, size))
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    num_chunks = 4 * num_nodes
    chunk = num_objects // num_chunks
    for i in range(num_chunks):
        values = ray.get(args[i * chunk:(i + 1) * chunk])
        del values
    assert cluster.remaining_processes_alive()

def sorted_random_indexes(total, output_num):
    if False:
        print('Hello World!')
    random_indexes = [np.random.randint(total) for _ in range(output_num)]
    random_indexes.sort()
    return random_indexes

@pytest.mark.skip(reason='Failing with new GCS API on Linux.')
def test_recursive(ray_start_reconstruction):
    if False:
        i = 10
        return i + 15
    (plasma_store_memory, num_nodes, cluster) = ray_start_reconstruction
    num_objects = 100
    size = int(plasma_store_memory * 1.5 / (num_objects * 8))

    @ray.remote
    def no_dependency_task(size):
        if False:
            print('Hello World!')
        array = np.zeros(size)
        return array

    @ray.remote
    def single_dependency(i, arg):
        if False:
            for i in range(10):
                print('nop')
        arg = np.copy(arg)
        arg[0] = i
        return arg
    arg = no_dependency_task.remote(size)
    args = []
    for i in range(num_objects):
        arg = single_dependency.remote(i, arg)
        args.append(arg)
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    random_indexes = sorted_random_indexes(num_objects, 10)
    for i in random_indexes:
        value = ray.get(args[i])
        assert value[0] == i
    num_chunks = 4 * num_nodes
    chunk = num_objects // num_chunks
    for i in range(num_chunks):
        values = ray.get(args[i * chunk:(i + 1) * chunk])
        del values
    assert cluster.remaining_processes_alive()

@pytest.mark.skip(reason='This test often hangs or fails in CI.')
def test_multiple_recursive(ray_start_reconstruction):
    if False:
        print('Hello World!')
    (plasma_store_memory, _, cluster) = ray_start_reconstruction
    num_objects = 100
    size = plasma_store_memory * 2 // (num_objects * 8)

    @ray.remote
    def no_dependency_task(size):
        if False:
            for i in range(10):
                print('nop')
        array = np.zeros(size)
        return array

    @ray.remote
    def multiple_dependency(i, arg1, arg2, arg3):
        if False:
            return 10
        arg1 = np.copy(arg1)
        arg1[0] = i
        return arg1
    num_args = 3
    args = []
    for i in range(num_args):
        arg = no_dependency_task.remote(size)
        args.append(arg)
    for i in range(num_objects):
        args.append(multiple_dependency.remote(i, *args[i:i + num_args]))
    args = args[num_args:]
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    random_indexes = sorted_random_indexes(num_objects, 10)
    for i in random_indexes:
        value = ray.get(args[i])
        assert value[0] == i
    assert cluster.remaining_processes_alive()

def wait_for_errors(p, error_check):
    if False:
        while True:
            i = 10
    errors = []
    time_left = 100
    while time_left > 0:
        errors.extend(get_error_message(p, 1))
        if error_check(errors):
            break
        time_left -= 1
        time.sleep(1)
    assert error_check(errors)
    return errors

@pytest.mark.skip('This test does not work yet.')
def test_nondeterministic_task(ray_start_reconstruction, error_pubsub):
    if False:
        for i in range(10):
            print('nop')
    p = error_pubsub
    (plasma_store_memory, num_nodes, cluster) = ray_start_reconstruction
    num_objects = 1000
    size = plasma_store_memory * 2 // (num_objects * 8)

    @ray.remote
    def foo(i, size):
        if False:
            for i in range(10):
                print('nop')
        array = np.random.rand(size)
        array[0] = i
        return array

    @ray.remote
    def bar(i, size):
        if False:
            print('Hello World!')
        array = np.zeros(size)
        array[0] = i
        return array
    args = []
    for i in range(num_objects):
        if i % 2 == 0:
            args.append(foo.remote(i, size))
        else:
            args.append(bar.remote(i, size))
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i

    def error_check(errors):
        if False:
            i = 10
            return i + 15
        if num_nodes == 1:
            min_errors = num_objects // 2
        else:
            min_errors = 1
        return len(errors) >= min_errors
    errors = wait_for_errors(p, error_check)
    assert all((error.type == ray_constants.HASH_MISMATCH_PUSH_ERROR for error in errors))
    assert cluster.remaining_processes_alive()

@pytest.mark.skip(reason='Failing with new GCS API on Linux.')
@pytest.mark.parametrize('ray_start_object_store_memory', [10 ** 9], indirect=True)
def test_driver_put_errors(ray_start_object_store_memory, error_pubsub):
    if False:
        print('Hello World!')
    p = error_pubsub
    plasma_store_memory = ray_start_object_store_memory
    num_objects = 100
    size = plasma_store_memory * 2 // (num_objects * 8)

    @ray.remote
    def single_dependency(i, arg):
        if False:
            print('Hello World!')
        arg = np.copy(arg)
        arg[0] = i
        return arg
    args = []
    arg = single_dependency.remote(0, np.zeros(size))
    for i in range(num_objects):
        arg = single_dependency.remote(i, arg)
        args.append(arg)
    for i in range(num_objects):
        value = ray.get(args[i])
        assert value[0] == i
    ray.wait([args[0]], timeout=30)

    def error_check(errors):
        if False:
            for i in range(10):
                print('nop')
        return len(errors) > 1
    errors = wait_for_errors(p, error_check)
    assert all((error.type == ray_constants.PUT_RECONSTRUCTION_PUSH_ERROR or 'ray.exceptions.ObjectLostError' in error.error_messages for error in errors))
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))