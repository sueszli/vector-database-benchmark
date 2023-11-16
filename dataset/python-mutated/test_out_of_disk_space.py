import os
import platform
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
import numpy as np
import pytest
import ray
from ray.util.state import list_cluster_events

def calculate_capacity_threshold(disk_capacity_in_bytes):
    if False:
        while True:
            i = 10
    usage = shutil.disk_usage('/tmp')
    threshold = min(1, 1.0 - 1.0 * (usage.free - disk_capacity_in_bytes) / usage.total)
    return threshold

def get_current_usage():
    if False:
        return 10
    usage = shutil.disk_usage('/tmp')
    print(f'free: {usage.free} ')
    print(f'current usage: {1.0 - 1.0 * usage.free / usage.total}')
    return 1.0 - 1.0 * usage.free / usage.total

@contextmanager
def create_tmp_file(bytes):
    if False:
        print('Hello World!')
    tmp_dir = tempfile.mkdtemp(dir='/tmp')
    tmp_path = os.path.join(tmp_dir, 'test.txt')
    with open(tmp_path, 'wb') as f:
        f.write(os.urandom(bytes))
    try:
        yield tmp_path
    finally:
        os.remove(tmp_path)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_put_out_of_disk(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    local_fs_capacity_threshold = calculate_capacity_threshold(200 * 1024 * 1024)
    ray.init(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': local_fs_capacity_threshold, 'local_fs_monitor_interval_ms': 10})
    assert get_current_usage() < local_fs_capacity_threshold
    ref = ray.put(np.random.rand(20 * 1024 * 1024))
    del ref
    with create_tmp_file(250 * 1024 * 1024):
        assert get_current_usage() > local_fs_capacity_threshold
        time.sleep(1)
        with pytest.raises(ray.exceptions.OutOfDiskError):
            ray.put(np.random.rand(20 * 1024 * 1024))
    assert get_current_usage() < local_fs_capacity_threshold
    time.sleep(1)
    ray.put(np.random.rand(20 * 1024 * 1024))

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_task_returns(shutdown_only):
    if False:
        print('Hello World!')
    local_fs_capacity_threshold = calculate_capacity_threshold(10 * 1024 * 1024)
    ray.init(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': local_fs_capacity_threshold, 'local_fs_monitor_interval_ms': 10})
    with create_tmp_file(250 * 1024 * 1024):
        assert get_current_usage() > local_fs_capacity_threshold
        time.sleep(1)

        @ray.remote
        def foo():
            if False:
                return 10
            time.sleep(1)
            return np.random.rand(20 * 1024 * 1024)
        try:
            ray.get(foo.remote())
        except ray.exceptions.RayTaskError as e:
            assert isinstance(e.cause, ray.exceptions.OutOfDiskError)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_task_put(shutdown_only):
    if False:
        print('Hello World!')
    local_fs_capacity_threshold = calculate_capacity_threshold(1 * 1024 * 1024)
    ray.init(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': local_fs_capacity_threshold, 'local_fs_monitor_interval_ms': 10})
    with create_tmp_file(250 * 1024 * 1024):
        assert get_current_usage() > local_fs_capacity_threshold
        time.sleep(1)

        @ray.remote
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            ref = ray.put(np.random.rand(20 * 1024 * 1024))
            return ref
        try:
            ray.get(foo.remote())
        except ray.exceptions.RayTaskError as e:
            assert isinstance(e.cause, ray.exceptions.OutOfDiskError)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_task_args(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': 0}, resources={'out_of_memory': 1})
    cluster.add_node(num_cpus=1, object_store_memory=200 * 1024 * 1024, resources={'sufficient_memory': 1})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote
    def foo():
        if False:
            i = 10
            return i + 15
        return np.random.rand(20 * 1024 * 1024)

    @ray.remote
    def bar(obj):
        if False:
            for i in range(10):
                print('nop')
        print(obj)
    ref = foo.options(resources={'sufficient_memory': 1}).remote()
    try:
        ray.get(bar.options(resources={'out_of_memory': 1}).remote(ref))
    except ray.exceptions.RayTaskError as e:
        assert isinstance(e.cause, ray.exceptions.OutOfDiskError)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_actor(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': 0}, resources={'out_of_memory': 1})
    cluster.add_node(num_cpus=1, object_store_memory=200 * 1024 * 1024, resources={'sufficient_memory': 1})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote
    def foo():
        if False:
            print('Hello World!')
        return np.random.rand(20 * 1024 * 1024)

    @ray.remote
    class Actor:

        def __init__(self, obj):
            if False:
                while True:
                    i = 10
            self._obj = obj

        def foo(self):
            if False:
                for i in range(10):
                    print('nop')
            print(self._obj)

        def args_ood(self, obj):
            if False:
                i = 10
                return i + 15
            print(obj)

        def return_ood(self):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.rand(20 * 1024 * 1024)
    ref = foo.options(resources={'sufficient_memory': 1}).remote()
    with pytest.raises(ray.exceptions.RayActorError):
        a = Actor.options(resources={'out_of_memory': 0.001}).remote(ref)
        ray.get(a.foo.remote())
    a = Actor.options(resources={'out_of_memory': 1}).remote(1)
    ray.get(a.foo.remote())
    try:
        ray.get(a.args_ood.remote(ref))
    except ray.exceptions.RayTaskError as e:
        assert isinstance(e.cause, ray.exceptions.OutOfDiskError)
    ray.get(a.foo.remote())
    try:
        ray.get(a.return_ood.remote())
    except ray.exceptions.RayTaskError as e:
        assert isinstance(e.cause, ray.exceptions.OutOfDiskError)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Not targeting Windows')
def test_ood_events(shutdown_only):
    if False:
        return 10
    local_fs_capacity_threshold = calculate_capacity_threshold(200 * 1024 * 1024)
    ray.init(num_cpus=1, object_store_memory=80 * 1024 * 1024, _system_config={'local_fs_capacity_threshold': local_fs_capacity_threshold, 'local_fs_monitor_interval_ms': 10})
    with create_tmp_file(250 * 1024 * 1024):
        assert get_current_usage() > local_fs_capacity_threshold
        time.sleep(1)

        @ray.remote
        def foo():
            if False:
                print('Hello World!')
            ref = ray.put(np.random.rand(20 * 1024 * 1024))
            return ref
        try:
            ray.get(foo.remote())
        except ray.exceptions.RayTaskError as e:
            assert isinstance(e.cause, ray.exceptions.OutOfDiskError)
    events = list_cluster_events()
    print(events)
    assert len(events) >= 1
    assert 'Object creation will fail if spilling is required' in events[0]['message']
    assert events[0]['severity'] == 'ERROR'
if __name__ == '__main__':
    sys.exit(pytest.main(['-sv', __file__]))