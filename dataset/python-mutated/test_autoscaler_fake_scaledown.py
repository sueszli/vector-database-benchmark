import platform
import re
import numpy as np
import pytest
import ray
from ray._private.test_utils import wait_for_condition
from ray.cluster_utils import AutoscalingCluster

@ray.remote(num_cpus=1)
class Actor:

    def __init__(self):
        if False:
            print('Hello World!')
        self.data = []

    def f(self):
        if False:
            while True:
                i = 10
        pass

    def recv(self, obj):
        if False:
            for i in range(10):
                print('nop')
        pass

    def create(self, size):
        if False:
            for i in range(10):
                print('nop')
        return np.zeros(size)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Failing on Windows.')
def test_scaledown_shared_objects(shutdown_only):
    if False:
        return 10
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'cpu_node': {'resources': {'CPU': 1, 'object_store_memory': 100 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 5}}, idle_timeout_minutes=0.05)
    try:
        cluster.start(_system_config={'scheduler_report_pinned_bytes_only': True})
        ray.init('auto')
        actors = [Actor.remote() for _ in range(5)]
        ray.get([a.f.remote() for a in actors])
        print('All five nodes launched')
        wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == 5)
        data = actors[0].create.remote(1024 * 1024 * 5)
        ray.get([a.recv.remote(data) for a in actors])
        print('Data broadcast successfully, deleting actors.')
        del actors
        wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == 1, timeout=30)
    finally:
        cluster.shutdown()

def check_memory(local_objs, num_spilled_objects=None, num_plasma_objects=None):
    if False:
        i = 10
        return i + 15

    def ok():
        if False:
            for i in range(10):
                print('nop')
        s = ray._private.internal_api.memory_summary()
        print(f'\n\nMemory Summary:\n{s}\n')
        actual_objs = re.findall('LOCAL_REFERENCE[\\s|\\|]+([0-9a-f]+)', s)
        if sorted(actual_objs) != sorted(local_objs):
            raise RuntimeError(f'Expect local objects={local_objs}, actual={actual_objs}')
        if num_spilled_objects is not None:
            m = re.search('Spilled (\\d+) MiB, (\\d+) objects', s)
            if m is not None:
                actual_spilled_objects = int(m.group(2))
                if actual_spilled_objects < num_spilled_objects:
                    raise RuntimeError(f'Expected spilled objects={num_spilled_objects} greater than actual={actual_spilled_objects}')
        if num_plasma_objects is not None:
            m = re.search('Plasma memory usage (\\d+) MiB, (\\d+) objects', s)
            if m is None:
                raise RuntimeError('Memory summary does not contain Plasma memory objects count')
            actual_plasma_objects = int(m.group(2))
            if actual_plasma_objects != num_plasma_objects:
                raise RuntimeError(f'Expected plasma objects={num_plasma_objects} not equal to actual={actual_plasma_objects}')
        return True
    wait_for_condition(ok, timeout=30, retry_interval_ms=5000)

@pytest.mark.skipif(platform.system() == 'Windows', reason='Failing on Windows.')
def test_no_scaledown_with_spilled_objects(shutdown_only):
    if False:
        return 10
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'cpu_node': {'resources': {'CPU': 1, 'object_store_memory': 75 * 1024 * 1024}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}}, idle_timeout_minutes=0.05)
    try:
        cluster.start(_system_config={'scheduler_report_pinned_bytes_only': True, 'min_spilling_size': 0})
        ray.init('auto')
        actors = [Actor.remote() for _ in range(2)]
        ray.get([a.f.remote() for a in actors])
        wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == 2)
        print('All nodes launched')
        obj_size = 10 * 1024 * 1024
        objs = []
        for i in range(10):
            obj = actors[0].create.remote(obj_size)
            ray.get(actors[1].recv.remote(obj))
            objs.append(obj)
            print(f'obj {i}={obj.hex()}')
            del obj
        check_memory([obj.hex() for obj in objs], num_spilled_objects=9)
        print('Objects spilled, deleting actors and object references.')
        spilled_obj = objs[0]
        del objs
        del actors

        def scaledown_to_one():
            if False:
                return 10
            cpu = ray.cluster_resources().get('CPU', 0)
            assert cpu > 0, 'Scale-down should keep at least 1 node'
            return cpu == 1
        wait_for_condition(scaledown_to_one, timeout=30)
        check_memory([spilled_obj.hex()], num_plasma_objects=0)
        del spilled_obj
        wait_for_condition(lambda : ray.cluster_resources().get('CPU', 0) == 0)
        check_memory([], num_plasma_objects=0)
    finally:
        cluster.shutdown()
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))