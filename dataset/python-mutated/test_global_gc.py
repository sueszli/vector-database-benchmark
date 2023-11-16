import gc
import logging
import weakref
import numpy as np
import pytest
import ray
import ray.cluster_utils
from ray._private.internal_api import global_gc
from ray._private.test_utils import wait_for_condition
logger = logging.getLogger(__name__)

def test_auto_local_gc(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2, _system_config={'local_gc_interval_s': 10, 'local_gc_min_interval_s': 5, 'global_gc_min_interval_s': 10})

    class ObjectWithCyclicRef:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.loop = self

    @ray.remote(num_cpus=1)
    class GarbageHolder:

        def __init__(self):
            if False:
                return 10
            gc.disable()
            x = ObjectWithCyclicRef()
            self.garbage = weakref.ref(x)

        def has_garbage(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.garbage() is not None
    try:
        gc.disable()
        local_ref = weakref.ref(ObjectWithCyclicRef())
        actors = [GarbageHolder.remote() for _ in range(2)]
        assert local_ref() is not None
        assert all(ray.get([a.has_garbage.remote() for a in actors]))

        def check_refs_gced():
            if False:
                print('Hello World!')
            return local_ref() is None and (not any(ray.get([a.has_garbage.remote() for a in actors])))
        wait_for_condition(check_refs_gced)
    finally:
        gc.enable()

@pytest.mark.xfail(ray.cluster_utils.cluster_not_supported, reason='cluster not supported')
def test_global_gc(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray.cluster_utils.Cluster()
    cluster.add_node(num_cpus=1, num_gpus=0, _system_config={'local_gc_interval_s': 10, 'local_gc_min_interval_s': 5, 'global_gc_min_interval_s': 10})
    cluster.add_node(num_cpus=1, num_gpus=0)
    ray.init(address=cluster.address)

    class ObjectWithCyclicRef:

        def __init__(self):
            if False:
                print('Hello World!')
            self.loop = self

    @ray.remote(num_cpus=1)
    class GarbageHolder:

        def __init__(self):
            if False:
                while True:
                    i = 10
            gc.disable()
            x = ObjectWithCyclicRef()
            self.garbage = weakref.ref(x)

        def has_garbage(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.garbage() is not None
    try:
        gc.disable()
        local_ref = weakref.ref(ObjectWithCyclicRef())
        actors = [GarbageHolder.remote() for _ in range(2)]
        assert local_ref() is not None
        assert all(ray.get([a.has_garbage.remote() for a in actors]))
        global_gc()

        def check_refs_gced():
            if False:
                for i in range(10):
                    print('nop')
            return local_ref() is None and (not any(ray.get([a.has_garbage.remote() for a in actors])))
        wait_for_condition(check_refs_gced, timeout=30)
    finally:
        gc.enable()

@pytest.mark.xfail(ray.cluster_utils.cluster_not_supported, reason='cluster not supported')
def test_global_gc_when_full(shutdown_only):
    if False:
        return 10
    cluster = ray.cluster_utils.Cluster()
    for _ in range(2):
        cluster.add_node(num_cpus=1, num_gpus=0, object_store_memory=100 * 1024 * 1024)
    ray.init(address=cluster.address)

    class LargeObjectWithCyclicRef:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.loop = self
            self.large_object = ray.put(np.zeros(20 * 1024 * 1024, dtype=np.uint8))

    @ray.remote(num_cpus=1)
    class GarbageHolder:

        def __init__(self):
            if False:
                print('Hello World!')
            gc.disable()
            x = LargeObjectWithCyclicRef()
            self.garbage = weakref.ref(x)

        def has_garbage(self):
            if False:
                i = 10
                return i + 15
            return self.garbage() is not None

        def return_large_array(self):
            if False:
                print('Hello World!')
            return np.zeros(60 * 1024 * 1024, dtype=np.uint8)
    try:
        gc.disable()
        local_ref = weakref.ref(LargeObjectWithCyclicRef())
        actors = [GarbageHolder.remote() for _ in range(2)]
        assert local_ref() is not None
        assert all(ray.get([a.has_garbage.remote() for a in actors]))
        ray.put(np.zeros(80 * 1024 * 1024, dtype=np.uint8))

        def check_refs_gced():
            if False:
                while True:
                    i = 10
            return local_ref() is None and (not any(ray.get([a.has_garbage.remote() for a in actors])))
        wait_for_condition(check_refs_gced)
        local_ref = weakref.ref(LargeObjectWithCyclicRef())
        actors = [GarbageHolder.remote() for _ in range(2)]
        assert all(ray.get([a.has_garbage.remote() for a in actors]))
        ray.get(actors[0].return_large_array.remote())

        def check_refs_gced():
            if False:
                for i in range(10):
                    print('nop')
            return local_ref() is None and (not any(ray.get([a.has_garbage.remote() for a in actors])))
        wait_for_condition(check_refs_gced)
    finally:
        gc.enable()

def test_global_gc_actors(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=1, _system_config={'debug_dump_period_milliseconds': 500})
    try:
        gc.disable()

        @ray.remote(num_cpus=1)
        class A:

            def f(self):
                if False:
                    print('Hello World!')
                return 'Ok'
        for i in range(3):
            a = A.remote()
            cycle = [a]
            cycle.append(cycle)
            ray.get(a.f.remote())
            print('iteration', i)
            del a
            del cycle
    finally:
        gc.enable()
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))