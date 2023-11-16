import copy
import logging
import os
import sys
import time
import numpy as np
import pytest
import ray
import ray._private.gcs_utils as gcs_utils
import ray.cluster_utils
from ray._private.test_utils import SignalActor, convert_actor_state, kill_actor_and_wait_for_failure, put_object, wait_for_condition
logger = logging.getLogger(__name__)

@pytest.fixture
def one_worker_100MiB(request):
    if False:
        i = 10
        return i + 15
    config = {'task_retry_delay_ms': 0, 'automatic_object_spilling_enabled': False}
    yield ray.init(num_cpus=1, object_store_memory=100 * 1024 * 1024, _system_config=config)
    ray.shutdown()

def _fill_object_store_and_get(obj, succeed=True, object_MiB=20, num_objects=5):
    if False:
        for i in range(10):
            print('nop')
    for _ in range(num_objects):
        ray.put(np.zeros(object_MiB * 1024 * 1024, dtype=np.uint8))
    if type(obj) is bytes:
        obj = ray.ObjectRef(obj)
    if succeed:
        wait_for_condition(lambda : ray._private.worker.global_worker.core_worker.object_exists(obj))
    else:
        wait_for_condition(lambda : not ray._private.worker.global_worker.core_worker.object_exists(obj), timeout=30)

def _check_refcounts(expected):
    if False:
        return 10
    actual = ray._private.worker.global_worker.core_worker.get_all_reference_counts()
    assert len(expected) == len(actual)
    for (object_ref, (local, submitted)) in expected.items():
        hex_id = object_ref.hex().encode('ascii')
        assert hex_id in actual
        assert local == actual[hex_id]['local']
        assert submitted == actual[hex_id]['submitted']

def check_refcounts(expected, timeout=10):
    if False:
        return 10
    start = time.time()
    while True:
        try:
            _check_refcounts(expected)
            break
        except AssertionError as e:
            if time.time() - start > timeout:
                raise e
            else:
                time.sleep(0.1)

def test_local_refcounts(ray_start_regular):
    if False:
        while True:
            i = 10
    obj_ref1 = ray.put(None)
    check_refcounts({obj_ref1: (1, 0)})
    obj_ref1_copy = copy.copy(obj_ref1)
    check_refcounts({obj_ref1: (2, 0)})
    del obj_ref1
    check_refcounts({obj_ref1_copy: (1, 0)})
    del obj_ref1_copy
    check_refcounts({})

def test_dependency_refcounts(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote
    def one_dep(dep, signal=None, fail=False):
        if False:
            for i in range(10):
                print('nop')
        if signal is not None:
            ray.get(signal.wait.remote())
        if fail:
            raise Exception('failed on purpose')

    @ray.remote
    def one_dep_large(dep, signal=None):
        if False:
            i = 10
            return i + 15
        if signal is not None:
            ray.get(signal.wait.remote())
        return np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    signal = SignalActor.remote()
    large_dep = ray.put(np.zeros(10 * 1024 * 1024, dtype=np.uint8))
    result = one_dep.remote(large_dep, signal=signal)
    check_refcounts({large_dep: (1, 1), result: (1, 0)})
    ray.get(signal.send.remote())
    check_refcounts({large_dep: (1, 0), result: (1, 0)})
    del large_dep, result
    check_refcounts({})
    signal = SignalActor.remote()
    dep = one_dep.remote(None, signal=signal)
    check_refcounts({dep: (1, 0)})
    result = one_dep.remote(dep)
    check_refcounts({dep: (1, 1), result: (1, 0)})
    ray.get(signal.send.remote())
    check_refcounts({dep: (1, 0), result: (1, 0)})
    del dep, result
    check_refcounts({})
    (signal1, signal2) = (SignalActor.remote(), SignalActor.remote())
    dep = one_dep_large.remote(None, signal=signal1)
    check_refcounts({dep: (1, 0)})
    result = one_dep.remote(dep, signal=signal2)
    check_refcounts({dep: (1, 1), result: (1, 0)})
    ray.get(signal1.send.remote())
    ray.get(dep, timeout=10)
    check_refcounts({dep: (1, 1), result: (1, 0)})
    ray.get(signal2.send.remote())
    check_refcounts({dep: (1, 0), result: (1, 0)})
    del dep, result
    check_refcounts({})
    signal = SignalActor.remote()
    large_dep = ray.put(np.zeros(10 * 1024 * 1024, dtype=np.uint8))
    result = one_dep.remote(large_dep, signal=signal, fail=True)
    check_refcounts({large_dep: (1, 1), result: (1, 0)})
    ray.get(signal.send.remote())
    check_refcounts({large_dep: (1, 0), result: (1, 0)})
    del large_dep, result
    check_refcounts({})
    (signal1, signal2) = (SignalActor.remote(), SignalActor.remote())
    dep = one_dep_large.remote(None, signal=signal1)
    check_refcounts({dep: (1, 0)})
    result = one_dep.remote(dep, signal=signal2, fail=True)
    check_refcounts({dep: (1, 1), result: (1, 0)})
    ray.get(signal1.send.remote())
    ray.get(dep, timeout=10)
    check_refcounts({dep: (1, 1), result: (1, 0)})
    ray.get(signal2.send.remote())
    check_refcounts({dep: (1, 0), result: (1, 0)})
    del dep, result
    check_refcounts({})

def test_basic_pinning(one_worker_100MiB):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def f(array):
        if False:
            return 10
        return np.sum(array)

    @ray.remote
    class Actor(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.large_object = ray.put(np.zeros(25 * 1024 * 1024, dtype=np.uint8))

        def get_large_object(self):
            if False:
                print('Hello World!')
            return ray.get(self.large_object)
    actor = Actor.remote()
    for batch in range(10):
        intermediate_result = f.remote(np.zeros(10 * 1024 * 1024, dtype=np.uint8))
        ray.get(intermediate_result)
    ray.get(actor.get_large_object.remote())

def test_pending_task_dependency_pinning(one_worker_100MiB):
    if False:
        print('Hello World!')

    @ray.remote
    def pending(input1, input2):
        if False:
            while True:
                i = 10
        return
    np_array = np.zeros(20 * 1024 * 1024, dtype=np.uint8)
    signal = SignalActor.remote()
    obj_ref = pending.remote(np_array, signal.wait.remote())
    for _ in range(2):
        ray.put(np.zeros(20 * 1024 * 1024, dtype=np.uint8))
    ray.get(signal.send.remote())
    ray.get(obj_ref)

def test_feature_flag(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    def f(array):
        if False:
            while True:
                i = 10
        return np.sum(array)

    @ray.remote
    class Actor(object):

        def __init__(self):
            if False:
                return 10
            self.large_object = ray.put(np.zeros(25 * 1024 * 1024, dtype=np.uint8))

        def wait_for_actor_to_start(self):
            if False:
                return 10
            pass

        def get_large_object(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray.get(self.large_object)
    actor = Actor.remote()
    ray.get(actor.wait_for_actor_to_start.remote())
    ref = actor.get_large_object.remote()
    ray.get(ref)
    for _ in range(5):
        put_ref = ray.put(np.zeros(40 * 1024 * 1024, dtype=np.uint8))
    del put_ref
    wait_for_condition(lambda : not ray._private.worker.global_worker.core_worker.object_exists(ref))

def test_out_of_band_serialized_object_ref(one_worker_100MiB):
    if False:
        while True:
            i = 10
    assert len(ray._private.worker.global_worker.core_worker.get_all_reference_counts()) == 0
    obj_ref = ray.put('hello')
    _check_refcounts({obj_ref: (1, 0)})
    obj_ref_str = ray.cloudpickle.dumps(obj_ref)
    _check_refcounts({obj_ref: (2, 0)})
    del obj_ref
    assert len(ray._private.worker.global_worker.core_worker.get_all_reference_counts()) == 1
    assert ray.get(ray.cloudpickle.loads(obj_ref_str)) == 'hello'

def test_captured_object_ref(one_worker_100MiB):
    if False:
        print('Hello World!')
    captured_id = ray.put(np.zeros(10 * 1024 * 1024, dtype=np.uint8))

    @ray.remote
    def f(signal):
        if False:
            i = 10
            return i + 15
        ray.get(signal.wait.remote())
        ray.get(captured_id)
    signal = SignalActor.remote()
    obj_ref = f.remote(signal)
    del f
    del captured_id
    ray.get(signal.send.remote())
    _fill_object_store_and_get(obj_ref)
    captured_id = ray.put(np.zeros(10 * 1024 * 1024, dtype=np.uint8))

    @ray.remote
    class Actor:

        def get(self, signal):
            if False:
                print('Hello World!')
            ray.get(signal.wait.remote())
            ray.get(captured_id)
    signal = SignalActor.remote()
    actor = Actor.remote()
    obj_ref = actor.get.remote(signal)
    del Actor
    del captured_id
    ray.get(signal.send.remote())
    _fill_object_store_and_get(obj_ref)

@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_basic_serialized_reference(one_worker_100MiB, use_ray_put, failure):
    if False:
        while True:
            i = 10

    @ray.remote(max_retries=1)
    def pending(ref, dep):
        if False:
            return 10
        ray.get(ref[0])
        if failure:
            os._exit(0)
    array_oid = put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)
    signal = SignalActor.remote()
    obj_ref = pending.remote([array_oid], signal.wait.remote())
    array_oid_bytes = array_oid.binary()
    del array_oid
    _fill_object_store_and_get(array_oid_bytes)
    ray.get(signal.send.remote())
    try:
        ray.get(obj_ref)
        assert not failure
    except ray.exceptions.WorkerCrashedError:
        assert failure
    _fill_object_store_and_get(array_oid_bytes, succeed=False)

@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_recursive_serialized_reference(one_worker_100MiB, use_ray_put, failure):
    if False:
        print('Hello World!')

    @ray.remote(max_retries=1)
    def recursive(ref, signal, max_depth, depth=0):
        if False:
            return 10
        ray.get(ref[0])
        if depth == max_depth:
            ray.get(signal.wait.remote())
            if failure:
                os._exit(0)
            return
        else:
            return recursive.remote(ref, signal, max_depth, depth + 1)
    signal = SignalActor.remote()
    max_depth = 5
    array_oid = put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)
    head_oid = recursive.remote([array_oid], signal, max_depth)
    array_oid_bytes = array_oid.binary()
    del array_oid
    tail_oid = head_oid
    for _ in range(max_depth):
        tail_oid = ray.get(tail_oid)
    _fill_object_store_and_get(array_oid_bytes)
    ray.get(signal.send.remote())
    try:
        assert ray.get(tail_oid) is None
        assert not failure
    except ray.exceptions.OwnerDiedError:
        assert failure
    _fill_object_store_and_get(array_oid_bytes, succeed=False)

@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_actor_holding_serialized_reference(one_worker_100MiB, use_ray_put, failure):
    if False:
        while True:
            i = 10

    @ray.remote
    class GreedyActor(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

        def set_ref1(self, ref):
            if False:
                return 10
            self.ref1 = ref

        def add_ref2(self, new_ref):
            if False:
                for i in range(10):
                    print('nop')
            self.ref2 = new_ref

        def delete_ref1(self):
            if False:
                print('Hello World!')
            self.ref1 = None

        def delete_ref2(self):
            if False:
                i = 10
                return i + 15
            self.ref2 = None
    array_oid = put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)
    actor = GreedyActor.remote()
    actor.set_ref1.remote([array_oid])
    ray.get(actor.add_ref2.remote([array_oid]))
    array_oid_bytes = array_oid.binary()
    del array_oid
    _fill_object_store_and_get(array_oid_bytes)
    ray.get(actor.delete_ref1.remote())
    _fill_object_store_and_get(array_oid_bytes)
    if failure:
        kill_actor_and_wait_for_failure(actor)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.delete_ref1.remote())
    else:
        ray.get(actor.delete_ref2.remote())
    _fill_object_store_and_get(array_oid_bytes, succeed=False)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_worker_holding_serialized_reference(one_worker_100MiB, use_ray_put, failure):
    if False:
        i = 10
        return i + 15

    @ray.remote(max_retries=1)
    def child(dep1, dep2):
        if False:
            while True:
                i = 10
        if failure:
            os._exit(0)
        return

    @ray.remote
    class Submitter:

        def __init__(self):
            if False:
                return 10
            pass

        def launch_pending_task(self, ref, signal):
            if False:
                return 10
            return child.remote(ref[0], signal.wait.remote())
    signal = SignalActor.remote()
    array_oid = put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)
    s = Submitter.remote()
    child_return_id = ray.get(s.launch_pending_task.remote([array_oid], signal))
    array_oid_bytes = array_oid.binary()
    del array_oid
    _fill_object_store_and_get(array_oid_bytes)
    ray.get(signal.send.remote())
    try:
        ray.get(child_return_id)
        assert not failure
    except ray.exceptions.WorkerCrashedError:
        assert failure
    del child_return_id
    _fill_object_store_and_get(array_oid_bytes, succeed=False)

def test_basic_nested_ids(one_worker_100MiB):
    if False:
        while True:
            i = 10
    inner_oid = ray.put(np.zeros(20 * 1024 * 1024, dtype=np.uint8))
    outer_oid = ray.put([inner_oid])
    inner_oid_bytes = inner_oid.binary()
    del inner_oid
    _fill_object_store_and_get(inner_oid_bytes)
    del outer_oid
    _fill_object_store_and_get(inner_oid_bytes, succeed=False)

def _all_actors_dead():
    if False:
        i = 10
        return i + 15
    return all((actor['State'] == convert_actor_state(gcs_utils.ActorTableData.DEAD) for actor in list(ray._private.state.actors().values())))

def test_kill_actor_immediately_after_creation(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class A:
        pass
    a = A.remote()
    b = A.remote()
    ray.kill(a)
    ray.kill(b)
    wait_for_condition(_all_actors_dead, timeout=10)

def test_remove_actor_immediately_after_creation(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class A:
        pass
    a = A.remote()
    b = A.remote()
    del a
    del b
    wait_for_condition(_all_actors_dead, timeout=10)

def test_actor_constructor_borrow_cancellation(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote(resources={'nonexistent_resource': 1})
    class Actor:

        def __init__(self, obj_containing_ref):
            if False:
                i = 10
                return i + 15
            raise ValueError('The actor constructor should not be reached; the actor creation task should be cancelled before the actor is scheduled.')

        def should_not_be_run(self):
            if False:
                print('Hello World!')
            raise ValueError('This method should never be reached.')

    def test_implicit_cancel():
        if False:
            for i in range(10):
                print('nop')
        ref = ray.put(1)
        Actor.remote({'foo': ref})
    test_implicit_cancel()
    check_refcounts({})
    ref = ray.put(1)
    a = Actor.remote({'foo': ref})
    ray.kill(a)
    del ref
    check_refcounts({})
    with pytest.raises(ray.exceptions.RayActorError, match='it was killed by `ray.kill') as exc_info:
        ray.get(a.should_not_be_run.remote())
    print(exc_info._excinfo[1])
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))