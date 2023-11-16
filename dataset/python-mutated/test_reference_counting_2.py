import logging
import os
import copy
import platform
import random
import signal
import sys
import time
import numpy as np
import pytest
import ray
import ray.cluster_utils
from ray._private.internal_api import memory_summary
from ray._private.test_utils import SignalActor, put_object, wait_for_condition, wait_for_num_actors
import ray._private.gcs_utils as gcs_utils
SIGKILL = signal.SIGKILL if sys.platform != 'win32' else signal.SIGTERM
logger = logging.getLogger(__name__)

@pytest.fixture
def one_worker_100MiB(request):
    if False:
        for i in range(10):
            print('nop')
    config = {'task_retry_delay_ms': 0, 'object_timeout_milliseconds': 1000, 'automatic_object_spilling_enabled': False}
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
        wait_for_condition(lambda : not ray._private.worker.global_worker.core_worker.object_exists(obj))

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_recursively_nest_ids(one_worker_100MiB, use_ray_put, failure):
    if False:
        print('Hello World!')

    @ray.remote(max_retries=1)
    def recursive(ref, signal, max_depth, depth=0):
        if False:
            i = 10
            return i + 15
        unwrapped = ray.get(ref[0])
        if depth == max_depth:
            ray.get(signal.wait.remote())
            if failure:
                os._exit(0)
            return
        else:
            return recursive.remote(unwrapped, signal, max_depth, depth + 1)
    signal = SignalActor.remote()
    max_depth = 5
    array_oid = put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)
    nested_oid = array_oid
    for _ in range(max_depth):
        nested_oid = ray.put([nested_oid])
    head_oid = recursive.remote([nested_oid], signal, max_depth)
    array_oid_bytes = array_oid.binary()
    del array_oid, nested_oid
    tail_oid = head_oid
    for _ in range(max_depth):
        tail_oid = ray.get(tail_oid)
    _fill_object_store_and_get(array_oid_bytes)
    ray.get(signal.send.remote())
    if not failure:
        ray.get(tail_oid)
    else:
        with pytest.raises(ray.exceptions.OwnerDiedError):
            ray.get(tail_oid)
    _fill_object_store_and_get(array_oid_bytes, succeed=False)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_return_object_ref(one_worker_100MiB, use_ray_put, failure):
    if False:
        return 10

    @ray.remote
    def return_an_id():
        if False:
            print('Hello World!')
        return [put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)]

    @ray.remote(max_retries=1)
    def exit():
        if False:
            while True:
                i = 10
        os._exit(0)
    outer_oid = return_an_id.remote()
    inner_oid_binary = ray.get(outer_oid)[0].binary()
    inner_oid = ray.get(outer_oid)[0]
    del outer_oid
    _fill_object_store_and_get(inner_oid_binary)
    if failure:
        with pytest.raises(ray.exceptions.WorkerCrashedError):
            ray.get(exit.remote())
    else:
        del inner_oid
    _fill_object_store_and_get(inner_oid_binary, succeed=False)

@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_pass_returned_object_ref(one_worker_100MiB, use_ray_put, failure):
    if False:
        print('Hello World!')

    @ray.remote
    def return_an_id():
        if False:
            while True:
                i = 10
        return [put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)]

    @ray.remote(max_retries=0)
    def pending(ref, signal):
        if False:
            return 10
        ray.get(signal.wait.remote())
        ray.get(ref[0])
        if failure:
            os._exit(0)
    signal = SignalActor.remote()
    outer_oid = return_an_id.remote()
    inner_oid_binary = ray.get(outer_oid)[0].binary()
    pending_oid = pending.remote([outer_oid], signal)
    del outer_oid
    ray.get(signal.send.remote())
    try:
        ray.get(pending_oid)
        assert not failure
    except ray.exceptions.WorkerCrashedError:
        assert failure

    def ref_not_exists():
        if False:
            i = 10
            return i + 15
        worker = ray._private.worker.global_worker
        inner_oid = ray.ObjectRef(inner_oid_binary)
        return not worker.core_worker.object_exists(inner_oid)
    wait_for_condition(ref_not_exists)

@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_recursively_pass_returned_object_ref(one_worker_100MiB, use_ray_put, failure):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def return_an_id():
        if False:
            print('Hello World!')
        return put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put)

    @ray.remote(max_retries=1)
    def recursive(ref, signal, max_depth, depth=0):
        if False:
            while True:
                i = 10
        inner_id = ray.get(ref[0])
        if depth == max_depth:
            ray.get(signal.wait.remote())
            if failure:
                os._exit(0)
            return inner_id
        else:
            return (inner_id, recursive.remote(ref, signal, max_depth, depth + 1))
    max_depth = 5
    outer_oid = return_an_id.remote()
    signal = SignalActor.remote()
    head_oid = recursive.remote([outer_oid], signal, max_depth)
    inner_oid = None
    outer_oid = head_oid
    for i in range(max_depth):
        (inner_oid, outer_oid) = ray.get(outer_oid)
    _fill_object_store_and_get(outer_oid, succeed=False)
    ray.get(signal.send.remote())
    try:
        ray.get(outer_oid)
        _fill_object_store_and_get(inner_oid)
        assert not failure
    except ray.exceptions.OwnerDiedError:
        assert failure
    inner_oid_bytes = inner_oid.binary()
    del inner_oid
    del head_oid
    del outer_oid
    _fill_object_store_and_get(inner_oid_bytes, succeed=False)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('use_ray_put,failure', [(False, False), (False, True), (True, False), (True, True)])
def test_recursively_return_borrowed_object_ref(one_worker_100MiB, use_ray_put, failure):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def recursive(num_tasks_left):
        if False:
            print('Hello World!')
        if num_tasks_left == 0:
            return (put_object(np.zeros(20 * 1024 * 1024, dtype=np.uint8), use_ray_put), os.getpid())
        return ray.get(recursive.remote(num_tasks_left - 1))
    max_depth = 5
    head_oid = recursive.remote(max_depth)
    (final_oid, owner_pid) = ray.get(head_oid)
    final_oid_bytes = final_oid.binary()
    _fill_object_store_and_get(final_oid_bytes)
    _fill_object_store_and_get(final_oid_bytes)
    if failure:
        os.kill(owner_pid, SIGKILL)
    else:
        del head_oid
        del final_oid
    _fill_object_store_and_get(final_oid_bytes, succeed=False)
    if failure:
        with pytest.raises(ray.exceptions.OwnerDiedError):
            ray.get(final_oid)

@pytest.mark.parametrize('failure', [False, True])
def test_borrowed_id_failure(one_worker_100MiB, failure):
    if False:
        return 10

    @ray.remote
    class Parent:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

        def pass_ref(self, ref, borrower):
            if False:
                return 10
            self.ref = ref[0]
            ray.get(borrower.receive_ref.remote(ref))
            if failure:
                sys.exit(-1)

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return

    @ray.remote
    class Borrower:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.ref = None

        def receive_ref(self, ref):
            if False:
                i = 10
                return i + 15
            self.ref = ref[0]

        def resolve_ref(self):
            if False:
                print('Hello World!')
            assert self.ref is not None
            if failure:
                with pytest.raises(ray.exceptions.ReferenceCountingAssertionError):
                    ray.get(self.ref)
            else:
                ray.get(self.ref)

        def ping(self):
            if False:
                print('Hello World!')
            return
    parent = Parent.remote()
    borrower = Borrower.remote()
    ray.get(borrower.ping.remote())
    obj = ray.put(np.zeros(20 * 1024 * 1024, dtype=np.uint8))
    if failure:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(parent.pass_ref.remote([obj], borrower))
    else:
        ray.get(parent.pass_ref.remote([obj], borrower))
    obj_bytes = obj.binary()
    del obj
    _fill_object_store_and_get(obj_bytes, succeed=not failure)
    ray.get(borrower.resolve_ref.remote())

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows.')
def test_object_unpin(ray_start_cluster):
    if False:
        return 10
    nodes = []
    cluster = ray_start_cluster
    head_node = cluster.add_node(num_cpus=0, object_store_memory=100 * 1024 * 1024, _system_config={'subscriber_timeout_ms': 100, 'health_check_initial_delay_ms': 0, 'health_check_period_ms': 1000, 'health_check_failure_threshold': 5})
    ray.init(address=cluster.address)
    for i in range(2):
        nodes.append(cluster.add_node(num_cpus=1, resources={f'node_{i}': 1}, object_store_memory=100 * 1024 * 1024))
    cluster.wait_for_nodes()
    one_mb_array = np.ones(1 * 1024 * 1024, dtype=np.uint8)
    ten_mb_array = np.ones(10 * 1024 * 1024, dtype=np.uint8)

    @ray.remote
    class ObjectsHolder:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ten_mb_objs = []
            self.one_mb_objs = []

        def put_10_mb(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ten_mb_objs.append(ray.put(ten_mb_array))

        def put_1_mb(self):
            if False:
                i = 10
                return i + 15
            self.one_mb_objs.append(ray.put(one_mb_array))

        def pop_10_mb(self):
            if False:
                while True:
                    i = 10
            if len(self.ten_mb_objs) == 0:
                return False
            self.ten_mb_objs.pop()
            return True

        def pop_1_mb(self):
            if False:
                for i in range(10):
                    print('nop')
            if len(self.one_mb_objs) == 0:
                return False
            self.one_mb_objs.pop()
            return True
    one_mb_arrays = []
    ten_mb_arrays = []
    one_mb_arrays.append(ray.put(one_mb_array))
    ten_mb_arrays.append(ray.put(ten_mb_array))

    def check_memory(mb):
        if False:
            i = 10
            return i + 15
        return f'Plasma memory usage {mb} MiB' in memory_summary(address=head_node.address, stats_only=True)

    def wait_until_node_dead(node):
        if False:
            while True:
                i = 10
        for n in ray.nodes():
            if n['ObjectStoreSocketName'] == node.address_info['object_store_address']:
                return not n['Alive']
        return False
    wait_for_condition(lambda : check_memory(11))
    one_mb_arrays.pop()
    wait_for_condition(lambda : check_memory(10))
    ten_mb_arrays.pop()
    wait_for_condition(lambda : check_memory(0))
    actor_on_node_1 = ObjectsHolder.options(resources={'node_0': 1}).remote()
    actor_on_node_2 = ObjectsHolder.options(resources={'node_1': 1}).remote()
    ray.get(actor_on_node_1.put_1_mb.remote())
    ray.get(actor_on_node_1.put_10_mb.remote())
    ray.get(actor_on_node_2.put_1_mb.remote())
    ray.get(actor_on_node_2.put_10_mb.remote())
    wait_for_condition(lambda : check_memory(22))
    ray.get(actor_on_node_1.pop_1_mb.remote())
    ray.get(actor_on_node_2.pop_10_mb.remote())
    wait_for_condition(lambda : check_memory(11))
    cluster.remove_node(nodes[1], allow_graceful=False)
    wait_for_condition(lambda : wait_until_node_dead(nodes[1]))
    wait_for_condition(lambda : check_memory(10))
    ray.kill(actor_on_node_1)
    wait_for_condition(lambda : check_memory(0))

@pytest.mark.skipif(platform.system() in ['Windows'], reason='Failing on Windows.')
def test_object_unpin_stress(ray_start_cluster):
    if False:
        print('Hello World!')
    nodes = []
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, resources={'head': 1}, object_store_memory=1000 * 1024 * 1024)
    ray.init(address=cluster.address)
    for i in range(2):
        nodes.append(cluster.add_node(num_cpus=1, resources={f'node_{i}': 1}, object_store_memory=1000 * 1024 * 1024))
    cluster.wait_for_nodes()
    one_mb_array = np.ones(1 * 1024 * 1024, dtype=np.uint8)
    ten_mb_array = np.ones(10 * 1024 * 1024, dtype=np.uint8)

    @ray.remote
    class ObjectsHolder:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.ten_mb_objs = []
            self.one_mb_objs = []

        def put_10_mb(self):
            if False:
                return 10
            self.ten_mb_objs.append(ray.put(ten_mb_array))

        def put_1_mb(self):
            if False:
                print('Hello World!')
            self.one_mb_objs.append(ray.put(one_mb_array))

        def pop_10_mb(self):
            if False:
                return 10
            if len(self.ten_mb_objs) == 0:
                return False
            self.ten_mb_objs.pop()
            return True

        def pop_1_mb(self):
            if False:
                for i in range(10):
                    print('nop')
            if len(self.one_mb_objs) == 0:
                return False
            self.one_mb_objs.pop()
            return True

        def get_obj_size(self):
            if False:
                i = 10
                return i + 15
            return len(self.ten_mb_objs) * 10 + len(self.one_mb_objs)
    actor_on_node_1 = ObjectsHolder.options(resources={'node_0': 1}).remote()
    actor_on_node_2 = ObjectsHolder.options(resources={'node_1': 1}).remote()
    actor_on_head_node = ObjectsHolder.options(resources={'head': 1}).remote()
    ray.get(actor_on_node_1.get_obj_size.remote())
    ray.get(actor_on_node_2.get_obj_size.remote())
    ray.get(actor_on_head_node.get_obj_size.remote())

    def random_ops(actors):
        if False:
            return 10
        r = random.random()
        for actor in actors:
            if r <= 0.25:
                actor.put_10_mb.remote()
            elif r <= 0.5:
                actor.put_1_mb.remote()
            elif r <= 0.75:
                actor.pop_10_mb.remote()
            else:
                actor.pop_1_mb.remote()
    total_iter = 15
    for _ in range(total_iter):
        random_ops([actor_on_node_1, actor_on_node_2, actor_on_head_node])
    cluster.remove_node(nodes[1])
    for _ in range(total_iter):
        random_ops([actor_on_node_1, actor_on_head_node])
    total_size = sum([ray.get(actor_on_node_1.get_obj_size.remote()), ray.get(actor_on_head_node.get_obj_size.remote())])
    wait_for_condition(lambda : f'Plasma memory usage {total_size} MiB' in memory_summary(stats_only=True))

@pytest.mark.parametrize('inline_args', [True, False])
def test_inlined_nested_refs(ray_start_cluster, inline_args):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    config = {}
    if not inline_args:
        config['max_direct_call_object_size'] = 0
    cluster.add_node(num_cpus=2, object_store_memory=100 * 1024 * 1024, _system_config=config)
    ray.init(address=cluster.address)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            return

        def nested(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray.put('x')

    @ray.remote
    def nested_nested(a):
        if False:
            print('Hello World!')
        return a.nested.remote()

    @ray.remote
    def foo(ref):
        if False:
            while True:
                i = 10
        time.sleep(1)
        return ray.get(ref)
    a = Actor.remote()
    nested_nested_ref = nested_nested.remote(a)
    nested_ref = ray.get(nested_nested_ref)
    del nested_nested_ref
    x = foo.remote(nested_ref)
    del nested_ref
    ray.get(x)

@pytest.mark.parametrize('inline_args', [True, False])
def test_return_nested_ids(shutdown_only, inline_args):
    if False:
        while True:
            i = 10
    config = dict()
    if inline_args:
        config['max_direct_call_object_size'] = 100 * 1024 * 1024
    else:
        config['max_direct_call_object_size'] = 0
    ray.init(object_store_memory=100 * 1024 * 1024, _system_config=config)

    class Nested:

        def __init__(self, blocks):
            if False:
                for i in range(10):
                    print('nop')
            self._blocks = blocks

    @ray.remote
    def echo(fn):
        if False:
            while True:
                i = 10
        return fn()

    @ray.remote
    def create_nested():
        if False:
            for i in range(10):
                print('nop')
        refs = [ray.put(np.random.random(1024 * 1024)) for _ in range(10)]
        return Nested(refs)

    @ray.remote
    def test():
        if False:
            print('Hello World!')
        ref = create_nested.remote()
        result1 = ray.get(ref)
        del ref
        result = echo.remote(lambda : result1)
        del result1
        time.sleep(5)
        block = ray.get(result)._blocks[0]
        print(ray.get(block))
    ray.get(test.remote())

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_actor_constructor_borrowed_refs(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    class Borrower:

        def __init__(self, borrowed_refs):
            if False:
                for i in range(10):
                    print('nop')
            self.borrowed_refs = borrowed_refs

        def test(self):
            if False:
                i = 10
                return i + 15
            ray.get(self.borrowed_refs)
    ref = ray.put(np.random.random(1024 * 1024))
    b = Borrower.remote([ref])
    del ref
    for _ in range(3):
        ray.get(b.test.remote())
        time.sleep(1)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_deep_nested_refs(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    def f(x):
        if False:
            while True:
                i = 10
        print(f'=> step {x}')
        if x > 200:
            return x
        return f.remote(x + 1)
    r = f.remote(1)
    i = 0
    while isinstance(r, ray.ObjectRef):
        print(i, r)
        i += 1
        r = ray.get(r)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
def test_forward_nested_ref(shutdown_only):
    if False:
        return 10
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    def nested_ref():
        if False:
            while True:
                i = 10
        return ray.put(1)

    @ray.remote
    def nested_nested_ref():
        if False:
            print('Hello World!')
        return nested_ref.remote()

    @ray.remote
    class Borrower:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            return

        def pass_ref(self, middle_ref):
            if False:
                return 10
            self.inner_ref = ray.get(middle_ref)

        def check_ref(self):
            if False:
                print('Hello World!')
            ray.get(self.inner_ref)

    @ray.remote
    def pass_nested_ref(borrower, outer_ref):
        if False:
            i = 10
            return i + 15
        ray.get(borrower.pass_ref.remote(outer_ref[0]))
    b = Borrower.remote()
    outer_ref = nested_nested_ref.remote()
    x = pass_nested_ref.remote(b, [outer_ref])
    del outer_ref
    ray.get(x)
    for _ in range(3):
        ray.get(b.check_ref.remote())
        time.sleep(1)

def test_out_of_band_actor_handle_deserialization(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                print('Hello World!')
            return 1
    actor = Actor.remote()

    @ray.remote
    def func(config):
        if False:
            i = 10
            return i + 15
        config = copy.deepcopy(config)
        return ray.get(config['actor'].ping.remote())
    assert ray.get(func.remote({'actor': actor})) == 1

def test_out_of_band_actor_handle_bypass_reference_counting(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    import pickle
    ray.init(object_store_memory=100 * 1024 * 1024)

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                return 10
            return 1
    actor = Actor.remote()
    serialized = pickle.dumps({'actor': actor})
    del actor
    wait_for_num_actors(1, gcs_utils.ActorTableData.DEAD)
    config = pickle.loads(serialized)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(config['actor'].ping.remote())

def test_generators(one_worker_100MiB):
    if False:
        print('Hello World!')

    @ray.remote(num_returns='dynamic')
    def remote_generator():
        if False:
            while True:
                i = 10
        for _ in range(3):
            yield np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    gen = ray.get(remote_generator.remote())
    refs = list(gen)
    for r in refs:
        _fill_object_store_and_get(r)
    del gen
    for r in refs:
        _fill_object_store_and_get(r)
    refs_oids = [r.binary() for r in refs]
    del r
    del refs
    for r_oid in refs_oids:
        _fill_object_store_and_get(r_oid, succeed=False)

def test_lineage_leak(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init()

    @ray.remote
    def process(data):
        if False:
            return 10
        return b'\x00' * 100000000
    data = ray.put(b'\x00' * 100000000)
    ref = process.remote(data)
    ray.get(ref)
    del data
    del ref

    def check_usage():
        if False:
            print('Hello World!')
        from ray._private.internal_api import memory_summary
        return 'Plasma memory usage 0 MiB' in memory_summary(stats_only=True)
    wait_for_condition(check_usage)
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))