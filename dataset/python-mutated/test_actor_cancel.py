import asyncio
import os
import sys
import time
from collections import defaultdict
import pytest
import ray
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.exceptions import TaskCancelledError
from ray.util.state import list_tasks

def test_input_validation(shutdown_only):
    if False:
        return 10

    @ray.remote
    class A:

        async def f(self):
            pass
    a = A.remote()
    with pytest.raises(ValueError, match='force=True is not supported'):
        ray.cancel(a.f.remote(), force=True)

def test_async_actor_cancel(shutdown_only):
    if False:
        print('Hello World!')
    '\n    Test async actor task is canceled and\n    asyncio.CancelledError is raised within a task.\n    '
    ray.init(num_cpus=1)

    @ray.remote
    class VerifyActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.called = False
            self.running = False

        def called(self):
            if False:
                i = 10
                return i + 15
            self.called = True

        def set_running(self):
            if False:
                while True:
                    i = 10
            self.running = True

        def is_called(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.called

        def is_running(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.running

        def reset(self):
            if False:
                for i in range(10):
                    print('nop')
            self.called = False
            self.running = False

    @ray.remote
    class Actor:

        async def f(self, verify_actor):
            try:
                ray.get(verify_actor.set_running.remote())
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                print(asyncio.current_task().cancelled())
                assert not asyncio.current_task().cancelled()
                ray.get(verify_actor.called.remote())
                raise
            except Exception:
                return True
            return True
    v = VerifyActor.remote()
    a = Actor.remote()
    for i in range(50):
        ref = a.f.remote(v)
        wait_for_condition(lambda : ray.get(v.is_running.remote()))
        ray.cancel(ref)
        with pytest.raises(ray.exceptions.TaskCancelledError, match='was cancelled'):
            ray.get(ref)
        assert ray.get(v.is_running.remote())
        assert ray.get(v.is_called.remote())
        ray.get(v.reset.remote())

def test_async_actor_client_side_cancel(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a task is cancelled while it is queued on a client side.\n    It should raise ray.exceptions.TaskCancelledError.\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            self.f_called = False

        async def g(self, ref):
            await asyncio.sleep(30)

        async def f(self):
            self.f_called = True
            await asyncio.sleep(5)

        def is_f_called(self):
            if False:
                while True:
                    i = 10
            return self.f_called

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        time.sleep(100)
    a = Actor.remote()
    ref = a.f.remote()
    ray.cancel(ref)
    with pytest.raises(TaskCancelledError):
        ray.get(ref)
    cluster.add_node(num_cpus=1)
    assert not ray.get(a.is_f_called.remote())
    a = Actor.remote()
    ref_dep_not_resolved = a.g.remote(f.remote())
    ray.cancel(ref_dep_not_resolved)
    with pytest.raises(TaskCancelledError):
        ray.get(ref_dep_not_resolved)

@pytest.mark.skip(reason='The guarantee in this case is too weak now. Need more work.')
def test_in_flight_queued_requests_canceled(shutdown_only, monkeypatch):
    if False:
        return 10
    "\n    When there are large input size in-flight actor tasks\n    tasks are queued inside a RPC layer (core_worker_client.h)\n    In this case, we don't cancel a request from a client side\n    but wait until it is sent to the server side and cancel it.\n    See SendRequests() inside core_worker_client.h\n    "
    input_arg = b'1' * 15 * 1024
    sig = SignalActor.remote()

    @ray.remote
    class Actor:

        def __init__(self, signal_actor):
            if False:
                for i in range(10):
                    print('nop')
            self.signal_actor = signal_actor

        def f(self, input_arg):
            if False:
                while True:
                    i = 10
            ray.get(self.signal_actor.wait.remote())
            return True
    a = Actor.remote(sig)
    refs = [a.f.remote(input_arg) for _ in range(5000)]
    wait_for_condition(lambda : len(list_tasks(filters=[('STATE', '=', 'RUNNING')])) == 1)
    for ref in refs:
        ray.cancel(ref)
    first_ref = refs.pop(0)
    ray.get(sig.send.remote())
    canceled = 0
    for ref in refs:
        try:
            ray.get(ref)
        except TaskCancelledError:
            canceled += 1
    assert canceled > 2500
    assert ray.get(first_ref)

def test_async_actor_server_side_cancel(shutdown_only):
    if False:
        while True:
            i = 10
    '\n    Test Cancelation when a task is queued on a server side.\n    '

    @ray.remote
    class Actor:

        async def f(self):
            await asyncio.sleep(5)

        async def g(self):
            await asyncio.sleep(0)
    a = Actor.options(max_concurrency=1).remote()
    ray.get(a.__ray_ready__.remote())
    ref = a.f.remote()
    refs = [a.g.remote() for _ in range(100)]
    wait_for_condition(lambda : len(list_tasks(filters=[('name', '=', 'Actor.g'), ('STATE', '=', 'SUBMITTED_TO_WORKER')])) == 100)
    for ref in refs:
        ray.cancel(ref)
    tasks = list_tasks(filters=[('name', '=', 'Actor.g')])
    for ref in refs:
        with pytest.raises(TaskCancelledError, match=ref.task_id().hex()):
            ray.get(ref)
    for task in tasks:
        assert task.state == 'SUBMITTED_TO_WORKER'

def test_async_actor_cancel_after_task_finishes(shutdown_only):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:

        async def f(self):
            await asyncio.sleep(5)

        async def empty(self):
            pass
    a = Actor.options(max_concurrency=1).remote()
    ref = a.empty.remote()
    ref2 = a.empty.remote()
    ray.get([ref, ref2])
    ray.cancel(ref)
    ray.cancel(ref2)
    ray.get([ref, ref2])

def test_async_actor_cancel_restart(ray_start_cluster, monkeypatch):
    if False:
        return 10
    '\n    Verify a cancelation works if actor is restarted.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_testing_asio_delay_us', 'CoreWorkerService.grpc_server.CancelTask=3000000:3000000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        node = cluster.add_node(num_cpus=1)

        @ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
        class Actor:

            async def f(self):
                await asyncio.sleep(10)
        a = Actor.remote()
        ref = a.f.remote()
        ray.get(a.__ray_ready__.remote())
        ray.cancel(ref)
        cluster.remove_node(node)
        (r, ur) = ray.wait([ref])
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(ref)
        cluster.add_node(num_cpus=1)
        ray.get(a.__ray_ready__.remote())
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(ref)

def test_remote_cancel(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class Actor:

        async def sleep(self):
            await asyncio.sleep(1000)

    @ray.remote
    def f(refs):
        if False:
            print('Hello World!')
        ref = refs[0]
        ray.cancel(ref)
    a = Actor.remote()
    sleep_ref = a.sleep.remote()
    wait_for_condition(lambda : list_tasks(filters=[('name', '=', 'Actor.sleep')]))
    ref = f.remote([sleep_ref])
    with pytest.raises(ray.exceptions.TaskCancelledError):
        ray.get(sleep_ref)

@pytest.mark.skip(reason="Currently not passing. There's one edge case to fix.")
def test_cancel_stress(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init()

    @ray.remote
    class Actor:

        async def sleep(self):
            await asyncio.sleep(1000)
    actors = [Actor.remote() for _ in range(30)]
    refs = []
    for _ in range(20):
        for actor in actors:
            for i in range(100):
                ref = actor.sleep.remote()
                refs.append(ref)
                if i % 2 == 0:
                    ray.cancel(ref)
    for ref in refs:
        ray.cancel(ref)
    for ref in refs:
        with pytest.raises((ray.exceptions.TaskCancelledError, TaskCancelledError)):
            ray.get(ref)

def test_cancel_recursive_tree(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    'Verify recursive cancel works for tree-nested tasks.\n\n    Task A -> Task B\n           -> Task C\n    '
    ray.init(num_cpus=16)

    @ray.remote
    def child():
        if False:
            for i in range(10):
                print('nop')
        for _ in range(5):
            time.sleep(1)
        return True

    @ray.remote
    class ChildActor:

        async def child(self):
            await asyncio.sleep(5)
            return True

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.children_refs = defaultdict(list)

        def get_children_refs(self, task_id):
            if False:
                while True:
                    i = 10
            return self.children_refs[task_id]

        async def run(self, child_actor, sig):
            ref1 = child.remote()
            ref2 = child_actor.child.remote()
            task_id = ray.get_runtime_context().get_task_id()
            self.children_refs[task_id].append(ref1)
            self.children_refs[task_id].append(ref2)
            await sig.wait.remote()
            await ref1
            await ref2
    sig = SignalActor.remote()
    child_actor = ChildActor.remote()
    a = Actor.remote()
    ray.get(a.__ray_ready__.remote())
    '\n    Test the basic case.\n    '
    run_ref = a.run.remote(child_actor, sig)
    task_id = run_ref.task_id().hex()
    wait_for_condition(lambda : list_tasks(filters=[('task_id', '=', task_id)])[0].state == 'RUNNING')
    ray.cancel(run_ref, recursive=True)
    ray.get(sig.send.remote())
    children_refs = ray.get(a.get_children_refs.remote(task_id))
    for ref in children_refs + [run_ref]:
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ref)
    '\n    Test recursive = False\n    '
    run_ref = a.run.remote(child_actor, sig)
    task_id = run_ref.task_id().hex()
    wait_for_condition(lambda : list_tasks(filters=[('task_id', '=', task_id)])[0].state == 'RUNNING')
    ray.cancel(run_ref, recursive=False)
    ray.get(sig.send.remote())
    children_refs = ray.get(a.get_children_refs.remote(task_id))
    for ref in children_refs:
        assert ray.get(ref)
    with pytest.raises(ray.exceptions.TaskCancelledError):
        ray.get(run_ref)
    '\n    Test concurrent cases.\n    '
    run_refs = [a.run.remote(ChildActor.remote(), sig) for _ in range(10)]
    task_ids = []
    for (i, run_ref) in enumerate(run_refs):
        task_id = run_ref.task_id().hex()
        task_ids.append(task_id)
        wait_for_condition(lambda task_id=task_id: list_tasks(filters=[('task_id', '=', task_id)])[0].state == 'RUNNING')
        children_refs = ray.get(a.get_children_refs.remote(task_id))
        for child_ref in children_refs:
            task_id = child_ref.task_id().hex()
            wait_for_condition(lambda task_id=task_id: list_tasks(filters=[('task_id', '=', task_id)])[0].state == 'RUNNING')
        recursive = i % 2 == 0
        ray.cancel(run_ref, recursive=recursive)
    ray.get(sig.send.remote())
    for (i, task_id) in enumerate(task_ids):
        children_refs = ray.get(a.get_children_refs.remote(task_id))
        if i % 2 == 0:
            for ref in children_refs:
                with pytest.raises(ray.exceptions.TaskCancelledError):
                    ray.get(ref)
        else:
            for ref in children_refs:
                assert ray.get(ref)
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(run_ref)

@pytest.mark.parametrize('recursive', [True, False])
def test_cancel_recursive_chain(shutdown_only, recursive):
    if False:
        while True:
            i = 10

    @ray.remote
    class RecursiveActor:

        def __init__(self, child=None):
            if False:
                return 10
            self.child = child
            self.chlid_ref = None

        async def run(self, sig):
            if self.child is None:
                await sig.wait.remote()
                return True
            ref = self.child.run.remote(sig)
            self.child_ref = ref
            return await ref

        def get_child_ref(self):
            if False:
                print('Hello World!')
            return self.child_ref
    sig = SignalActor.remote()
    r1 = RecursiveActor.remote()
    r2 = RecursiveActor.remote(r1)
    r3 = RecursiveActor.remote(r2)
    r4 = RecursiveActor.remote(r3)
    ref = r4.run.remote(sig)
    ray.get(r4.__ray_ready__.remote())
    wait_for_condition(lambda : len(list_tasks(filters=[('name', '=', 'RecursiveActor.run')])) == 4)
    ray.cancel(ref, recursive=recursive)
    ray.get(sig.send.remote())
    if recursive:
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ref)
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ray.get(r4.get_child_ref.remote()))
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ray.get(r3.get_child_ref.remote()))
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ray.get(r2.get_child_ref.remote()))
    else:
        assert ray.get(ray.get(r2.get_child_ref.remote()))
        assert ray.get(ray.get(r3.get_child_ref.remote()))
        assert ray.get(ray.get(r4.get_child_ref.remote()))
        with pytest.raises(ray.exceptions.TaskCancelledError):
            ray.get(ref)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))