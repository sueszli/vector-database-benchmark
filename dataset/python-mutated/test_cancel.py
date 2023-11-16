import os
import random
import signal
import sys
import threading
import _thread
import time
import pytest
import ray
from ray.exceptions import TaskCancelledError, RayTaskError, GetTimeoutError, WorkerCrashedError, ObjectLostError
from ray._private.utils import DeferSigint
from ray._private.test_utils import SignalActor, wait_for_condition
from ray.util.state import list_tasks

def valid_exceptions(use_force):
    if False:
        i = 10
        return i + 15
    if use_force:
        return (RayTaskError, TaskCancelledError, WorkerCrashedError, ObjectLostError)
    else:
        return TaskCancelledError

@pytest.mark.parametrize('use_force', [True, False])
def test_cancel_chain(ray_start_regular, use_force):
    if False:
        for i in range(10):
            print('nop')
    signaler = SignalActor.remote()

    @ray.remote
    def wait_for(t):
        if False:
            for i in range(10):
                print('nop')
        return ray.get(t[0])
    obj1 = wait_for.remote([signaler.wait.remote()])
    obj2 = wait_for.remote([obj1])
    obj3 = wait_for.remote([obj2])
    obj4 = wait_for.remote([obj3])
    assert len(ray.wait([obj1], timeout=0.1)[0]) == 0
    ray.cancel(obj1, force=use_force)
    for ob in [obj1, obj2, obj3, obj4]:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(ob)
    signaler2 = SignalActor.remote()
    obj1 = wait_for.remote([signaler2.wait.remote()])
    obj2 = wait_for.remote([obj1])
    obj3 = wait_for.remote([obj2])
    obj4 = wait_for.remote([obj3])
    assert len(ray.wait([obj3], timeout=0.1)[0]) == 0
    ray.cancel(obj3, force=use_force)
    for ob in [obj3, obj4]:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(ob)
    with pytest.raises(GetTimeoutError):
        ray.get(obj1, timeout=0.1)
    with pytest.raises(GetTimeoutError):
        ray.get(obj2, timeout=0.1)
    signaler2.send.remote()
    ray.get(obj1)

@pytest.mark.parametrize('use_force', [True, False])
def test_cancel_during_arg_deser(ray_start_regular, use_force):
    if False:
        return 10
    time_to_sleep = 5

    class SlowToDeserialize:

        def __reduce__(self):
            if False:
                i = 10
                return i + 15

            def reconstruct():
                if False:
                    for i in range(10):
                        print('nop')
                import time
                time.sleep(time_to_sleep)
                return SlowToDeserialize()
            return (reconstruct, ())

    @ray.remote
    def dummy(a: SlowToDeserialize):
        if False:
            return 10
        assert False
    arg = SlowToDeserialize()
    obj = dummy.remote(arg)
    assert len(ray.wait([obj], timeout=0.1)[0]) == 0
    ray.cancel(obj, force=use_force)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(obj)

def test_defer_sigint():
    if False:
        return 10
    signal_was_deferred = False
    orig_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        with DeferSigint():
            _thread.interrupt_main()
            time.sleep(1)
            signal_was_deferred = True
    except KeyboardInterrupt:
        assert signal_was_deferred
        assert signal.getsignal(signal.SIGINT) is orig_sigint_handler
    else:
        pytest.fail('SIGINT signal was never sent in test')

def test_defer_sigint_monkey_patch():
    if False:
        return 10
    orig_sigint_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(ValueError):
        with DeferSigint():
            signal.signal(signal.SIGINT, orig_sigint_handler)

def test_defer_sigint_noop_in_non_main_thread():
    if False:
        return 10

    def check_no_defer():
        if False:
            print('Hello World!')
        cm = DeferSigint.create_if_main_thread()
        assert not isinstance(cm, DeferSigint)
    check_no_defer_thread = threading.Thread(target=check_no_defer)
    try:
        check_no_defer_thread.start()
        check_no_defer_thread.join()
    except AssertionError as e:
        pytest.fail(f'DeferSigint.create_if_main_thread() unexpected returned a DeferSigint instance when not in the main thread: {e}')
    signal_was_deferred = False

    def maybe_defer():
        if False:
            for i in range(10):
                print('nop')
        nonlocal signal_was_deferred
        with DeferSigint.create_if_main_thread() as cm:
            assert not isinstance(cm, DeferSigint)
            _thread.interrupt_main()
            time.sleep(1)
            signal_was_deferred = True
    maybe_defer_thread = threading.Thread(target=maybe_defer)
    try:
        maybe_defer_thread.start()
        maybe_defer_thread.join()
    except KeyboardInterrupt:
        assert not signal_was_deferred
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    else:
        pytest.fail('SIGINT signal was never sent in test')

@pytest.mark.skip('Using unsupported API.')
def test_cancel_during_arg_deser_non_reentrant_import(ray_start_regular):
    if False:
        i = 10
        return i + 15

    def non_reentrant_import():
        if False:
            return 10
        import pandas

    def non_reentrant_import_and_delegate(obj):
        if False:
            print('Hello World!')
        non_reentrant_import()
        reduced = obj.__reduce__()
        func = reduced[0]
        args = reduced[1]
        others = reduced[2:]

        def non_reentrant_import_on_reconstruction(*args, **kwargs):
            if False:
                print('Hello World!')
            non_reentrant_import()
            return func(*args, **kwargs)
        out = (non_reentrant_import_on_reconstruction, args) + others
        return out

    class DummyArg:
        pass

    def register_non_reentrant_import_and_delegate_reducer(worker_info):
        if False:
            print('Hello World!')
        from ray.exceptions import RayTaskError
        context = ray._private.worker.global_worker.get_serialization_context()
        context._register_cloudpickle_reducer(DummyArg, non_reentrant_import_and_delegate)
        context._register_cloudpickle_reducer(RayTaskError, non_reentrant_import_and_delegate)
    ray._private.worker.global_worker.run_function_on_all_workers(register_non_reentrant_import_and_delegate_reducer)
    time.sleep(3)

    @ray.remote
    def run_and_fail(a: DummyArg):
        if False:
            for i in range(10):
                print('nop')
        assert False
    arg = DummyArg()
    obj = run_and_fail.remote(arg)
    timeout_to_reach_arg_deserialization = 0.2
    assert len(ray.wait([obj], timeout=timeout_to_reach_arg_deserialization)[0]) == 0
    use_force = False
    ray.cancel(obj, force=use_force)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(obj)

@pytest.mark.parametrize('use_force', [True, False])
def test_cancel_multiple_dependents(ray_start_regular, use_force):
    if False:
        for i in range(10):
            print('nop')
    signaler = SignalActor.remote()

    @ray.remote
    def wait_for(t):
        if False:
            return 10
        return ray.get(t[0])
    head = wait_for.remote([signaler.wait.remote()])
    deps = []
    for _ in range(3):
        deps.append(wait_for.remote([head]))
    assert len(ray.wait([head], timeout=0.1)[0]) == 0
    ray.cancel(head, force=use_force)
    for d in deps:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(d)
    head2 = wait_for.remote([signaler.wait.remote()])
    deps2 = []
    for _ in range(3):
        deps2.append(wait_for.remote([head]))
    for d in deps2:
        ray.cancel(d, force=use_force)
    for d in deps2:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(d)
    signaler.send.remote()
    ray.get(head2)

@pytest.mark.parametrize('use_force', [True, False])
def test_single_cpu_cancel(shutdown_only, use_force):
    if False:
        return 10
    ray.init(num_cpus=1)
    signaler = SignalActor.remote()

    @ray.remote
    def wait_for(t):
        if False:
            while True:
                i = 10
        return ray.get(t[0])
    obj1 = wait_for.remote([signaler.wait.remote()])
    obj2 = wait_for.remote([obj1])
    obj3 = wait_for.remote([obj2])
    indep = wait_for.remote([signaler.wait.remote()])
    assert len(ray.wait([obj3], timeout=0.1)[0]) == 0
    ray.cancel(obj3, force=use_force)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(obj3)
    ray.cancel(obj1, force=use_force)
    for d in [obj1, obj2]:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(d)
    signaler.send.remote()
    ray.get(indep)

@pytest.mark.parametrize('use_force', [True, False])
def test_comprehensive(ray_start_regular, use_force):
    if False:
        i = 10
        return i + 15
    signaler = SignalActor.remote()

    @ray.remote
    def wait_for(t):
        if False:
            while True:
                i = 10
        ray.get(t[0])
        return 'Result'

    @ray.remote
    def combine(a, b):
        if False:
            for i in range(10):
                print('nop')
        return str(a) + str(b)
    a = wait_for.remote([signaler.wait.remote()])
    b = wait_for.remote([signaler.wait.remote()])
    combo = combine.remote(a, b)
    a2 = wait_for.remote([a])
    assert len(ray.wait([a, b, a2, combo], timeout=1)[0]) == 0
    ray.cancel(a, force=use_force)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(a, timeout=10)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(a2, timeout=40)
    signaler.send.remote()
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(combo)

@pytest.mark.parametrize('use_force', [True])
def test_stress(shutdown_only, use_force):
    if False:
        while True:
            i = 10
    ray.init(num_cpus=1)

    @ray.remote
    def infinite_sleep(y):
        if False:
            i = 10
            return i + 15
        if y:
            while True:
                time.sleep(1 / 10)
    first = infinite_sleep.remote(True)
    sleep_or_no = [random.randint(0, 1) for _ in range(100)]
    tasks = [infinite_sleep.remote(i) for i in sleep_or_no]
    cancelled = set()
    for t in tasks:
        if random.random() > 0.5:
            ray.cancel(t, force=use_force)
            cancelled.add(t)
    ray.cancel(first, force=use_force)
    cancelled.add(first)
    for done in cancelled:
        with pytest.raises(valid_exceptions(use_force)):
            ray.get(done, timeout=120)
    for (indx, t) in enumerate(tasks):
        if sleep_or_no[indx]:
            ray.cancel(t, force=use_force)
            cancelled.add(t)
    for (indx, t) in enumerate(tasks):
        if t in cancelled:
            with pytest.raises(valid_exceptions(use_force)):
                ray.get(t, timeout=120)
        else:
            ray.get(t, timeout=120)

@pytest.mark.parametrize('use_force', [True, False])
def test_fast(shutdown_only, use_force):
    if False:
        return 10
    ray.init(num_cpus=2)

    @ray.remote
    def fast(y):
        if False:
            for i in range(10):
                print('nop')
        return y
    signaler = SignalActor.remote()
    ids = list()
    for _ in range(100):
        x = fast.remote('a')
        time.sleep(0.1)
        ray.cancel(x, force=use_force)
        ids.append(x)

    @ray.remote
    def wait_for(y):
        if False:
            for i in range(10):
                print('nop')
        return y
    sig = signaler.wait.remote()
    for _ in range(5000):
        x = wait_for.remote(sig)
        ids.append(x)
    for idx in range(100, 5100):
        if random.random() > 0.95:
            ray.cancel(ids[idx], force=use_force)
    signaler.send.remote()
    for (i, obj_ref) in enumerate(ids):
        try:
            ray.get(obj_ref, timeout=120)
        except Exception as e:
            assert isinstance(e, valid_exceptions(use_force)), f'Failure on iteration: {i}'

@pytest.mark.parametrize('use_force', [True, False])
def test_remote_cancel(ray_start_regular, use_force):
    if False:
        i = 10
        return i + 15
    signaler = SignalActor.remote()

    @ray.remote
    def wait_for(y):
        if False:
            i = 10
            return i + 15
        return ray.get(y[0])

    @ray.remote
    def remote_wait(sg):
        if False:
            while True:
                i = 10
        return [wait_for.remote([sg[0]])]
    sig = signaler.wait.remote()
    outer = remote_wait.remote([sig])
    inner = ray.get(outer)[0]
    with pytest.raises(GetTimeoutError):
        ray.get(inner, timeout=1)
    ray.cancel(inner, force=use_force)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(inner, timeout=10)

@pytest.mark.parametrize('use_force', [True, False])
def test_recursive_cancel(shutdown_only, use_force):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=4)

    @ray.remote(num_cpus=1)
    def inner():
        if False:
            for i in range(10):
                print('nop')
        while True:
            time.sleep(0.1)

    @ray.remote(num_cpus=1)
    def outer():
        if False:
            print('Hello World!')
        x = [inner.remote()]
        print(x)
        while True:
            time.sleep(0.1)

    @ray.remote(num_cpus=4)
    def many_resources():
        if False:
            for i in range(10):
                print('nop')
        return 300
    outer_fut = outer.remote()
    many_fut = many_resources.remote()
    with pytest.raises(GetTimeoutError):
        ray.get(many_fut, timeout=1)
    ray.cancel(outer_fut)
    with pytest.raises(valid_exceptions(use_force)):
        ray.get(outer_fut, timeout=10)
    assert ray.get(many_fut, timeout=30)

def test_recursive_cancel_actor_task(shutdown_only):
    if False:
        return 10
    ray.init()

    @ray.remote(num_cpus=0)
    class Semaphore:

        def wait(self):
            if False:
                for i in range(10):
                    print('nop')
            import time
            time.sleep(600)

    @ray.remote(num_cpus=0)
    class Actor2:

        def __init__(self, obj):
            if False:
                return 10
            (self.obj,) = obj

        def cancel(self):
            if False:
                for i in range(10):
                    print('nop')
            ray.cancel(self.obj)

    @ray.remote
    def task(sema):
        if False:
            for i in range(10):
                print('nop')
        return ray.get(sema.wait.remote())
    sema = Semaphore.remote()
    t = task.remote(sema)

    def wait_until_wait_task_starts():
        if False:
            for i in range(10):
                print('nop')
        wait_state = list_tasks(filters=[('func_or_class_name', '=', 'Semaphore.wait')])[0]
        return wait_state['state'] == 'RUNNING'
    wait_for_condition(wait_until_wait_task_starts)
    a2 = Actor2.remote((t,))
    a2.cancel.remote()
    with pytest.raises(RayTaskError, match='TaskCancelledError'):
        ray.get(t)
    wait_state = list_tasks(filters=[('func_or_class_name', '=', 'Semaphore.wait')])
    assert len(wait_state) == 1
    wait_state = wait_state[0]
    task_state = list_tasks(filters=[('func_or_class_name', '=', 'task')])
    assert len(task_state) == 1
    task_state = task_state[0]

    def verify():
        if False:
            while True:
                i = 10
        wait_state = list_tasks(filters=[('func_or_class_name', '=', 'Semaphore.wait')])
        assert len(wait_state) == 1
        wait_state = wait_state[0]
        task_state = list_tasks(filters=[('func_or_class_name', '=', 'task')])
        assert len(task_state) == 1
        task_state = task_state[0]
        assert task_state['state'] == 'FINISHED'
        assert wait_state['state'] == 'RUNNING'
        return True
    wait_for_condition(verify)

@pytest.mark.skip('Actor cancelation works now.')
def test_recursive_cancel_error_messages(shutdown_only, capsys):
    if False:
        i = 10
        return i + 15
    "\n    Make sure the error message printed from the core worker\n    when the recursive cancelation fails it correct.\n\n    It should only sample 10 tasks.\n\n    Example output:\n    (task pid=55118) [2023-02-07 12:51:45,000 E 55118 6637966] core_worker.cc:3360: Unknown error: Failed to cancel all the children tasks of 85748392bcd969ccffffffffffffffffffffffff01000000 recursively. # noqa\n    (task pid=55118) Here are up to 10 samples tasks that failed to be canceled # noqa\n    (task pid=55118) \tb2094147c88795c9678740914e63d022610d70d501000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \td33d38e548ef4f998e63e2e1aaf05a3270e2722e01000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \t46009b11e76c891daae7fa9272cac4a2755bb1a901000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \t163f27568ace977d38a1ee4f11d3a358e694488901000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \t4a0fec5a878ccb98afd7e48837351bfd14957bf001000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \t45757cb171c13b7409953bfd8065a5eb36ba936201000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \ta5220c501dc8f624f3ab13166dcf73e3f35068a101000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \tf8bdb7979cd66dfc0fb4f8225e6197a779e4b7e901000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \t3d941239bca36a1cef9d9405523ce46181ebecfe01000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) \td6fe9100f5c082db407a983e2f7ada3b5a065e3f01000000, Invalid: Actor task cancellation is not supported. The task won't be cancelled. # noqa\n    (task pid=55118) Total Recursive cancelation success: 0, failures: 12\n    "
    ray.init(num_cpus=12)
    NUM_ACTORS = 12

    @ray.remote(num_cpus=0)
    class Semaphore:

        def wait(self):
            if False:
                for i in range(10):
                    print('nop')
            print('wait called')
            import time
            time.sleep(600)

    @ray.remote
    def task(semas):
        if False:
            i = 10
            return i + 15
        refs = []
        for sema in semas:
            refs.append(sema.wait.remote())
        return ray.get(refs)
    semas = [Semaphore.remote() for _ in range(NUM_ACTORS)]
    t = task.remote(semas)

    def wait_until_wait_task_starts():
        if False:
            for i in range(10):
                print('nop')
        wait_state = list_tasks(filters=[('func_or_class_name', '=', 'Semaphore.wait')])
        return len(wait_state) == 12
    wait_for_condition(wait_until_wait_task_starts)
    ray.cancel(t)
    with pytest.raises(RayTaskError, match='TaskCancelledError'):
        ray.get(t)
    msgs = capsys.readouterr().err.strip(' \n').split('\n')
    total_result = msgs[-1]
    samples = []
    for msg in msgs:
        if 'Invalid: Actor task cancellation is not supported.' in msg:
            samples.append(msg)
    assert len(samples) == 10
    found_total_msg: bool = True
    for total_result in reversed(msgs):
        found_total_msg = found_total_msg or f'Total Recursive cancelation success: 0, failures:{NUM_ACTORS}' in msg
        if found_total_msg:
            break
    assert found_total_msg
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))