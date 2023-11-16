import time
from threading import Condition, Event
import pytest
import dramatiq
from dramatiq import group, middleware, pipeline
from dramatiq.middleware import GroupCallbacks
from dramatiq.results import Results, ResultTimeout

def test_messages_can_be_piped(stub_broker):
    if False:
        i = 10
        return i + 15

    @dramatiq.actor
    def add(x, y):
        if False:
            i = 10
            return i + 15
        return x + y
    pipe = add.message(1, 2) | add.message(3) | add.message(4)
    assert isinstance(pipe, pipeline)
    assert pipe.messages[0].options['pipe_target'] == pipe.messages[1].asdict()
    assert pipe.messages[1].options['pipe_target'] == pipe.messages[2].asdict()
    assert 'pipe_target' not in pipe.messages[2].options

def test_pipelines_flatten_child_pipelines(stub_broker):
    if False:
        return 10

    @dramatiq.actor
    def add(x, y):
        if False:
            print('Hello World!')
        return x + y
    pipe = pipeline([add.message(1, 2), add.message(3) | add.message(4), add.message(5)])
    assert len(pipe) == 4
    assert pipe.messages[0].args == (1, 2)
    assert pipe.messages[1].args == (3,)
    assert pipe.messages[2].args == (4,)
    assert pipe.messages[3].args == (5,)

def test_pipe_ignore_applies_to_receiving_message(stub_broker, stub_worker, result_backend):
    if False:
        return 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def return_args(*args):
        if False:
            while True:
                i = 10
        return args
    pipe = return_args.message(1) | return_args.message_with_options(pipe_ignore=True, args=(2,)) | return_args.message(3)
    pipe.run()
    stub_broker.join(return_args.queue_name)
    results = list(pipe.get_results())
    assert results == [[1], [2], [3, [2]]]

def test_pipeline_results_can_be_retrieved(stub_broker, stub_worker, result_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def add(x, y):
        if False:
            return 10
        return x + y
    pipe = add.message(1, 2) | (add.message(3) | add.message(4))
    pipe.run()
    assert pipe.get_result(block=True) == 10
    assert list(pipe.get_results()) == [3, 6, 10]

def test_pipeline_results_respect_timeouts(stub_broker, stub_worker, result_backend):
    if False:
        i = 10
        return i + 15
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def wait(n):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(n)
        return n * 2
    pipe = wait.message(1) | wait.message() | wait.message()
    pipe.run()
    with pytest.raises(ResultTimeout):
        for _ in pipe.get_results(block=True, timeout=1000):
            pass

def test_pipelines_expose_completion_stats(stub_broker, stub_worker, result_backend):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(Results(backend=result_backend))
    condition = Condition()

    @dramatiq.actor(store_results=True)
    def wait(n):
        if False:
            return 10
        time.sleep(n)
        with condition:
            condition.notify_all()
            return n
    pipe = wait.message(1) | wait.message()
    pipe.run()
    for count in range(1, len(pipe) + 1):
        with condition:
            condition.wait(2)
            time.sleep(0.1)
            assert pipe.completed_count == count
    assert pipe.completed

def test_pipelines_can_be_incomplete(stub_broker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def do_nothing():
        if False:
            print('Hello World!')
        return None
    pipe = do_nothing.message() | do_nothing.message_with_options(pipe_ignore=True)
    pipe.run()
    assert not pipe.completed

def test_groups_execute_jobs_in_parallel(stub_broker, stub_worker, result_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def wait():
        if False:
            i = 10
            return i + 15
        time.sleep(0.1)
    t = time.monotonic()
    g = group([wait.message() for _ in range(5)])
    g.run()
    results = list(g.get_results(block=True))
    assert time.monotonic() - t <= 0.5
    assert len(results) == len(g)
    assert g.completed

def test_groups_execute_inner_groups(stub_broker, stub_worker, result_backend):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def wait():
        if False:
            while True:
                i = 10
        time.sleep(0.1)
    t = time.monotonic()
    g = group((group((wait.message() for _ in range(2))) for _ in range(3)))
    g.run()
    results = list(g.get_results(block=True))
    assert time.monotonic() - t <= 0.5
    assert results == [[None, None]] * 3
    assert g.completed

def test_groups_can_time_out(stub_broker, stub_worker, result_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def wait():
        if False:
            i = 10
            return i + 15
        time.sleep(0.3)
    g = group((wait.message() for _ in range(2)))
    g.run()
    with pytest.raises(ResultTimeout):
        g.wait(timeout=100)
    assert not g.completed

def test_groups_expose_completion_stats(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))
    condition = Condition()

    @dramatiq.actor(store_results=True)
    def wait(n):
        if False:
            print('Hello World!')
        time.sleep(n)
        with condition:
            condition.notify_all()
            return n
    g = group((wait.message(n) for n in range(1, 4)))
    g.run()
    for count in range(1, len(g) + 1):
        with condition:
            condition.wait(5)
            time.sleep(0.1)
            assert g.completed_count == count
    assert g.completed

def test_pipeline_does_not_continue_to_next_actor_when_message_is_marked_as_failed(stub_broker, stub_worker):
    if False:
        i = 10
        return i + 15

    class FailMessageMiddleware(middleware.Middleware):

        def after_process_message(self, broker, message, *, result=None, exception=None):
            if False:
                return 10
            message.fail()
    stub_broker.add_middleware(FailMessageMiddleware())
    has_run = False

    @dramatiq.actor
    def do_nothing():
        if False:
            for i in range(10):
                print('nop')
        pass

    @dramatiq.actor
    def should_never_run():
        if False:
            print('Hello World!')
        nonlocal has_run
        has_run = True
    pipe = do_nothing.message_with_options(pipe_ignore=True) | should_never_run.message()
    pipe.run()
    stub_broker.join(should_never_run.queue_name, timeout=10 * 1000)
    stub_worker.join()
    assert not has_run

def test_pipeline_respects_own_delay(stub_broker, stub_worker, result_backend):
    if False:
        return 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def add(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    pipe = add.message(1, 2) | add.message(3)
    pipe.run(delay=10000)
    with pytest.raises(ResultTimeout):
        for _ in pipe.get_results(block=True, timeout=100):
            pass

def test_pipeline_respects_delay_of_first_message(stub_broker, stub_worker, result_backend):
    if False:
        for i in range(10):
            print('nop')
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def add(x, y):
        if False:
            print('Hello World!')
        return x + y
    pipe = add.message_with_options(args=(1, 2), delay=10000) | add.message(3)
    pipe.run()
    with pytest.raises(ResultTimeout):
        for _ in pipe.get_results(block=True, timeout=100):
            pass

def test_pipeline_respects_delay_of_second_message(stub_broker, stub_worker, result_backend):
    if False:
        i = 10
        return i + 15
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def add(x, y):
        if False:
            return 10
        return x + y
    pipe = add.message(1, 2) | add.message_with_options(args=(3,), delay=10000)
    pipe.run()
    with pytest.raises(ResultTimeout):
        for _ in pipe.get_results(block=True, timeout=100):
            pass

def test_pipeline_respects_bigger_of_first_messages_and_pipelines_delay(stub_broker, stub_worker, result_backend):
    if False:
        while True:
            i = 10
    stub_broker.add_middleware(Results(backend=result_backend))

    @dramatiq.actor(store_results=True)
    def add(x, y):
        if False:
            while True:
                i = 10
        return x + y
    pipe = add.message_with_options(args=(1, 2), delay=100) | add.message(3)
    pipe.run(delay=10000)
    with pytest.raises(ResultTimeout):
        for _ in pipe.get_results(block=True, timeout=300):
            pass

def test_groups_can_have_completion_callbacks(stub_broker, stub_worker, rate_limiter_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(GroupCallbacks(rate_limiter_backend))
    do_nothing_times = []
    finalize_times = []
    finalized = Event()

    @dramatiq.actor
    def do_nothing():
        if False:
            return 10
        do_nothing_times.append(time.monotonic())

    @dramatiq.actor
    def finalize(n):
        if False:
            print('Hello World!')
        assert n == 42
        finalize_times.append(time.monotonic())
        finalized.set()
    g = group((do_nothing.message() for n in range(5)))
    g.add_completion_callback(finalize.message(42))
    g.run()
    finalized.wait(timeout=30)
    assert len(do_nothing_times) == 5
    assert len(finalize_times) == 1
    assert sorted(do_nothing_times)[-1] <= finalize_times[0]

def test_groups_with_completion_callbacks_fail_unless_group_callbacks_is_set_up(stub_broker, stub_worker):
    if False:
        while True:
            i = 10

    @dramatiq.actor
    def do_nothing():
        if False:
            return 10
        pass

    @dramatiq.actor
    def finalize(n):
        if False:
            while True:
                i = 10
        pass
    g = group((do_nothing.message() for n in range(5)))
    g.add_completion_callback(finalize.message(42))
    with pytest.raises(RuntimeError):
        g.run()

def test_groups_of_pipelines_can_have_completion_callbacks(stub_broker, stub_worker, rate_limiter_backend):
    if False:
        print('Hello World!')
    stub_broker.add_middleware(GroupCallbacks(rate_limiter_backend))
    do_nothing_times = []
    finalize_times = []
    finalized = Event()

    @dramatiq.actor
    def do_nothing(_):
        if False:
            print('Hello World!')
        do_nothing_times.append(time.monotonic())

    @dramatiq.actor
    def finalize(n):
        if False:
            print('Hello World!')
        assert n == 42
        finalize_times.append(time.monotonic())
        finalized.set()
    g = group([do_nothing.message(1) | do_nothing.message(), do_nothing.message(1)])
    g.add_completion_callback(finalize.message(42))
    g.run()
    finalized.wait(timeout=30)
    assert len(do_nothing_times) == 3
    assert len(finalize_times) == 1
    assert sorted(do_nothing_times)[-1] <= finalize_times[0]