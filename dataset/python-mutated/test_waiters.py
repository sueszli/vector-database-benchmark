import asyncio
import threading
import time
import pytest
from prefect._internal.concurrency.calls import Call
from prefect._internal.concurrency.cancellation import CancelledError
from prefect._internal.concurrency.threads import WorkerThread
from prefect._internal.concurrency.waiters import AsyncWaiter, SyncWaiter, get_waiter_for_thread

def fake_fn(*args, **kwargs):
    if False:
        return 10
    pass

def identity(x):
    if False:
        while True:
            i = 10
    return x

async def aidentity(x):
    return x

def raises(exc):
    if False:
        for i in range(10):
            print('nop')
    raise exc

async def araises(exc):
    raise exc

def sleep_repeatedly(seconds: int):
    if False:
        while True:
            i = 10
    for i in range(seconds * 10):
        time.sleep(float(i) / 10)

async def test_get_waiter_with_call_done():
    call = Call.new(identity, 1)
    waiter = AsyncWaiter(call)

    def call_is_done():
        if False:
            while True:
                i = 10
        return True
    waiter.call_is_done = call_is_done
    waiter_for_thread = get_waiter_for_thread(threading.current_thread())
    assert waiter_for_thread is None

@pytest.mark.parametrize('cls', [AsyncWaiter, SyncWaiter])
async def test_waiter_repr(cls):
    waiter = cls(Call.new(fake_fn, 1, 2))
    assert repr(waiter) == f"<{cls.__name__} call=fake_fn(1, 2), owner='MainThread'>"

def test_async_waiter_created_outside_of_loop():
    if False:
        i = 10
        return i + 15
    call = Call.new(identity, 1)
    call.run()
    asyncio.run(AsyncWaiter(call).wait())
    assert call.result() == 1

def test_async_waiter_early_submission():
    if False:
        return 10
    call = Call.new(identity, 1)
    waiter = AsyncWaiter(call)
    callback = waiter.submit(Call.new(identity, 2))
    call.run()
    asyncio.run(waiter.wait())
    assert call.result() == 1
    assert callback.result() == 2

def test_async_waiter_done_callback():
    if False:
        while True:
            i = 10
    call = Call.new(identity, 1)
    waiter = AsyncWaiter(call)
    callback = Call.new(identity, 2)
    assert not callback.future.done()
    waiter.add_done_callback(callback)
    call.run()
    asyncio.run(waiter.wait())
    assert call.result() == 1
    assert callback.result() == 2

def test_async_waiter_done_callbacks():
    if False:
        i = 10
        return i + 15
    call = Call.new(identity, 1)
    waiter = AsyncWaiter(call)
    callbacks = [Call.new(identity, i) for i in range(10)]
    for callback in callbacks:
        waiter.add_done_callback(callback)
    call.run()
    asyncio.run(waiter.wait())
    assert call.result() == 1
    for (i, callback) in enumerate(callbacks):
        assert callback.result() == i

def test_sync_waiter_timeout_in_worker_thread():
    if False:
        print('Hello World!')
    '\n    In this test, a timeout is raised due to a slow call that is occurring on the worker\n    thread.\n    '
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:
        call = Call.new(sleep_repeatedly, 1)
        waiter = SyncWaiter(call)
        waiter.add_done_callback(done_callback)
        call.set_timeout(0.1)
        runner.submit(call)
    t0 = time.time()
    waiter.wait()
    t1 = time.time()
    with pytest.raises(CancelledError):
        call.result()
    assert t1 - t0 < 1
    assert call.cancelled()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on cancel'

@pytest.mark.timeout(method='thread')
def test_sync_waiter_timeout_in_main_thread():
    if False:
        for i in range(10):
            print('nop')
    '\n    In this test, a timeout is raised due to a slow call that is sent back to the main\n    thread by the worker thread.\n    '
    done_callback = Call.new(identity, 1)
    waiting_callback = Call.new(sleep_repeatedly, 2)
    with WorkerThread(run_once=True) as runner:

        def on_worker_thread():
            if False:
                return 10
            waiter.submit(waiting_callback)
            waiting_callback.result()
        call = Call.new(on_worker_thread)
        waiter = SyncWaiter(call)
        waiter.add_done_callback(done_callback)
        call.set_timeout(0.1)
        runner.submit(call)
        t0 = time.time()
        waiter.wait()
        t1 = time.time()
    with pytest.raises(CancelledError):
        call.result()
    with pytest.raises(CancelledError):
        waiting_callback.result()
    assert t1 - t0 < 2
    assert waiting_callback.cancelled()
    assert call.cancelled()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on cancel'

async def test_async_waiter_timeout_in_worker_thread():
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:
        call = Call.new(sleep_repeatedly, 1)
        waiter = AsyncWaiter(call)
        waiter.add_done_callback(done_callback)
        call.set_timeout(0.1)
        runner.submit(call)
        t0 = time.time()
        await waiter.wait()
        t1 = time.time()
    assert t1 - t0 < 1
    with pytest.raises(CancelledError):
        call.result()
    assert call.cancelled()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on cancel'

async def test_async_waiter_timeout_in_main_thread():
    done_callback = Call.new(identity, 1)
    waiting_callback = Call.new(asyncio.sleep, 2)
    with WorkerThread(run_once=True) as runner:

        def on_worker_thread():
            if False:
                while True:
                    i = 10
            waiter.submit(waiting_callback)
            waiting_callback.result()
        call = Call.new(on_worker_thread)
        waiter = AsyncWaiter(call)
        waiter.add_done_callback(done_callback)
        call.set_timeout(1)
        runner.submit(call)
        t0 = time.time()
        await waiter.wait()
        t1 = time.time()
    with pytest.raises(CancelledError):
        call.result()
    with pytest.raises(CancelledError):
        waiting_callback.result()
    assert t1 - t0 < 2
    assert call.cancelled()
    assert waiting_callback.cancelled()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on cancel'

async def test_async_waiter_timeout_in_worker_thread_mixed_sleeps():

    def sync_then_async_sleep():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        return asyncio.sleep(0.25)
    with WorkerThread(run_once=True) as runner:
        call = Call.new(sync_then_async_sleep)
        waiter = AsyncWaiter(call)
        call.set_timeout(0.3)
        runner.submit(call)
        t0 = time.time()
        await waiter.wait()
        t1 = time.time()
        assert t1 - t0 < 1
    with pytest.raises(CancelledError):
        call.result()
    assert call.cancelled()

@pytest.mark.parametrize('raise_fn', [raises, araises], ids=['sync', 'async'])
@pytest.mark.parametrize('exception_cls', [BaseException, KeyboardInterrupt, SystemExit])
async def test_async_waiter_base_exception_in_worker_thread(exception_cls, raise_fn):
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:
        call = Call.new(raise_fn, exception_cls('test'))
        waiter = AsyncWaiter(call)
        waiter.add_done_callback(done_callback)
        runner.submit(call)
        await waiter.wait()
    with pytest.raises(exception_cls, match='test'):
        call.result()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on exception'

@pytest.mark.parametrize('raise_fn', [raises, araises], ids=['sync', 'async'])
@pytest.mark.parametrize('exception_cls', [BaseException, KeyboardInterrupt, SystemExit])
async def test_async_waiter_base_exception_in_main_thread(exception_cls, raise_fn):
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:

        def on_worker_thread():
            if False:
                i = 10
                return i + 15
            callback = Call.new(raise_fn, exception_cls('test'))
            waiter.submit(callback)
            return callback
        call = Call.new(on_worker_thread)
        waiter = AsyncWaiter(call)
        waiter.add_done_callback(done_callback)
        runner.submit(call)
        await waiter.wait()
        callback = call.result()
    with pytest.raises(exception_cls, match='test'):
        callback.result()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on exception'

@pytest.mark.parametrize('raise_fn', [raises, araises], ids=['sync', 'async'])
@pytest.mark.parametrize('exception_cls', [BaseException, KeyboardInterrupt, SystemExit])
def test_sync_waiter_base_exception_in_worker_thread(exception_cls, raise_fn):
    if False:
        for i in range(10):
            print('nop')
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:
        call = Call.new(raise_fn, exception_cls('test'))
        waiter = SyncWaiter(call)
        waiter.add_done_callback(done_callback)
        runner.submit(call)
        waiter.wait()
    with pytest.raises(exception_cls, match='test'):
        call.result()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on exception'

@pytest.mark.parametrize('raise_fn', [raises, araises], ids=['sync', 'async'])
@pytest.mark.parametrize('exception_cls', [BaseException, KeyboardInterrupt, SystemExit])
def test_sync_waiter_base_exception_in_main_thread(exception_cls, raise_fn):
    if False:
        print('Hello World!')
    done_callback = Call.new(identity, 1)
    with WorkerThread(run_once=True) as runner:

        def on_worker_thread():
            if False:
                while True:
                    i = 10
            callback = Call.new(raise_fn, exception_cls('test'))
            waiter.submit(callback)
            return callback
        call = Call.new(on_worker_thread)
        waiter = SyncWaiter(call)
        waiter.add_done_callback(done_callback)
        runner.submit(call)
        callback = waiter.wait().result()
    with pytest.raises(exception_cls, match='test'):
        callback.result()
    assert done_callback.result(timeout=0) == 1, 'The done callback should still be called on exception'