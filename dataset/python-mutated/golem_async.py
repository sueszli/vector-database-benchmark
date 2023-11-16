import asyncio
import concurrent.futures
import datetime
import functools
import logging
from typing import Any, Callable, Dict, Optional
from twisted.internet import defer
from twisted.internet import threads
from twisted.web.iweb import IBodyProducer
from zope.interface import implementer
logger = logging.getLogger(__name__)

class AsyncHTTPRequest:
    agent = None
    timeout = 5

    @implementer(IBodyProducer)
    class BytesBodyProducer:

        def __init__(self, body):
            if False:
                i = 10
                return i + 15
            self.body = body
            self.length = len(body)

        def startProducing(self, consumer):
            if False:
                for i in range(10):
                    print('nop')
            consumer.write(self.body)
            return defer.succeed(None)

        def pauseProducing(self):
            if False:
                return 10
            pass

        def resumeProducing(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def stopProducing(self):
            if False:
                return 10
            pass

    @classmethod
    def run(cls, method, uri, headers, body):
        if False:
            i = 10
            return i + 15
        if not cls.agent:
            cls.agent = cls.create_agent()
        return cls.agent.request(method, uri, headers, cls.BytesBodyProducer(body))

    @classmethod
    def create_agent(cls):
        if False:
            print('Hello World!')
        from twisted.internet import reactor
        from twisted.web.client import Agent
        return Agent(reactor, connectTimeout=cls.timeout)

class AsyncRequest(object):
    """ Deferred job descriptor """

    def __init__(self, method, *args, **kwargs):
        if False:
            print('Hello World!')
        self.method = method
        self.args = args or []
        self.kwargs = kwargs or {}

def async_run(deferred_call: AsyncRequest, success: Optional[Callable]=None, error: Optional[Callable]=None):
    if False:
        print('Hello World!')
    'Execute a deferred job in a separate thread (Twisted)'
    deferred = threads.deferToThread(deferred_call.method, *deferred_call.args, **deferred_call.kwargs)
    if error is None:
        error = default_errback
    if success:
        deferred.addCallback(success)
    deferred.addErrback(error)
    return deferred

def default_errback(failure):
    if False:
        return 10
    logger.error('Caught async exception:\n%s', failure.getTraceback())
    return failure

def deferred_run():
    if False:
        for i in range(10):
            print('nop')

    def wrapped(f):
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        def curry(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            from twisted.internet import reactor
            if reactor.running:
                execute = threads.deferToThread
            else:
                logger.debug('Reactor not running. Switching to blocking call for %r', f)
                execute = defer.execute
            return execute(f, *args, **kwargs)
        return curry
    return wrapped
_ASYNCIO_THREAD_POOL = concurrent.futures.ThreadPoolExecutor()

def get_event_loop():
    if False:
        while True:
            i = 10
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        from twisted.internet import reactor
        return reactor._asyncioEventloop

def soon():
    if False:
        i = 10
        return i + 15
    'Run non-async function in next iteration of event loop'

    def wrapped(f):
        if False:
            return 10

        @functools.wraps(f)
        def curry(*args, **kwargs):
            if False:
                return 10
            loop = get_event_loop()
            loop.call_soon_threadsafe(functools.partial(f, *args, **kwargs))
            return None
        return curry
    return wrapped

def taskify():
    if False:
        for i in range(10):
            print('nop')
    'Run async function as a Task in current loop'

    def wrapped(f):
        if False:
            while True:
                i = 10
        assert asyncio.iscoroutinefunction(f)

        @functools.wraps(f)
        def curry(*args, **kwargs):
            if False:
                print('Hello World!')
            task = asyncio.ensure_future(f(*args, **kwargs), loop=get_event_loop())
            return task
        return curry
    return wrapped

def throttle(delta: datetime.timedelta):
    if False:
        print('Hello World!')
    'Invoke the decorated function only once per `delta`\n\n    All subsequent call will be dropped until delta passes.\n    '
    last_run = datetime.datetime.min

    def wrapped(f):
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        async def curry(*args, **kwargs):
            nonlocal last_run
            current_delta = datetime.datetime.now() - last_run
            if current_delta < delta:
                return
            last_run = datetime.datetime.now()
            result = f(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return curry
    return wrapped

def run_in_thread():
    if False:
        i = 10
        return i + 15

    def wrapped(f):
        if False:
            for i in range(10):
                print('nop')
        assert not asyncio.iscoroutinefunction(f)

        @functools.wraps(f)
        async def curry(*args, **kwargs):
            loop = get_event_loop()
            return await loop.run_in_executor(executor=_ASYNCIO_THREAD_POOL, func=functools.partial(f, *args, **kwargs, loop=loop))
        return curry
    return wrapped

class CallScheduler:

    def __init__(self):
        if False:
            return 10
        self._timers: Dict[str, asyncio.TimerHandle] = dict()

    def schedule(self, key: str, timeout: float, call: Callable[..., Any]) -> None:
        if False:
            print('Hello World!')

        def on_timeout():
            if False:
                for i in range(10):
                    print('nop')
            self._timers.pop(key, None)
            call()
        loop = asyncio.get_event_loop()
        self.cancel(key)
        self._timers[key] = loop.call_at(loop.time() + timeout, on_timeout)

    def cancel(self, key):
        if False:
            print('Hello World!')
        timer = self._timers.pop(key, None)
        if timer:
            timer.cancel()