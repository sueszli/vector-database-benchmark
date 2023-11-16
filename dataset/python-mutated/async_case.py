import asyncio
import inspect
from .case import TestCase

class IsolatedAsyncioTestCase(TestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        super().__init__(methodName)
        self._asyncioTestLoop = None
        self._asyncioCallsQueue = None

    async def asyncSetUp(self):
        pass

    async def asyncTearDown(self):
        pass

    def addAsyncCleanup(self, func, /, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.addCleanup(*(func, *args), **kwargs)

    def _callSetUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUp()
        self._callAsync(self.asyncSetUp)

    def _callTestMethod(self, method):
        if False:
            return 10
        self._callMaybeAsync(method)

    def _callTearDown(self):
        if False:
            while True:
                i = 10
        self._callAsync(self.asyncTearDown)
        self.tearDown()

    def _callCleanup(self, function, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._callMaybeAsync(function, *args, **kwargs)

    def _callAsync(self, func, /, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        ret = func(*args, **kwargs)
        assert inspect.isawaitable(ret), f'{func!r} returned non-awaitable'
        fut = self._asyncioTestLoop.create_future()
        self._asyncioCallsQueue.put_nowait((fut, ret))
        return self._asyncioTestLoop.run_until_complete(fut)

    def _callMaybeAsync(self, func, /, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        ret = func(*args, **kwargs)
        if inspect.isawaitable(ret):
            fut = self._asyncioTestLoop.create_future()
            self._asyncioCallsQueue.put_nowait((fut, ret))
            return self._asyncioTestLoop.run_until_complete(fut)
        else:
            return ret

    async def _asyncioLoopRunner(self, fut):
        self._asyncioCallsQueue = queue = asyncio.Queue()
        fut.set_result(None)
        while True:
            query = await queue.get()
            queue.task_done()
            if query is None:
                return
            (fut, awaitable) = query
            try:
                ret = await awaitable
                if not fut.cancelled():
                    fut.set_result(ret)
            except (SystemExit, KeyboardInterrupt):
                raise
            except (BaseException, asyncio.CancelledError) as ex:
                if not fut.cancelled():
                    fut.set_exception(ex)

    def _setupAsyncioLoop(self):
        if False:
            print('Hello World!')
        assert self._asyncioTestLoop is None, 'asyncio test loop already initialized'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(True)
        self._asyncioTestLoop = loop
        fut = loop.create_future()
        self._asyncioCallsTask = loop.create_task(self._asyncioLoopRunner(fut))
        loop.run_until_complete(fut)

    def _tearDownAsyncioLoop(self):
        if False:
            i = 10
            return i + 15
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        loop = self._asyncioTestLoop
        self._asyncioTestLoop = None
        self._asyncioCallsQueue.put_nowait(None)
        loop.run_until_complete(self._asyncioCallsQueue.join())
        try:
            to_cancel = asyncio.all_tasks(loop)
            if not to_cancel:
                return
            for task in to_cancel:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))
            for task in to_cancel:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    loop.call_exception_handler({'message': 'unhandled exception during test shutdown', 'exception': task.exception(), 'task': task})
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.run_until_complete(loop.shutdown_default_executor())
            asyncio.set_event_loop(None)
            loop.close()

    def run(self, result=None):
        if False:
            return 10
        self._setupAsyncioLoop()
        try:
            return super().run(result)
        finally:
            self._tearDownAsyncioLoop()

    def debug(self):
        if False:
            i = 10
            return i + 15
        self._setupAsyncioLoop()
        super().debug()
        self._tearDownAsyncioLoop()

    def __del__(self):
        if False:
            print('Hello World!')
        if self._asyncioTestLoop is not None:
            self._tearDownAsyncioLoop()