import asyncio
import logging
import weakref
from ._asyncio_loop import get_running_loop, get_task_loop

class StreamSink:

    def __init__(self, stream):
        if False:
            return 10
        self._stream = stream
        self._flushable = callable(getattr(stream, 'flush', None))
        self._stoppable = callable(getattr(stream, 'stop', None))
        self._completable = asyncio.iscoroutinefunction(getattr(stream, 'complete', None))

    def write(self, message):
        if False:
            print('Hello World!')
        self._stream.write(message)
        if self._flushable:
            self._stream.flush()

    def stop(self):
        if False:
            print('Hello World!')
        if self._stoppable:
            self._stream.stop()

    def tasks_to_complete(self):
        if False:
            i = 10
            return i + 15
        if not self._completable:
            return []
        return [self._stream.complete()]

class StandardSink:

    def __init__(self, handler):
        if False:
            i = 10
            return i + 15
        self._handler = handler

    def write(self, message):
        if False:
            for i in range(10):
                print('nop')
        record = message.record
        message = str(message)
        exc = record['exception']
        record = logging.getLogger().makeRecord(record['name'], record['level'].no, record['file'].path, record['line'], message, (), (exc.type, exc.value, exc.traceback) if exc else None, record['function'], {'extra': record['extra']})
        if exc:
            record.exc_text = '\n'
        self._handler.handle(record)

    def stop(self):
        if False:
            i = 10
            return i + 15
        self._handler.close()

    def tasks_to_complete(self):
        if False:
            while True:
                i = 10
        return []

class AsyncSink:

    def __init__(self, function, loop, error_interceptor):
        if False:
            for i in range(10):
                print('nop')
        self._function = function
        self._loop = loop
        self._error_interceptor = error_interceptor
        self._tasks = weakref.WeakSet()

    def write(self, message):
        if False:
            print('Hello World!')
        try:
            loop = self._loop or get_running_loop()
        except RuntimeError:
            return
        coroutine = self._function(message)
        task = loop.create_task(coroutine)

        def check_exception(future):
            if False:
                for i in range(10):
                    print('nop')
            if future.cancelled() or future.exception() is None:
                return
            if not self._error_interceptor.should_catch():
                raise future.exception()
            self._error_interceptor.print(message.record, exception=future.exception())
        task.add_done_callback(check_exception)
        self._tasks.add(task)

    def stop(self):
        if False:
            return 10
        for task in self._tasks:
            task.cancel()

    def tasks_to_complete(self):
        if False:
            i = 10
            return i + 15
        return [self._complete_task(task) for task in self._tasks]

    async def _complete_task(self, task):
        loop = get_running_loop()
        if get_task_loop(task) is not loop:
            return
        try:
            await task
        except Exception:
            pass

    def __getstate__(self):
        if False:
            return 10
        state = self.__dict__.copy()
        state['_tasks'] = None
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(state)
        self._tasks = weakref.WeakSet()

class CallableSink:

    def __init__(self, function):
        if False:
            for i in range(10):
                print('nop')
        self._function = function

    def write(self, message):
        if False:
            return 10
        self._function(message)

    def stop(self):
        if False:
            i = 10
            return i + 15
        pass

    def tasks_to_complete(self):
        if False:
            print('Hello World!')
        return []