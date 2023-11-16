__author__ = 'maartenbreddels'
import asyncio
import atexit
import traceback
import sys
import aplus
from aplus import listPromise
try:
    import contextvars
    has_contextvars = True
    auto_await_executor = contextvars.ContextVar('auto await executor', default=None)
except ImportError:
    has_contextvars = False

def check_unhandled():
    if False:
        print('Hello World!')
    if Promise.unhandled_exceptions:
        print('Unhandled exceptions in Promises:')
        for (exctype, value, tb) in Promise.unhandled_exceptions:
            traceback.print_exception(exctype, value, tb)

def rereaise_unhandled():
    if False:
        i = 10
        return i + 15
    if Promise.unhandled_exceptions:
        for (exctype, value, tb) in Promise.unhandled_exceptions:
            if value:
                raise value.with_traceback(tb)
atexit.register(check_unhandled)

class Promise(aplus.Promise):
    last_exc_info = None
    unhandled_exceptions = []

    @staticmethod
    def create():
        if False:
            for i in range(10):
                print('nop')
        return Promise()

    def create_next(self):
        if False:
            print('Hello World!')
        return Promise()

    @classmethod
    def fulfilled(cls, x):
        if False:
            print('Hello World!')
        p = cls.create()
        p.fulfill(x)
        return p

    @classmethod
    def rejected(cls, reason):
        if False:
            return 10
        p = cls.create()
        p.reject(reason)
        return p

    @staticmethod
    def unhandled(exctype, value, traceback):
        if False:
            while True:
                i = 10
        Promise.unhandled_exceptions.append((exctype, value, traceback))

    def then(self, success=None, failure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method takes two optional arguments.  The first argument\n        is used if the "self promise" is fulfilled and the other is\n        used if the "self promise" is rejected.  In either case, this\n        method returns another promise that effectively represents\n        the result of either the first of the second argument (in the\n        case that the "self promise" is fulfilled or rejected,\n        respectively).\n\n        Each argument can be either:\n          * None - Meaning no action is taken\n          * A function - which will be called with either the value\n                of the "self promise" or the reason for rejection of\n                the "self promise".  The function may return:\n                * A value - which will be used to fulfill the promise\n                  returned by this method.\n                * A promise - which, when fulfilled or rejected, will\n                  cascade its value or reason to the promise returned\n                  by this method.\n          * A value - which will be assigned as either the value\n                or the reason for the promise returned by this method\n                when the "self promise" is either fulfilled or rejected,\n                respectively.\n\n        :type success: (object) -> object\n        :type failure: (object) -> object\n        :rtype : Promise\n        '
        ret = self.create_next()

        def callAndFulfill(v):
            if False:
                i = 10
                return i + 15
            '\n            A callback to be invoked if the "self promise"\n            is fulfilled.\n            '
            try:
                if aplus._isFunction(success):
                    ret.fulfill(success(v))
                else:
                    ret.fulfill(v)
            except Exception as e:
                Promise.last_exc_info = sys.exc_info()
                e.exc_info = sys.exc_info()
                ret.reject(e)

        def callAndReject(r):
            if False:
                while True:
                    i = 10
            '\n            A callback to be invoked if the "self promise"\n            is rejected.\n            '
            try:
                if aplus._isFunction(failure):
                    ret.fulfill(failure(r))
                else:
                    ret.reject(r)
            except Exception as e:
                Promise.last_exc_info = sys.exc_info()
                e.exc_info = sys.exc_info()
                ret.reject(e)
        self.done(callAndFulfill, callAndReject)
        return ret

    def end(self):
        if False:
            for i in range(10):
                print('nop')

        def failure(reason):
            if False:
                while True:
                    i = 10
            args = sys.exc_info()
            if args is None or args[0] is None:
                args = Promise.last_exc_info
            if hasattr(reason, 'exc_info'):
                args = reason.exc_info
            try:
                Promise.unhandled(*args)
            except:
                print('Error in unhandled handler')
                traceback.print_exc()
        return self.then(None, failure)

    def __await__(self):
        if False:
            i = 10
            return i + 15
        return self._create_awaitable().__await__()

    async def _create_awaitable(self):
        future = asyncio.Future()
        self.then(future.set_result, future.set_exception)
        if has_contextvars:
            executor = auto_await_executor.get()
            if executor:
                result = await asyncio.gather(executor.execute_async(), future)
                return result[-1]
        return await future
aplus.Promise = Promise