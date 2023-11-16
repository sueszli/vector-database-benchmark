from py4j.protocol import Py4JJavaError
from pyflink.util.exceptions import convert_py4j_exception
__all__ = ['CompletableFuture']

class CompletableFuture(object):
    """
    A Future that may be explicitly completed (setting its value and status), supporting dependent
    functions and actions that trigger upon its completion.

    When two or more threads attempt to set_result, set_exception, or cancel a CompletableFuture,
    only one of them succeeds.

    .. versionadded:: 1.11.0
    """

    def __init__(self, j_completable_future, py_class=None):
        if False:
            while True:
                i = 10
        self._j_completable_future = j_completable_future
        self._py_class = py_class

    def cancel(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Completes this CompletableFuture if not already completed.\n\n        :return: true if this task is now cancelled\n\n        .. versionadded:: 1.11.0\n        '
        return self._j_completable_future.cancel(True)

    def cancelled(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns true if this CompletableFuture was cancelled before it completed normally.\n\n        .. versionadded:: 1.11.0\n        '
        return self._j_completable_future.isCancelled()

    def done(self) -> bool:
        if False:
            return 10
        '\n        Returns true if completed in any fashion: normally, exceptionally, or via cancellation.\n\n        .. versionadded:: 1.11.0\n        '
        return self._j_completable_future.isDone()

    def result(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Waits if necessary for this future to complete, and then returns its result.\n\n        :return: the result value\n\n        .. versionadded:: 1.11.0\n        '
        if self._py_class is None:
            return self._j_completable_future.get()
        else:
            return self._py_class(self._j_completable_future.get())

    def exception(self):
        if False:
            return 10
        '\n        Returns the exception that was set on this future or None if no exception was set.\n\n        .. versionadded:: 1.11.0\n        '
        if self._j_completable_future.isCompletedExceptionally():
            try:
                self._j_completable_future.getNow(None)
            except Py4JJavaError as e:
                return convert_py4j_exception(e)
        else:
            return None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._j_completable_future.toString()