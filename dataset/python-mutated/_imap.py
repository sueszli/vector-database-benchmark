"""
Iterators across greenlets or AsyncResult objects.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gevent import lock
from gevent import queue
__all__ = ['IMapUnordered', 'IMap']
locals()['Greenlet'] = __import__('gevent').Greenlet
locals()['Semaphore'] = lock.Semaphore
locals()['UnboundQueue'] = queue.UnboundQueue

class Failure(object):
    __slots__ = ('exc', 'raise_exception')

    def __init__(self, exc, raise_exception=None):
        if False:
            for i in range(10):
                print('nop')
        self.exc = exc
        self.raise_exception = raise_exception

def _raise_exc(failure):
    if False:
        i = 10
        return i + 15
    if failure.raise_exception:
        failure.raise_exception()
    else:
        raise failure.exc

class IMapUnordered(Greenlet):
    """
    At iterator of map results.
    """

    def __init__(self, func, iterable, spawn, maxsize=None, _zipped=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        An iterator that.\n\n        :param callable spawn: The function we use to create new greenlets.\n        :keyword int maxsize: If given and not-None, specifies the maximum number of\n            finished results that will be allowed to accumulated awaiting the reader;\n            more than that number of results will cause map function greenlets to begin\n            to block. This is most useful is there is a great disparity in the speed of\n            the mapping code and the consumer and the results consume a great deal of resources.\n            Using a bound is more computationally expensive than not using a bound.\n\n        .. versionchanged:: 1.1b3\n            Added the *maxsize* parameter.\n        '
        Greenlet.__init__(self)
        self.spawn = spawn
        self._zipped = _zipped
        self.func = func
        self.iterable = iterable
        self.queue = UnboundQueue()
        if maxsize:
            self._result_semaphore = Semaphore(maxsize)
        else:
            self._result_semaphore = None
        self._outstanding_tasks = 0
        self._max_index = -1
        self.finished = False

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self._result_semaphore is not None:
            self._result_semaphore.release()
        value = self._inext()
        if isinstance(value, Failure):
            _raise_exc(value)
        return value
    next = __next__

    def _inext(self):
        if False:
            i = 10
            return i + 15
        return self.queue.get()

    def _ispawn(self, func, item, item_index):
        if False:
            for i in range(10):
                print('nop')
        if self._result_semaphore is not None:
            self._result_semaphore.acquire()
        self._outstanding_tasks += 1
        g = self.spawn(func, item) if not self._zipped else self.spawn(func, *item)
        g._imap_task_index = item_index
        g.rawlink(self._on_result)
        return g

    def _run(self):
        if False:
            i = 10
            return i + 15
        try:
            func = self.func
            for item in self.iterable:
                self._max_index += 1
                self._ispawn(func, item, self._max_index)
            self._on_finish(None)
        except BaseException as e:
            self._on_finish(e)
            raise
        finally:
            self.spawn = None
            self.func = None
            self.iterable = None
            self._result_semaphore = None

    def _on_result(self, greenlet):
        if False:
            for i in range(10):
                print('nop')
        self._outstanding_tasks -= 1
        count = self._outstanding_tasks
        finished = self.finished
        ready = self.ready()
        put_finished = False
        if ready and count <= 0 and (not finished):
            finished = self.finished = True
            put_finished = True
        if greenlet.successful():
            self.queue.put(self._iqueue_value_for_success(greenlet))
        else:
            self.queue.put(self._iqueue_value_for_failure(greenlet))
        if put_finished:
            self.queue.put(self._iqueue_value_for_self_finished())

    def _on_finish(self, exception):
        if False:
            return 10
        if self.finished:
            return
        if exception is not None:
            self.finished = True
            self.queue.put(self._iqueue_value_for_self_failure(exception))
            return
        if self._outstanding_tasks <= 0:
            self.finished = True
            self.queue.put(self._iqueue_value_for_self_finished())

    def _iqueue_value_for_success(self, greenlet):
        if False:
            print('Hello World!')
        return greenlet.value

    def _iqueue_value_for_failure(self, greenlet):
        if False:
            return 10
        return Failure(greenlet.exception, getattr(greenlet, '_raise_exception'))

    def _iqueue_value_for_self_finished(self):
        if False:
            while True:
                i = 10
        return Failure(StopIteration())

    def _iqueue_value_for_self_failure(self, exception):
        if False:
            i = 10
            return i + 15
        return Failure(exception, self._raise_exception)

class IMap(IMapUnordered):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._results = {}
        self.index = 0
        IMapUnordered.__init__(self, *args, **kwargs)

    def _inext(self):
        if False:
            while True:
                i = 10
        try:
            value = self._results.pop(self.index)
        except KeyError:
            while 1:
                (index, value) = self.queue.get()
                if index == self.index:
                    break
                self._results[index] = value
        self.index += 1
        return value

    def _iqueue_value_for_success(self, greenlet):
        if False:
            i = 10
            return i + 15
        return (greenlet._imap_task_index, IMapUnordered._iqueue_value_for_success(self, greenlet))

    def _iqueue_value_for_failure(self, greenlet):
        if False:
            for i in range(10):
                print('nop')
        return (greenlet._imap_task_index, IMapUnordered._iqueue_value_for_failure(self, greenlet))

    def _iqueue_value_for_self_finished(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._max_index + 1, IMapUnordered._iqueue_value_for_self_finished(self))

    def _iqueue_value_for_self_failure(self, exception):
        if False:
            while True:
                i = 10
        return (self._max_index + 1, IMapUnordered._iqueue_value_for_self_failure(self, exception))
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__imap')