"""Abortable Tasks.

Abortable tasks overview
=========================

For long-running :class:`Task`'s, it can be desirable to support
aborting during execution.  Of course, these tasks should be built to
support abortion specifically.

The :class:`AbortableTask` serves as a base class for all :class:`Task`
objects that should support abortion by producers.

* Producers may invoke the :meth:`abort` method on
  :class:`AbortableAsyncResult` instances, to request abortion.

* Consumers (workers) should periodically check (and honor!) the
  :meth:`is_aborted` method at controlled points in their task's
  :meth:`run` method.  The more often, the better.

The necessary intermediate communication is dealt with by the
:class:`AbortableTask` implementation.

Usage example
-------------

In the consumer:

.. code-block:: python

    from celery.contrib.abortable import AbortableTask
    from celery.utils.log import get_task_logger

    from proj.celery import app

    logger = get_logger(__name__)

    @app.task(bind=True, base=AbortableTask)
    def long_running_task(self):
        results = []
        for i in range(100):
            # check after every 5 iterations...
            # (or alternatively, check when some timer is due)
            if not i % 5:
                if self.is_aborted():
                    # respect aborted state, and terminate gracefully.
                    logger.warning('Task aborted')
                    return
                value = do_something_expensive(i)
                results.append(y)
        logger.info('Task complete')
        return results

In the producer:

.. code-block:: python

    import time

    from proj.tasks import MyLongRunningTask

    def myview(request):
        # result is of type AbortableAsyncResult
        result = long_running_task.delay()

        # abort the task after 10 seconds
        time.sleep(10)
        result.abort()

After the `result.abort()` call, the task execution isn't
aborted immediately.  In fact, it's not guaranteed to abort at all.
Keep checking `result.state` status, or call `result.get(timeout=)` to
have it block until the task is finished.

.. note::

   In order to abort tasks, there needs to be communication between the
   producer and the consumer.  This is currently implemented through the
   database backend.  Therefore, this class will only work with the
   database backends.
"""
from celery import Task
from celery.result import AsyncResult
__all__ = ('AbortableAsyncResult', 'AbortableTask')
'\nTask States\n-----------\n\n.. state:: ABORTED\n\nABORTED\n~~~~~~~\n\nTask is aborted (typically by the producer) and should be\naborted as soon as possible.\n\n'
ABORTED = 'ABORTED'

class AbortableAsyncResult(AsyncResult):
    """Represents an abortable result.

    Specifically, this gives the `AsyncResult` a :meth:`abort()` method,
    that sets the state of the underlying Task to `'ABORTED'`.
    """

    def is_aborted(self):
        if False:
            for i in range(10):
                print('nop')
        'Return :const:`True` if the task is (being) aborted.'
        return self.state == ABORTED

    def abort(self):
        if False:
            while True:
                i = 10
        'Set the state of the task to :const:`ABORTED`.\n\n        Abortable tasks monitor their state at regular intervals and\n        terminate execution if so.\n\n        Warning:\n            Be aware that invoking this method does not guarantee when the\n            task will be aborted (or even if the task will be aborted at all).\n        '
        return self.backend.store_result(self.id, result=None, state=ABORTED, traceback=None)

class AbortableTask(Task):
    """Task that can be aborted.

    This serves as a base class for all :class:`Task`'s
    that support aborting during execution.

    All subclasses of :class:`AbortableTask` must call the
    :meth:`is_aborted` method periodically and act accordingly when
    the call evaluates to :const:`True`.
    """
    abstract = True

    def AsyncResult(self, task_id):
        if False:
            print('Hello World!')
        'Return the accompanying AbortableAsyncResult instance.'
        return AbortableAsyncResult(task_id, backend=self.backend)

    def is_aborted(self, **kwargs):
        if False:
            while True:
                i = 10
        'Return true if task is aborted.\n\n        Checks against the backend whether this\n        :class:`AbortableAsyncResult` is :const:`ABORTED`.\n\n        Always return :const:`False` in case the `task_id` parameter\n        refers to a regular (non-abortable) :class:`Task`.\n\n        Be aware that invoking this method will cause a hit in the\n        backend (for example a database query), so find a good balance\n        between calling it regularly (for responsiveness), but not too\n        often (for performance).\n        '
        task_id = kwargs.get('task_id', self.request.id)
        result = self.AsyncResult(task_id)
        if not isinstance(result, AbortableAsyncResult):
            return False
        return result.is_aborted()