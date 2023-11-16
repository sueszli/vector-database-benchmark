"""
This module provides functionality to support "async steps" (coroutines)
in a step-module with behave. This functionality simplifies to test
frameworks and protocols that make use of `asyncio.coroutines`_ or
provide `asyncio.coroutines`_.

EXAMPLE:

.. code-block:: python

    # -- FILE: features/steps/my_async_steps.py
    # EXAMPLE REQUIRES: Python >= 3.5
    from behave import step
    from behave.api.async_step import async_run_until_complete

    @step('an async coroutine step waits {duration:f} seconds')
    @async_run_until_complete
    async def step_async_step_waits_seconds(context, duration):
        await asyncio.sleep(duration)

.. code-block:: python

    # -- FILE: features/steps/my_async_steps2.py
    # EXAMPLE REQUIRES: Python >= 3.4
    from behave import step
    from behave.api.async_step import async_run_until_complete
    import asyncio

    @step('a tagged-coroutine async step waits {duration:f} seconds')
    @async_run_until_complete
    @asyncio.coroutine
    def step_async_step_waits_seconds2(context, duration):
        yield from asyncio.sleep(duration)


.. requires:: Python 3.5 (or 3.4) or :mod:`asyncio` backport (like :pypi:`trollius`)
.. seealso::
    https://docs.python.org/3/library/asyncio.html

.. _asyncio.coroutines: https://docs.python.org/3/library/asyncio-task.html#coroutines
"""
from __future__ import print_function
import functools
import sys
from six import string_types
try:
    import asyncio
    has_asyncio = True
except ImportError:
    has_asyncio = False
_PYTHON_VERSION = sys.version_info[:2]

def async_run_until_complete(astep_func=None, loop=None, timeout=None, async_context=None, should_close=False):
    if False:
        return 10
    'Provides a function decorator for async-steps (coroutines).\n    Provides an async event loop and runs the async-step until completion\n    (or timeout, if specified).\n\n    .. code-block:: python\n\n        from behave import step\n        from behave.api.async_step import async_run_until_complete\n        import asyncio\n\n        @step("an async step is executed")\n        @async_run_until_complete\n        async def astep_impl(context)\n            await asycio.sleep(0.1)\n\n        @step("an async step is executed")\n        @async_run_until_complete(timeout=1.2)\n        async def astep_impl2(context)\n            # -- NOTE: Wrapped event loop waits with timeout=1.2 seconds.\n            await asycio.sleep(0.3)\n\n    Parameters:\n        astep_func: Async step function (coroutine)\n        loop (asyncio.EventLoop):   Event loop to use or None.\n        timeout (int, float):       Timeout to wait for async-step completion.\n        async_context (name):       Async_context name or object to use.\n        should_close (bool):        Indicates if event loop should be closed.\n\n    .. note::\n\n        * If :param:`loop` is None, the default event loop will be used\n          or a new event loop is created.\n        * If :param:`timeout` is provided, the event loop waits only the\n          specified time.\n        * :param:`async_context` is only used, if :param:`loop` is None.\n        * If :param:`async_context` is a name, it will be used to retrieve\n          the real async_context object from the context.\n\n    '

    @functools.wraps(astep_func)
    def step_decorator(astep_func, context, *args, **kwargs):
        if False:
            print('Hello World!')
        loop = kwargs.pop('_loop', None)
        timeout = kwargs.pop('_timeout', None)
        async_context = kwargs.pop('_async_context', None)
        should_close = kwargs.pop('_should_close', None)
        if isinstance(loop, string_types):
            loop = getattr(context, loop, None)
        elif async_context:
            if isinstance(async_context, string_types):
                name = async_context
                async_context = use_or_create_async_context(context, name)
                loop = async_context.loop
            else:
                assert isinstance(async_context, AsyncContext)
                loop = async_context.loop
        if loop is None:
            if _PYTHON_VERSION < (3, 10):
                loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                should_close = True
        try:
            if timeout is None:
                loop.run_until_complete(astep_func(context, *args, **kwargs))
            else:
                task = loop.create_task(astep_func(context, *args, **kwargs))
                (done, pending) = loop.run_until_complete(asyncio.wait([task], timeout=timeout))
                assert not pending, 'TIMEOUT-OCCURED: timeout=%s' % timeout
                finished_task = done.pop()
                exception = finished_task.exception()
                if exception:
                    raise exception
        finally:
            if loop and should_close:
                loop.close()
    if astep_func is None:

        def wrapped_decorator1(astep_func):
            if False:
                return 10

            @functools.wraps(astep_func)
            def wrapped_decorator2(context, *args, **kwargs):
                if False:
                    return 10
                return step_decorator(astep_func, context, *args, _loop=loop, _timeout=timeout, _async_context=async_context, _should_close=should_close, **kwargs)
            assert callable(astep_func)
            return wrapped_decorator2
        return wrapped_decorator1
    else:
        assert callable(astep_func)

        @functools.wraps(astep_func)
        def wrapped_decorator(context, *args, **kwargs):
            if False:
                print('Hello World!')
            return step_decorator(astep_func, context, *args, **kwargs)
        return wrapped_decorator
run_until_complete = async_run_until_complete

class AsyncContext(object):
    """Provides a context object for "async steps" to keep track:

    * which event loop is used
    * which (asyncio) tasks are used or of interest

    .. attribute:: loop
        Event loop object to use.
        If none is provided, the current event-loop is used
        (or a new one is created).

    .. attribute:: tasks
        List of started :mod:`asyncio` tasks (of interest).

    .. attribute:: name

        Optional name of this object (in the behave context).
        If none is provided, :attr:`AsyncContext.default_name` is used.

    .. attribute:: should_close
        Indicates if the :attr:`loop` (event-loop) should be closed or not.

    EXAMPLE:

    .. code-block:: python

        # -- FILE: features/steps/my_async_steps.py
        # REQUIRES: Python 3.5
        from behave import given, when, then, step
        from behave.api.async_step import AsyncContext

        @when('I dispatch an async-call with param "{param}"')
        def step_impl(context, param):
            async_context = getattr(context, "async_context", None)
            if async_context is None:
                async_context = context.async_context = AsyncContext()
            task = async_context.loop.create_task(my_async_func(param))
            async_context.tasks.append(task)

        @then('I wait at most {duration:f} seconds until all async-calls are completed')
        def step_impl(context, duration):
            async_context = context.async_context
            assert async_context.tasks
            done, pending = async_context.loop.run_until_complete(asyncio.wait(
                async_context.tasks, loop=async_context.loop, timeout=duration))
            assert len(pending) == 0

        # -- COROUTINE:
        async def my_async_func(param):
            await asyncio.sleep(0.5)
            return param.upper()
    """
    default_name = 'async_context'

    def __init__(self, loop=None, name=None, should_close=False, tasks=None):
        if False:
            while True:
                i = 10
        self.loop = use_or_create_event_loop(loop)
        self.tasks = tasks or []
        self.name = name or self.default_name
        self.should_close = should_close

    def __del__(self):
        if False:
            print('Hello World!')
        if self.loop and self.should_close:
            self.close()

    def close(self):
        if False:
            print('Hello World!')
        if self.loop and (not self.loop.is_closed()):
            self.loop.close()
            self.loop = None

def use_or_create_event_loop(loop=None):
    if False:
        return 10
    if loop:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()
    except AttributeError:
        return asyncio.get_event_loop()

def use_or_create_async_context(context, name=None, loop=None, **kwargs):
    if False:
        while True:
            i = 10
    'Utility function to be used in step implementations to ensure that an\n    :class:`AsyncContext` object is stored in the :param:`context` object.\n\n    If no such attribute exists (under the given name),\n    a new :class:`AsyncContext` object is created with the provided args.\n    Otherwise, the existing context attribute is used.\n\n    EXAMPLE:\n\n    .. code-block:: python\n\n        # -- FILE: features/steps/my_async_steps.py\n        # EXAMPLE REQUIRES: Python 3.5\n        from behave import when\n        from behave.api.async_step import use_or_create_async_context\n\n        @when(\'I dispatch an async-call with param "{param}"\')\n        def step_impl(context, param):\n            async_context = use_or_create_async_context(context, "async_context")\n            task = async_context.loop.create_task(my_async_func(param))\n            async_context.tasks.append(task)\n\n        # -- COROUTINE:\n        async def my_async_func(param):\n            await asyncio.sleep(0.5)\n            return param.upper()\n\n    :param context:     Behave context object to use.\n    :param name:        Optional name of async-context object (as string or None).\n    :param loop:        Optional event_loop object to use for create call.\n    :param kwargs:      Optional :class:`AsyncContext` params for create call.\n    :return: :class:`AsyncContext` object from the param:`context`.\n    '
    if name is None:
        name = AsyncContext.default_name
    async_context = getattr(context, name, None)
    if async_context is None:
        async_context = AsyncContext(loop=loop, name=name, **kwargs)
        setattr(context, async_context.name, async_context)
    assert isinstance(async_context, AsyncContext)
    assert getattr(context, async_context.name) is async_context
    return async_context