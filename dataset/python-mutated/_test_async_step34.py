"""
Unit tests for :mod:`behave.api.async_test`.
"""
from __future__ import absolute_import, print_function
import sys
from unittest.mock import Mock
from hamcrest import assert_that, close_to
import pytest
from behave.api.async_step import AsyncContext, use_or_create_async_context
from behave._stepimport import use_step_import_modules, SimpleStepContainer
from behave.runner import Context, Runner
from .testing_support import StopWatch
from .testing_support_async import AsyncStepTheory
PYTHON_3_5 = (3, 5)
PYTHON_3_8 = (3, 8)
python_version = sys.version_info[:2]
requires_py34_to_py37 = pytest.mark.skipif(not PYTHON_3_5 <= python_version < PYTHON_3_8, reason='Supported only for python.versions: 3.4 .. 3.7 (inclusive)')
SLEEP_DELTA = 0.05
if sys.platform.startswith('win'):
    SLEEP_DELTA = 0.1

@requires_py34_to_py37
class TestAsyncStepDecoratorPy34:

    def test_step_decorator_async_run_until_complete2(self):
        if False:
            for i in range(10):
                print('nop')
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import step
            from behave.api.async_step import async_run_until_complete
            import asyncio

            @step('a tagged-coroutine async step waits "{duration:f}" seconds')
            @async_run_until_complete
            @asyncio.coroutine
            def step_async_step_waits_seconds2(context, duration):
                if False:
                    return 10
                yield from asyncio.sleep(duration)
        AsyncStepTheory.validate(step_async_step_waits_seconds2)
        context = Context(runner=Runner(config={}))
        with StopWatch() as stop_watch:
            step_async_step_waits_seconds2(context, duration=0.2)
        assert_that(stop_watch.duration, close_to(0.2, delta=SLEEP_DELTA))

class TestAsyncContext:

    @staticmethod
    def make_context():
        if False:
            for i in range(10):
                print('nop')
        return Context(runner=Runner(config={}))

    def test_use_or_create_async_context__when_missing(self):
        if False:
            print('Hello World!')
        context = self.make_context()
        context._push()
        async_context = use_or_create_async_context(context)
        assert isinstance(async_context, AsyncContext)
        assert async_context.name == 'async_context'
        assert getattr(context, 'async_context', None) is async_context
        context._pop()
        assert getattr(context, 'async_context', None) is None

    def test_use_or_create_async_context__when_exists(self):
        if False:
            for i in range(10):
                print('nop')
        context = self.make_context()
        async_context0 = context.async_context = AsyncContext()
        assert context.async_context.name == 'async_context'
        assert hasattr(context, AsyncContext.default_name)
        async_context = use_or_create_async_context(context)
        assert isinstance(async_context, AsyncContext)
        assert async_context.name == 'async_context'
        assert getattr(context, 'async_context', None) is async_context
        assert async_context is async_context0

    def test_use_or_create_async_context__when_missing_with_name(self):
        if False:
            return 10
        loop0 = Mock()
        context = self.make_context()
        async_context = use_or_create_async_context(context, 'acontext', loop=loop0)
        assert isinstance(async_context, AsyncContext)
        assert async_context.name == 'acontext'
        assert getattr(context, 'acontext', None) is async_context
        assert async_context.loop is loop0

    def test_use_or_create_async_context__when_exists_with_name(self):
        if False:
            print('Hello World!')
        loop0 = Mock()
        context = self.make_context()
        async_context0 = context.acontext = AsyncContext(name='acontext', loop=loop0)
        assert context.acontext.name == 'acontext'
        loop1 = Mock()
        async_context = use_or_create_async_context(context, 'acontext', loop=loop1)
        assert isinstance(async_context, AsyncContext)
        assert async_context is async_context0
        assert getattr(context, 'acontext', None) is async_context
        assert async_context.loop is loop0

@requires_py34_to_py37
class TestAsyncStepRunPy34:
    """Ensure that execution of async-steps works as expected."""

    def test_async_step_passes(self):
        if False:
            print('Hello World!')
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import given, when
            from behave.api.async_step import async_run_until_complete
            import asyncio

            @given('an async-step passes')
            @async_run_until_complete
            @asyncio.coroutine
            def given_async_step_passes(context):
                if False:
                    while True:
                        i = 10
                context.traced_steps.append('async-step1')

            @when('an async-step passes')
            @async_run_until_complete
            @asyncio.coroutine
            def when_async_step_passes(context):
                if False:
                    return 10
                context.traced_steps.append('async-step2')
        context = Context(runner=Runner(config={}))
        context.traced_steps = []
        given_async_step_passes(context)
        when_async_step_passes(context)
        assert context.traced_steps == ['async-step1', 'async-step2']

    def test_async_step_fails(self):
        if False:
            while True:
                i = 10
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import when
            from behave.api.async_step import async_run_until_complete
            import asyncio

            @when('an async-step fails')
            @async_run_until_complete
            @asyncio.coroutine
            def when_async_step_fails(context):
                if False:
                    i = 10
                    return i + 15
                assert False, 'XFAIL in async-step'
        context = Context(runner=Runner(config={}))
        with pytest.raises(AssertionError):
            when_async_step_fails(context)

    def test_async_step_raises_exception(self):
        if False:
            print('Hello World!')
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import when
            from behave.api.async_step import async_run_until_complete
            import asyncio

            @when('an async-step raises exception')
            @async_run_until_complete
            @asyncio.coroutine
            def when_async_step_raises_exception(context):
                if False:
                    return 10
                1 / 0
        context = Context(runner=Runner(config={}))
        with pytest.raises(ZeroDivisionError):
            when_async_step_raises_exception(context)