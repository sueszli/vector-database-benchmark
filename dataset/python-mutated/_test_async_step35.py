"""
Unit tests for :mod:`behave.api.async_test` for Python 3.5 (or newer).
"""
from __future__ import absolute_import, print_function
import sys
from hamcrest import assert_that, close_to
import pytest
from behave._stepimport import use_step_import_modules, SimpleStepContainer
from behave.runner import Context, Runner
from .testing_support import StopWatch
from .testing_support_async import AsyncStepTheory
PYTHON_3_5 = (3, 5)
python_version = sys.version_info[:2]
py35_or_newer = pytest.mark.skipif(python_version < PYTHON_3_5, reason='Needs Python >= 3.5')
SLEEP_DELTA = 0.05
if sys.platform.startswith('win'):
    SLEEP_DELTA = 0.1

@py35_or_newer
class TestAsyncStepDecoratorPy35:

    def test_step_decorator_async_run_until_complete1(self):
        if False:
            i = 10
            return i + 15
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import step
            from behave.api.async_step import async_run_until_complete
            import asyncio

            @step('an async coroutine step waits "{duration:f}" seconds')
            @async_run_until_complete
            async def step_async_step_waits_seconds(context, duration):
                await asyncio.sleep(duration)
        AsyncStepTheory.validate(step_async_step_waits_seconds)
        context = Context(runner=Runner(config={}))
        with StopWatch() as stop_watch:
            step_async_step_waits_seconds(context, 0.2)
        assert_that(stop_watch.duration, close_to(0.2, delta=SLEEP_DELTA))

@py35_or_newer
class TestAsyncStepRunPy35:
    """Ensure that execution of async-steps works as expected."""

    def test_async_step_passes(self):
        if False:
            return 10
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import given, when
            from behave.api.async_step import async_run_until_complete

            @given('an async-step passes')
            @async_run_until_complete
            async def given_async_step_passes(context):
                context.traced_steps.append('async-step1')

            @when('an async-step passes')
            @async_run_until_complete
            async def when_async_step_passes(context):
                context.traced_steps.append('async-step2')
        context = Context(runner=Runner(config={}))
        context.traced_steps = []
        given_async_step_passes(context)
        when_async_step_passes(context)
        assert context.traced_steps == ['async-step1', 'async-step2']

    def test_async_step_fails(self):
        if False:
            return 10
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import when
            from behave.api.async_step import async_run_until_complete

            @when('an async-step fails')
            @async_run_until_complete
            async def when_async_step_fails(context):
                assert False, 'XFAIL in async-step'
        context = Context(runner=Runner(config={}))
        with pytest.raises(AssertionError):
            when_async_step_fails(context)

    def test_async_step_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        'ENSURE: Failures in async-steps are detected correctly.'
        step_container = SimpleStepContainer()
        with use_step_import_modules(step_container):
            from behave import when
            from behave.api.async_step import async_run_until_complete

            @when('an async-step raises exception')
            @async_run_until_complete
            async def when_async_step_raises_exception(context):
                1 / 0
        context = Context(runner=Runner(config={}))
        with pytest.raises(ZeroDivisionError):
            when_async_step_raises_exception(context)