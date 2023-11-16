from typing import Awaitable, cast
from twisted.internet import defer
from twisted.test.proto_helpers import MemoryReactorClock
from synapse.logging.context import LoggingContext, make_deferred_yieldable, run_in_background
from synapse.logging.opentracing import start_active_span, start_active_span_follows_from, tag_args, trace_with_opname
from synapse.util import Clock
try:
    from synapse.logging.scopecontextmanager import LogContextScopeManager
except ImportError:
    LogContextScopeManager = None
try:
    import jaeger_client
except ImportError:
    jaeger_client = None
import logging
from tests.unittest import TestCase
logger = logging.getLogger(__name__)

class LogContextScopeManagerTestCase(TestCase):
    """
    Test logging contexts and active opentracing spans.

    There's casts throughout this from generic opentracing objects (e.g.
    opentracing.Span) to the ones specific to Jaeger since they have additional
    properties that these tests depend on. This is safe since the only supported
    opentracing backend is Jaeger.
    """
    if LogContextScopeManager is None:
        skip = 'Requires opentracing'
    if jaeger_client is None:
        skip = 'Requires jaeger_client'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        scope_manager = LogContextScopeManager()
        config = jaeger_client.config.Config(config={}, service_name='test', scope_manager=scope_manager)
        self._reporter = jaeger_client.reporter.InMemoryReporter()
        self._tracer = config.create_tracer(sampler=jaeger_client.ConstSampler(True), reporter=self._reporter)

    def test_start_active_span(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LoggingContext('root context'):
            self.assertIsNone(self._tracer.active_span)
            scope = start_active_span('span', tracer=self._tracer)
            span = cast(jaeger_client.Span, scope.span)
            self.assertEqual(self._tracer.active_span, span)
            self.assertIsNotNone(span.start_time)
            with scope as ctx:
                self.assertIs(ctx, scope)
                self.assertEqual(self._tracer.active_span, span)
            self.assertIsNone(self._tracer.active_span)
            self.assertIsNotNone(span.end_time)
        self.assertEqual(self._reporter.get_spans(), [span])

    def test_nested_spans(self) -> None:
        if False:
            print('Hello World!')
        'Starting two spans off inside each other should work'
        with LoggingContext('root context'):
            with start_active_span('root span', tracer=self._tracer) as root_scope:
                self.assertEqual(self._tracer.active_span, root_scope.span)
                root_context = cast(jaeger_client.SpanContext, root_scope.span.context)
                scope1 = start_active_span('child1', tracer=self._tracer)
                self.assertEqual(self._tracer.active_span, scope1.span, 'child1 was not activated')
                context1 = cast(jaeger_client.SpanContext, scope1.span.context)
                self.assertEqual(context1.parent_id, root_context.span_id)
                scope2 = start_active_span_follows_from('child2', contexts=(scope1,), tracer=self._tracer)
                self.assertEqual(self._tracer.active_span, scope2.span)
                context2 = cast(jaeger_client.SpanContext, scope2.span.context)
                self.assertEqual(context2.parent_id, context1.span_id)
                with scope1, scope2:
                    pass
                self.assertEqual(self._tracer.active_span, root_scope.span)
                span2 = cast(jaeger_client.Span, scope2.span)
                span1 = cast(jaeger_client.Span, scope1.span)
                self.assertIsNotNone(span2.end_time)
                self.assertIsNotNone(span1.end_time)
            self.assertIsNone(self._tracer.active_span)
        self.assertEqual(self._reporter.get_spans(), [scope2.span, scope1.span, root_scope.span])

    def test_overlapping_spans(self) -> None:
        if False:
            print('Hello World!')
        'Overlapping spans which are not neatly nested should work'
        reactor = MemoryReactorClock()
        clock = Clock(reactor)
        scopes = []

        async def task(i: int) -> None:
            scope = start_active_span(f'task{i}', tracer=self._tracer)
            scopes.append(scope)
            self.assertEqual(self._tracer.active_span, scope.span)
            await clock.sleep(4)
            self.assertEqual(self._tracer.active_span, scope.span)
            scope.close()

        async def root() -> None:
            with start_active_span('root span', tracer=self._tracer) as root_scope:
                self.assertEqual(self._tracer.active_span, root_scope.span)
                scopes.append(root_scope)
                d1 = run_in_background(task, 1)
                await clock.sleep(2)
                d2 = run_in_background(task, 2)
                self.assertEqual(self._tracer.active_span, root_scope.span)
                await make_deferred_yieldable(defer.gatherResults([d1, d2], consumeErrors=True))
                self.assertEqual(self._tracer.active_span, root_scope.span)
        with LoggingContext('root context'):
            d1 = defer.ensureDeferred(root())
            reactor.pump((2,) * 8)
            self.successResultOf(d1)
            self.assertIsNone(self._tracer.active_span)
        self.assertEqual(self._reporter.get_spans(), [scopes[1].span, scopes[2].span, scopes[0].span])

    def test_trace_decorator_sync(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test whether we can use `@trace_with_opname` (`@trace`) and `@tag_args`\n        with sync functions\n        '
        with LoggingContext('root context'):

            @trace_with_opname('fixture_sync_func', tracer=self._tracer)
            @tag_args
            def fixture_sync_func() -> str:
                if False:
                    return 10
                return 'foo'
            result = fixture_sync_func()
            self.assertEqual(result, 'foo')
        self.assertEqual([span.operation_name for span in self._reporter.get_spans()], ['fixture_sync_func'])

    def test_trace_decorator_deferred(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test whether we can use `@trace_with_opname` (`@trace`) and `@tag_args`\n        with functions that return deferreds\n        '
        with LoggingContext('root context'):

            @trace_with_opname('fixture_deferred_func', tracer=self._tracer)
            @tag_args
            def fixture_deferred_func() -> 'defer.Deferred[str]':
                if False:
                    i = 10
                    return i + 15
                d1: defer.Deferred[str] = defer.Deferred()
                d1.callback('foo')
                return d1
            result_d1 = fixture_deferred_func()
            self.assertEqual(self.successResultOf(result_d1), 'foo')
        self.assertEqual([span.operation_name for span in self._reporter.get_spans()], ['fixture_deferred_func'])

    def test_trace_decorator_async(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether we can use `@trace_with_opname` (`@trace`) and `@tag_args`\n        with async functions\n        '
        with LoggingContext('root context'):

            @trace_with_opname('fixture_async_func', tracer=self._tracer)
            @tag_args
            async def fixture_async_func() -> str:
                return 'foo'
            d1 = defer.ensureDeferred(fixture_async_func())
            self.assertEqual(self.successResultOf(d1), 'foo')
        self.assertEqual([span.operation_name for span in self._reporter.get_spans()], ['fixture_async_func'])

    def test_trace_decorator_awaitable_return(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether we can use `@trace_with_opname` (`@trace`) and `@tag_args`\n        with functions that return an awaitable (e.g. a coroutine)\n        '
        with LoggingContext('root context'):

            async def fixture_async_func() -> str:
                return 'foo'

            @trace_with_opname('fixture_awaitable_return_func', tracer=self._tracer)
            @tag_args
            def fixture_awaitable_return_func() -> Awaitable[str]:
                if False:
                    for i in range(10):
                        print('nop')
                return fixture_async_func()

            async def runner() -> str:
                return await fixture_awaitable_return_func()
            d1 = defer.ensureDeferred(runner())
            self.assertEqual(self.successResultOf(d1), 'foo')
        self.assertEqual([span.operation_name for span in self._reporter.get_spans()], ['fixture_awaitable_return_func'])