"""Transitional integration with Baseplate.

This module provides basic transitional integration with Baseplate. Its intent
is to integrate baseplate-provided functionality (like thrift clients) into
r2's existing diagnostics infrastructure. It is not meant to be the last word
on r2+baseplate; ideally r2 will move towards using more of baseplate rather
than its own implementations.

"""
import functools
import sys
from baseplate.core import BaseplateObserver, ServerSpanObserver, SpanObserver
from pylons import app_globals as g, tmpl_context as c

def make_server_span(span_name):
    if False:
        for i in range(10):
            print('nop')
    c.trace = g.baseplate.make_server_span(context=c, name=span_name)
    return c.trace

def finish_server_span():
    if False:
        for i in range(10):
            print('nop')
    c.trace.finish()

def with_server_span(name):
    if False:
        print('Hello World!')
    'A decorator for functions that run outside request context.\n\n    This will add a server span which starts just before invocation of the\n    function and ends immediately after. The context (`c`) will have all\n    appropriate baseplate stuff added to it, and metrics will be flushed when\n    the function returns.\n\n    This is useful for functions run in cron jobs or from the shell. Note that\n    you cannot call a function wrapped with this decorator from within an\n    existing server span.\n\n    '

    def with_server_span_decorator(fn):
        if False:
            i = 10
            return i + 15

        @functools.wraps(fn)
        def with_server_span_wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            assert not c.trace, 'called while already in a server span'
            try:
                with make_server_span(name):
                    return fn(*args, **kwargs)
            finally:
                g.stats.flush()
        return with_server_span_wrapper
    return with_server_span_decorator
with_root_span = with_server_span

class R2BaseplateObserver(BaseplateObserver):

    def on_server_span_created(self, context, server_span):
        if False:
            while True:
                i = 10
        observer = R2ServerSpanObserver()
        server_span.register(observer)

class R2ServerSpanObserver(ServerSpanObserver):

    def on_child_span_created(self, span):
        if False:
            return 10
        observer = R2SpanObserver(span.name)
        span.register(observer)

class R2SpanObserver(SpanObserver):

    def __init__(self, span_name):
        if False:
            for i in range(10):
                print('nop')
        self.metric_name = 'providers.{}'.format(span_name)
        self.timer = g.stats.get_timer(self.metric_name)

    def on_start(self):
        if False:
            return 10
        self.timer.start()

    def on_finish(self, exc_info):
        if False:
            while True:
                i = 10
        self.timer.stop()
        if exc_info:
            error = exc_info[1]
            g.log.warning('%s: error: %s', self.metric_name, error)
            g.stats.simple_event('{}.error'.format(self.metric_name))