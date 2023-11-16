"""Test script to find circular references.

Circular references are not leaks per se, because they will eventually
be GC'd. However, on CPython, they prevent the reference-counting fast
path from being used and instead rely on the slower full GC. This
increases memory footprint and CPU overhead, so we try to eliminate
circular references created by normal operation.
"""
import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython

def find_circular_references(garbage):
    if False:
        i = 10
        return i + 15
    'Find circular references in a list of objects.\n\n    The garbage list contains objects that participate in a cycle,\n    but also the larger set of objects kept alive by that cycle.\n    This function finds subsets of those objects that make up\n    the cycle(s).\n    '

    def inner(level):
        if False:
            return 10
        for item in level:
            item_id = id(item)
            if item_id not in garbage_ids:
                continue
            if item_id in visited_ids:
                continue
            if item_id in stack_ids:
                candidate = stack[stack.index(item):]
                candidate.append(item)
                found.append(candidate)
                continue
            stack.append(item)
            stack_ids.add(item_id)
            inner(gc.get_referents(item))
            stack.pop()
            stack_ids.remove(item_id)
            visited_ids.add(item_id)
    found: typing.List[object] = []
    stack = []
    stack_ids = set()
    garbage_ids = set(map(id, garbage))
    visited_ids = set()
    inner(garbage)
    return found

@contextlib.contextmanager
def assert_no_cycle_garbage():
    if False:
        for i in range(10):
            print('nop')
    'Raise AssertionError if the wrapped code creates garbage with cycles.'
    gc.disable()
    gc.collect()
    gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_SAVEALL)
    yield
    try:
        f = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = f
        try:
            gc.collect()
        finally:
            sys.stderr = old_stderr
        garbage = gc.garbage[:]
        gc.garbage[:] = []
        if len(garbage) == 0:
            return
        for circular in find_circular_references(garbage):
            f.write('\n==========\n Circular \n==========')
            for item in circular:
                f.write(f'\n    {repr(item)}')
            for item in circular:
                if isinstance(item, types.FrameType):
                    f.write(f'\nLocals: {item.f_locals}')
                    f.write(f'\nTraceback: {repr(item)}')
                    traceback.print_stack(item)
        del garbage
        raise AssertionError(f.getvalue())
    finally:
        gc.set_debug(0)
        gc.enable()

@skipNotCPython
class CircleRefsTest(unittest.TestCase):

    def test_known_leak(self):
        if False:
            for i in range(10):
                print('nop')

        class C(object):

            def __init__(self, name):
                if False:
                    for i in range(10):
                        print('nop')
                self.name = name
                self.a: typing.Optional[C] = None
                self.b: typing.Optional[C] = None
                self.c: typing.Optional[C] = None

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return f'name={self.name}'
        with self.assertRaises(AssertionError) as cm:
            with assert_no_cycle_garbage():
                a = C('a')
                b = C('b')
                c = C('c')
                a.b = b
                a.c = c
                b.a = a
                b.c = c
                del a, b
        self.assertIn('Circular', str(cm.exception))
        self.assertIn('    name=a', str(cm.exception))
        self.assertIn('    name=b', str(cm.exception))
        self.assertNotIn('    name=c', str(cm.exception))

    async def run_handler(self, handler_class):
        app = web.Application([('/', handler_class)])
        (socket, port) = tornado.testing.bind_unused_port()
        server = tornado.httpserver.HTTPServer(app)
        server.add_socket(socket)
        client = httpclient.AsyncHTTPClient()
        with assert_no_cycle_garbage():
            await client.fetch(f'http://127.0.0.1:{port}/')
        client.close()
        server.stop()
        socket.close()

    def test_sync_handler(self):
        if False:
            print('Hello World!')

        class Handler(web.RequestHandler):

            def get(self):
                if False:
                    return 10
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_finish_exception_handler(self):
        if False:
            i = 10
            return i + 15

        class Handler(web.RequestHandler):

            def get(self):
                if False:
                    while True:
                        i = 10
                raise web.Finish('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_coro_handler(self):
        if False:
            return 10

        class Handler(web.RequestHandler):

            @gen.coroutine
            def get(self):
                if False:
                    print('Hello World!')
                yield asyncio.sleep(0.01)
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_async_handler(self):
        if False:
            while True:
                i = 10

        class Handler(web.RequestHandler):

            async def get(self):
                await asyncio.sleep(0.01)
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_run_on_executor(self):
        if False:
            for i in range(10):
                print('nop')
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as thread_pool:

            class Factory(object):
                executor = thread_pool

                @tornado.concurrent.run_on_executor
                def run(self):
                    if False:
                        while True:
                            i = 10
                    return None
            factory = Factory()

            async def main():
                for i in range(2):
                    await factory.run()
            with assert_no_cycle_garbage():
                asyncio.run(main())