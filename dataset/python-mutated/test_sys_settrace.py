from test import support
import unittest
import sys
import difflib
import gc
from functools import wraps
import asyncio

class tracecontext:
    """Context manager that traces its enter and exit."""

    def __init__(self, output, value):
        if False:
            for i in range(10):
                print('nop')
        self.output = output
        self.value = value

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.output.append(self.value)

    def __exit__(self, *exc_info):
        if False:
            while True:
                i = 10
        self.output.append(-self.value)

class asynctracecontext:
    """Asynchronous context manager that traces its aenter and aexit."""

    def __init__(self, output, value):
        if False:
            print('Hello World!')
        self.output = output
        self.value = value

    async def __aenter__(self):
        self.output.append(self.value)

    async def __aexit__(self, *exc_info):
        self.output.append(-self.value)

async def asynciter(iterable):
    """Convert an iterable to an asynchronous iterator."""
    for x in iterable:
        yield x

def basic():
    if False:
        i = 10
        return i + 15
    return 1
basic.events = [(0, 'call'), (1, 'line'), (1, 'return')]

def arigo_example0():
    if False:
        print('Hello World!')
    x = 1
    del x
    while 0:
        pass
    x = 1
arigo_example0.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (5, 'line'), (5, 'return')]

def arigo_example1():
    if False:
        print('Hello World!')
    x = 1
    del x
    if 0:
        pass
    x = 1
arigo_example1.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (5, 'line'), (5, 'return')]

def arigo_example2():
    if False:
        print('Hello World!')
    x = 1
    del x
    if 1:
        x = 1
    else:
        pass
    return None
arigo_example2.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (7, 'line'), (7, 'return')]

def one_instr_line():
    if False:
        print('Hello World!')
    x = 1
    del x
    x = 1
one_instr_line.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'return')]

def no_pop_tops():
    if False:
        while True:
            i = 10
    x = 1
    for a in range(2):
        if a:
            x = 1
        else:
            x = 1
no_pop_tops.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (6, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (2, 'line'), (2, 'return')]

def no_pop_blocks():
    if False:
        i = 10
        return i + 15
    y = 1
    while not y:
        bla
    x = 1
no_pop_blocks.events = [(0, 'call'), (1, 'line'), (2, 'line'), (4, 'line'), (4, 'return')]

def called():
    if False:
        print('Hello World!')
    x = 1

def call():
    if False:
        while True:
            i = 10
    called()
call.events = [(0, 'call'), (1, 'line'), (-3, 'call'), (-2, 'line'), (-2, 'return'), (1, 'return')]

def raises():
    if False:
        for i in range(10):
            print('nop')
    raise Exception

def test_raise():
    if False:
        i = 10
        return i + 15
    try:
        raises()
    except Exception:
        pass
test_raise.events = [(0, 'call'), (1, 'line'), (2, 'line'), (-3, 'call'), (-2, 'line'), (-2, 'exception'), (-2, 'return'), (2, 'exception'), (3, 'line'), (4, 'line'), (4, 'return')]

def _settrace_and_return(tracefunc):
    if False:
        print('Hello World!')
    sys.settrace(tracefunc)
    sys._getframe().f_back.f_trace = tracefunc

def settrace_and_return(tracefunc):
    if False:
        while True:
            i = 10
    _settrace_and_return(tracefunc)
settrace_and_return.events = [(1, 'return')]

def _settrace_and_raise(tracefunc):
    if False:
        print('Hello World!')
    sys.settrace(tracefunc)
    sys._getframe().f_back.f_trace = tracefunc
    raise RuntimeError

def settrace_and_raise(tracefunc):
    if False:
        for i in range(10):
            print('nop')
    try:
        _settrace_and_raise(tracefunc)
    except RuntimeError:
        pass
settrace_and_raise.events = [(2, 'exception'), (3, 'line'), (4, 'line'), (4, 'return')]

def ireturn_example():
    if False:
        print('Hello World!')
    a = 5
    b = 5
    if a == b:
        b = a + 1
    else:
        pass
ireturn_example.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (4, 'return')]

def tightloop_example():
    if False:
        i = 10
        return i + 15
    items = range(0, 3)
    try:
        i = 0
        while 1:
            b = items[i]
            i += 1
    except IndexError:
        pass
tightloop_example.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (4, 'line'), (5, 'line'), (4, 'line'), (5, 'line'), (4, 'line'), (5, 'line'), (5, 'exception'), (6, 'line'), (7, 'line'), (7, 'return')]

def tighterloop_example():
    if False:
        for i in range(10):
            print('nop')
    items = range(1, 4)
    try:
        i = 0
        while 1:
            i = items[i]
    except IndexError:
        pass
tighterloop_example.events = [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (4, 'line'), (4, 'line'), (4, 'line'), (4, 'exception'), (5, 'line'), (6, 'line'), (6, 'return')]

def generator_function():
    if False:
        for i in range(10):
            print('nop')
    try:
        yield True
        'continued'
    finally:
        'finally'

def generator_example():
    if False:
        return 10
    x = any(generator_function())
    for x in range(10):
        y = x
generator_example.events = [(0, 'call'), (2, 'line'), (-6, 'call'), (-5, 'line'), (-4, 'line'), (-4, 'return'), (-4, 'call'), (-4, 'exception'), (-1, 'line'), (-1, 'return')] + [(5, 'line'), (6, 'line')] * 10 + [(5, 'line'), (5, 'return')]

class Tracer:

    def __init__(self, trace_line_events=None, trace_opcode_events=None):
        if False:
            for i in range(10):
                print('nop')
        self.trace_line_events = trace_line_events
        self.trace_opcode_events = trace_opcode_events
        self.events = []

    def _reconfigure_frame(self, frame):
        if False:
            for i in range(10):
                print('nop')
        if self.trace_line_events is not None:
            frame.f_trace_lines = self.trace_line_events
        if self.trace_opcode_events is not None:
            frame.f_trace_opcodes = self.trace_opcode_events

    def trace(self, frame, event, arg):
        if False:
            i = 10
            return i + 15
        self._reconfigure_frame(frame)
        self.events.append((frame.f_lineno, event))
        return self.trace

    def traceWithGenexp(self, frame, event, arg):
        if False:
            for i in range(10):
                print('nop')
        self._reconfigure_frame(frame)
        (o for o in [1])
        self.events.append((frame.f_lineno, event))
        return self.trace

class TraceTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.using_gc = gc.isenabled()
        gc.disable()
        self.addCleanup(sys.settrace, sys.gettrace())

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.using_gc:
            gc.enable()

    @staticmethod
    def make_tracer():
        if False:
            while True:
                i = 10
        'Helper to allow test subclasses to configure tracers differently'
        return Tracer()

    def compare_events(self, line_offset, events, expected_events):
        if False:
            i = 10
            return i + 15
        events = [(l - line_offset, e) for (l, e) in events]
        if events != expected_events:
            self.fail('events did not match expectation:\n' + '\n'.join(difflib.ndiff([str(x) for x in expected_events], [str(x) for x in events])))

    def run_and_compare(self, func, events):
        if False:
            print('Hello World!')
        tracer = self.make_tracer()
        sys.settrace(tracer.trace)
        func()
        sys.settrace(None)
        self.compare_events(func.__code__.co_firstlineno, tracer.events, events)

    def run_test(self, func):
        if False:
            while True:
                i = 10
        self.run_and_compare(func, func.events)

    def run_test2(self, func):
        if False:
            return 10
        tracer = self.make_tracer()
        func(tracer.trace)
        sys.settrace(None)
        self.compare_events(func.__code__.co_firstlineno, tracer.events, func.events)

    def test_set_and_retrieve_none(self):
        if False:
            for i in range(10):
                print('nop')
        sys.settrace(None)
        assert sys.gettrace() is None

    def test_set_and_retrieve_func(self):
        if False:
            print('Hello World!')

        def fn(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass
        sys.settrace(fn)
        try:
            assert sys.gettrace() is fn
        finally:
            sys.settrace(None)

    def test_01_basic(self):
        if False:
            print('Hello World!')
        self.run_test(basic)

    def test_02_arigo0(self):
        if False:
            i = 10
            return i + 15
        self.run_test(arigo_example0)

    def test_02_arigo1(self):
        if False:
            while True:
                i = 10
        self.run_test(arigo_example1)

    def test_02_arigo2(self):
        if False:
            while True:
                i = 10
        self.run_test(arigo_example2)

    def test_03_one_instr(self):
        if False:
            return 10
        self.run_test(one_instr_line)

    def test_04_no_pop_blocks(self):
        if False:
            print('Hello World!')
        self.run_test(no_pop_blocks)

    def test_05_no_pop_tops(self):
        if False:
            return 10
        self.run_test(no_pop_tops)

    def test_06_call(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(call)

    def test_07_raise(self):
        if False:
            print('Hello World!')
        self.run_test(test_raise)

    def test_08_settrace_and_return(self):
        if False:
            while True:
                i = 10
        self.run_test2(settrace_and_return)

    def test_09_settrace_and_raise(self):
        if False:
            while True:
                i = 10
        self.run_test2(settrace_and_raise)

    def test_10_ireturn(self):
        if False:
            i = 10
            return i + 15
        self.run_test(ireturn_example)

    def test_11_tightloop(self):
        if False:
            while True:
                i = 10
        self.run_test(tightloop_example)

    def test_12_tighterloop(self):
        if False:
            i = 10
            return i + 15
        self.run_test(tighterloop_example)

    def test_13_genexp(self):
        if False:
            while True:
                i = 10
        self.run_test(generator_example)
        tracer = self.make_tracer()
        sys.settrace(tracer.traceWithGenexp)
        generator_example()
        sys.settrace(None)
        self.compare_events(generator_example.__code__.co_firstlineno, tracer.events, generator_example.events)

    def test_14_onliner_if(self):
        if False:
            print('Hello World!')

        def onliners():
            if False:
                for i in range(10):
                    print('nop')
            if True:
                x = False
            else:
                x = True
            return 0
        self.run_and_compare(onliners, [(0, 'call'), (1, 'line'), (3, 'line'), (3, 'return')])

    def test_15_loops(self):
        if False:
            for i in range(10):
                print('nop')

        def for_example():
            if False:
                for i in range(10):
                    print('nop')
            for x in range(2):
                pass
        self.run_and_compare(for_example, [(0, 'call'), (1, 'line'), (2, 'line'), (1, 'line'), (2, 'line'), (1, 'line'), (1, 'return')])

        def while_example():
            if False:
                while True:
                    i = 10
            x = 2
            while x > 0:
                x -= 1
        self.run_and_compare(while_example, [(0, 'call'), (2, 'line'), (3, 'line'), (4, 'line'), (3, 'line'), (4, 'line'), (3, 'line'), (3, 'return')])

    def test_16_blank_lines(self):
        if False:
            print('Hello World!')
        namespace = {}
        exec('def f():\n' + '\n' * 256 + '    pass', namespace)
        self.run_and_compare(namespace['f'], [(0, 'call'), (257, 'line'), (257, 'return')])

    def test_17_none_f_trace(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                i = 10
                return i + 15
            sys._getframe().f_trace = None
            lineno = 2
        self.run_and_compare(func, [(0, 'call'), (1, 'line')])

    def test_18_except_with_name(self):
        if False:
            return 10

        def func():
            if False:
                for i in range(10):
                    print('nop')
            try:
                try:
                    raise Exception
                except Exception as e:
                    raise
                    x = 'Something'
                    y = 'Something'
            except Exception:
                pass
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'exception'), (4, 'line'), (5, 'line'), (8, 'line'), (9, 'line'), (9, 'return')])

    def test_19_except_with_finally(self):
        if False:
            i = 10
            return i + 15

        def func():
            if False:
                while True:
                    i = 10
            try:
                try:
                    raise Exception
                finally:
                    y = 'Something'
            except Exception:
                b = 23
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'exception'), (5, 'line'), (6, 'line'), (7, 'line'), (7, 'return')])

    def test_20_async_for_loop(self):
        if False:
            for i in range(10):
                print('nop')

        class AsyncIteratorWrapper:

            def __init__(self, obj):
                if False:
                    return 10
                self._it = iter(obj)

            def __aiter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        async def doit_async():
            async for letter in AsyncIteratorWrapper('abc'):
                x = letter
            y = 42

        def run(tracer):
            if False:
                print('Hello World!')
            x = doit_async()
            try:
                sys.settrace(tracer)
                x.send(None)
            finally:
                sys.settrace(None)
        tracer = self.make_tracer()
        events = [(0, 'call'), (1, 'line'), (-12, 'call'), (-11, 'line'), (-11, 'return'), (-9, 'call'), (-8, 'line'), (-8, 'return'), (-6, 'call'), (-5, 'line'), (-4, 'line'), (-4, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-6, 'call'), (-5, 'line'), (-4, 'line'), (-4, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-6, 'call'), (-5, 'line'), (-4, 'line'), (-4, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-6, 'call'), (-5, 'line'), (-4, 'line'), (-4, 'exception'), (-3, 'line'), (-2, 'line'), (-2, 'exception'), (-2, 'return'), (1, 'exception'), (3, 'line'), (3, 'return')]
        try:
            run(tracer.trace)
        except Exception:
            pass
        self.compare_events(doit_async.__code__.co_firstlineno, tracer.events, events)

    def test_async_for_backwards_jump_has_no_line(self):
        if False:
            return 10

        async def arange(n):
            for i in range(n):
                yield i

        async def f():
            async for i in arange(3):
                if i > 100:
                    break
        tracer = self.make_tracer()
        coro = f()
        try:
            sys.settrace(tracer.trace)
            coro.send(None)
        except Exception:
            pass
        finally:
            sys.settrace(None)
        events = [(0, 'call'), (1, 'line'), (-3, 'call'), (-2, 'line'), (-1, 'line'), (-1, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-1, 'call'), (-2, 'line'), (-1, 'line'), (-1, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-1, 'call'), (-2, 'line'), (-1, 'line'), (-1, 'return'), (1, 'exception'), (2, 'line'), (1, 'line'), (-1, 'call'), (-2, 'line'), (-2, 'return'), (1, 'exception'), (1, 'return')]
        self.compare_events(f.__code__.co_firstlineno, tracer.events, events)

    def test_21_repeated_pass(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                return 10
            pass
            pass
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (2, 'return')])

    def test_loop_in_try_except(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                print('Hello World!')
            try:
                for i in []:
                    pass
                return 1
            except:
                return 2
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'return')])

    def test_try_except_no_exception(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            try:
                2
            except:
                4
            finally:
                6
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (6, 'line'), (6, 'return')])

    def test_nested_loops(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            for i in range(2):
                for j in range(2):
                    a = i + j
            return a == 1
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (2, 'line'), (3, 'line'), (2, 'line'), (1, 'line'), (2, 'line'), (3, 'line'), (2, 'line'), (3, 'line'), (2, 'line'), (1, 'line'), (4, 'line'), (4, 'return')])

    def test_if_break(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                while True:
                    i = 10
            seq = [1, 0]
            while seq:
                n = seq.pop()
                if n:
                    break
            else:
                n = 99
            return n
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (8, 'line'), (8, 'return')])

    def test_break_through_finally(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            (a, c, d, i) = (1, 1, 1, 99)
            try:
                for i in range(3):
                    try:
                        a = 5
                        if i > 0:
                            break
                        a = 8
                    finally:
                        c = 10
            except:
                d = 12
            assert a == 5 and c == 10 and (d == 1)
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (8, 'line'), (10, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (7, 'line'), (10, 'line'), (13, 'line'), (13, 'return')])

    def test_continue_through_finally(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                print('Hello World!')
            (a, b, c, d, i) = (1, 1, 1, 1, 99)
            try:
                for i in range(2):
                    try:
                        a = 5
                        if i > 0:
                            continue
                        b = 8
                    finally:
                        c = 10
            except:
                d = 12
            assert (a, b, c, d) == (5, 8, 10, 1)
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (8, 'line'), (10, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (7, 'line'), (10, 'line'), (3, 'line'), (13, 'line'), (13, 'return')])

    def test_return_through_finally(self):
        if False:
            return 10

        def func():
            if False:
                print('Hello World!')
            try:
                return 2
            finally:
                4
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (4, 'line'), (4, 'return')])

    def test_try_except_with_wrong_type(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                for i in range(10):
                    print('nop')
            try:
                2 / 0
            except IndexError:
                4
            finally:
                return 6
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (2, 'exception'), (3, 'line'), (6, 'line'), (6, 'return')])

    def test_break_to_continue1(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                print('Hello World!')
            TRUE = 1
            x = [1]
            while x:
                x.pop()
                while TRUE:
                    break
                continue
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (7, 'line'), (3, 'line'), (3, 'return')])

    def test_break_to_continue2(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                i = 10
                return i + 15
            TRUE = 1
            x = [1]
            while x:
                x.pop()
                while TRUE:
                    break
                else:
                    continue
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (6, 'line'), (3, 'line'), (3, 'return')])

    def test_break_to_break(self):
        if False:
            return 10

        def func():
            if False:
                print('Hello World!')
            TRUE = 1
            while TRUE:
                while TRUE:
                    break
                break
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (5, 'line'), (5, 'return')])

    def test_nested_ifs(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                return 10
            a = b = 1
            if a == 1:
                if b == 1:
                    x = 4
                else:
                    y = 6
            else:
                z = 8
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (4, 'line'), (4, 'return')])

    def test_nested_ifs_with_and(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                return 10
            if A:
                if B:
                    if C:
                        if D:
                            return False
                else:
                    return False
            elif E and F:
                return True
        A = B = True
        C = False
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'return')])

    def test_nested_try_if(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                print('Hello World!')
            x = 'hello'
            try:
                3 / 0
            except ZeroDivisionError:
                if x == 'raise':
                    raise ValueError()
            f = 7
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'exception'), (4, 'line'), (5, 'line'), (7, 'line'), (7, 'return')])

    def test_if_false_in_with(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __enter__(self):
                if False:
                    return 10
                return self

            def __exit__(*args):
                if False:
                    while True:
                        i = 10
                pass

        def func():
            if False:
                print('Hello World!')
            with C():
                if False:
                    pass
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (-5, 'call'), (-4, 'line'), (-4, 'return'), (2, 'line'), (1, 'line'), (-3, 'call'), (-2, 'line'), (-2, 'return'), (1, 'return')])

    def test_if_false_in_try_except(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                i = 10
                return i + 15
            try:
                if False:
                    pass
            except Exception:
                X
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (2, 'return')])

    def test_implicit_return_in_class(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                return 10

            class A:
                if 3 < 9:
                    a = 1
                else:
                    a = 2
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (1, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'return'), (1, 'return')])

    def test_try_in_try(self):
        if False:
            return 10

        def func():
            if False:
                return 10
            try:
                try:
                    pass
                except Exception as ex:
                    pass
            except Exception:
                pass
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'return')])

    def test_if_in_if_in_if(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a=0, p=1, z=1):
            if False:
                while True:
                    i = 10
            if p:
                if a:
                    if z:
                        pass
                    else:
                        pass
            else:
                pass
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (2, 'return')])

    def test_early_exit_with(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                return self

            def __exit__(*args):
                if False:
                    while True:
                        i = 10
                pass

        def func_break():
            if False:
                print('Hello World!')
            for i in (1, 2):
                with C():
                    break
            pass

        def func_return():
            if False:
                return 10
            with C():
                return
        self.run_and_compare(func_break, [(0, 'call'), (1, 'line'), (2, 'line'), (-5, 'call'), (-4, 'line'), (-4, 'return'), (3, 'line'), (2, 'line'), (-3, 'call'), (-2, 'line'), (-2, 'return'), (4, 'line'), (4, 'return')])
        self.run_and_compare(func_return, [(0, 'call'), (1, 'line'), (-11, 'call'), (-10, 'line'), (-10, 'return'), (2, 'line'), (1, 'line'), (-9, 'call'), (-8, 'line'), (-8, 'return'), (1, 'return')])

    def test_flow_converges_on_same_line(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                i = 10
                return i + 15
            if x:
                try:
                    1 / (x - 1)
                except ZeroDivisionError:
                    pass
            return x

        def func():
            if False:
                for i in range(10):
                    print('nop')
            for i in range(2):
                foo(i)
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (-8, 'call'), (-7, 'line'), (-2, 'line'), (-2, 'return'), (1, 'line'), (2, 'line'), (-8, 'call'), (-7, 'line'), (-6, 'line'), (-5, 'line'), (-5, 'exception'), (-4, 'line'), (-3, 'line'), (-2, 'line'), (-2, 'return'), (1, 'line'), (1, 'return')])

    def test_no_tracing_of_named_except_cleanup(self):
        if False:
            i = 10
            return i + 15

        def func():
            if False:
                return 10
            x = 0
            try:
                1 / x
            except ZeroDivisionError as error:
                if x:
                    raise
            return 'done'
        self.run_and_compare(func, [(0, 'call'), (1, 'line'), (2, 'line'), (3, 'line'), (3, 'exception'), (4, 'line'), (5, 'line'), (7, 'line'), (7, 'return')])

class SkipLineEventsTraceTestCase(TraceTestCase):
    """Repeat the trace tests, but with per-line events skipped"""

    def compare_events(self, line_offset, events, expected_events):
        if False:
            i = 10
            return i + 15
        skip_line_events = [e for e in expected_events if e[1] != 'line']
        super().compare_events(line_offset, events, skip_line_events)

    @staticmethod
    def make_tracer():
        if False:
            for i in range(10):
                print('nop')
        return Tracer(trace_line_events=False)

@support.cpython_only
class TraceOpcodesTestCase(TraceTestCase):
    """Repeat the trace tests, but with per-opcodes events enabled"""

    def compare_events(self, line_offset, events, expected_events):
        if False:
            print('Hello World!')
        skip_opcode_events = [e for e in events if e[1] != 'opcode']
        if len(events) > 1:
            self.assertLess(len(skip_opcode_events), len(events), msg="No 'opcode' events received by the tracer")
        super().compare_events(line_offset, skip_opcode_events, expected_events)

    @staticmethod
    def make_tracer():
        if False:
            return 10
        return Tracer(trace_opcode_events=True)

class RaisingTraceFuncTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.addCleanup(sys.settrace, sys.gettrace())

    def trace(self, frame, event, arg):
        if False:
            for i in range(10):
                print('nop')
        'A trace function that raises an exception in response to a\n        specific trace event.'
        if event == self.raiseOnEvent:
            raise ValueError
        else:
            return self.trace

    def f(self):
        if False:
            print('Hello World!')
        "The function to trace; raises an exception if that's the case\n        we're testing, so that the 'exception' trace event fires."
        if self.raiseOnEvent == 'exception':
            x = 0
            y = 1 / x
        else:
            return 1

    def run_test_for_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Tests that an exception raised in response to the given event is\n        handled OK.'
        self.raiseOnEvent = event
        try:
            for i in range(sys.getrecursionlimit() + 1):
                sys.settrace(self.trace)
                try:
                    self.f()
                except ValueError:
                    pass
                else:
                    self.fail('exception not raised!')
        except RuntimeError:
            self.fail('recursion counter not reset')

    def test_call(self):
        if False:
            while True:
                i = 10
        self.run_test_for_event('call')

    def test_line(self):
        if False:
            return 10
        self.run_test_for_event('line')

    def test_return(self):
        if False:
            return 10
        self.run_test_for_event('return')

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test_for_event('exception')

    def test_trash_stack(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                while True:
                    i = 10
            for i in range(5):
                print(i)

        def g(frame, why, extra):
            if False:
                i = 10
                return i + 15
            if why == 'line' and frame.f_lineno == f.__code__.co_firstlineno + 2:
                raise RuntimeError('i am crashing')
            return g
        sys.settrace(g)
        try:
            f()
        except RuntimeError:
            import gc
            gc.collect()
        else:
            self.fail('exception not propagated')

    def test_exception_arguments(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            x = 0
            x.no_such_attr

        def g(frame, event, arg):
            if False:
                return 10
            if event == 'exception':
                (type, exception, trace) = arg
                self.assertIsInstance(exception, Exception)
            return g
        existing = sys.gettrace()
        try:
            sys.settrace(g)
            try:
                f()
            except AttributeError:
                pass
        finally:
            sys.settrace(existing)

class JumpTracer:
    """Defines a trace function that jumps from one place to another."""

    def __init__(self, function, jumpFrom, jumpTo, event='line', decorated=False):
        if False:
            while True:
                i = 10
        self.code = function.__code__
        self.jumpFrom = jumpFrom
        self.jumpTo = jumpTo
        self.event = event
        self.firstLine = None if decorated else self.code.co_firstlineno
        self.done = False

    def trace(self, frame, event, arg):
        if False:
            while True:
                i = 10
        if self.done:
            return
        if self.firstLine is None and frame.f_code == self.code and (event == 'line'):
            self.firstLine = frame.f_lineno - 1
        if event == self.event and self.firstLine is not None and (frame.f_lineno == self.firstLine + self.jumpFrom):
            f = frame
            while f is not None and f.f_code != self.code:
                f = f.f_back
            if f is not None:
                try:
                    frame.f_lineno = self.firstLine + self.jumpTo
                except TypeError:
                    frame.f_lineno = self.jumpTo
                self.done = True
        return self.trace

def no_jump_to_non_integers(output):
    if False:
        i = 10
        return i + 15
    try:
        output.append(2)
    except ValueError as e:
        output.append('integer' in str(e))

def no_jump_without_trace_function():
    if False:
        for i in range(10):
            print('nop')
    try:
        previous_frame = sys._getframe().f_back
        previous_frame.f_lineno = previous_frame.f_lineno
    except ValueError as e:
        if 'trace' not in str(e):
            raise
    else:
        raise AssertionError('Trace-function-less jump failed to fail')

class JumpTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(None)

    def compare_jump_output(self, expected, received):
        if False:
            i = 10
            return i + 15
        if received != expected:
            self.fail("Outputs don't match:\n" + 'Expected: ' + repr(expected) + '\n' + 'Received: ' + repr(received))

    def run_test(self, func, jumpFrom, jumpTo, expected, error=None, event='line', decorated=False):
        if False:
            i = 10
            return i + 15
        tracer = JumpTracer(func, jumpFrom, jumpTo, event, decorated)
        sys.settrace(tracer.trace)
        output = []
        if error is None:
            func(output)
        else:
            with self.assertRaisesRegex(*error):
                func(output)
        sys.settrace(None)
        self.compare_jump_output(expected, output)

    def run_async_test(self, func, jumpFrom, jumpTo, expected, error=None, event='line', decorated=False):
        if False:
            print('Hello World!')
        tracer = JumpTracer(func, jumpFrom, jumpTo, event, decorated)
        sys.settrace(tracer.trace)
        output = []
        if error is None:
            asyncio.run(func(output))
        else:
            with self.assertRaisesRegex(*error):
                asyncio.run(func(output))
        sys.settrace(None)
        asyncio.set_event_loop_policy(None)
        self.compare_jump_output(expected, output)

    def jump_test(jumpFrom, jumpTo, expected, error=None, event='line'):
        if False:
            for i in range(10):
                print('nop')
        'Decorator that creates a test that makes a jump\n        from one place to another in the following code.\n        '

        def decorator(func):
            if False:
                while True:
                    i = 10

            @wraps(func)
            def test(self):
                if False:
                    while True:
                        i = 10
                self.run_test(func, jumpFrom, jumpTo, expected, error=error, event=event, decorated=True)
            return test
        return decorator

    def async_jump_test(jumpFrom, jumpTo, expected, error=None, event='line'):
        if False:
            i = 10
            return i + 15
        'Decorator that creates a test that makes a jump\n        from one place to another in the following asynchronous code.\n        '

        def decorator(func):
            if False:
                print('Hello World!')

            @wraps(func)
            def test(self):
                if False:
                    i = 10
                    return i + 15
                self.run_async_test(func, jumpFrom, jumpTo, expected, error=error, event=event, decorated=True)
            return test
        return decorator

    @jump_test(1, 3, [3])
    def test_jump_simple_forwards(output):
        if False:
            print('Hello World!')
        output.append(1)
        output.append(2)
        output.append(3)

    @jump_test(2, 1, [1, 1, 2])
    def test_jump_simple_backwards(output):
        if False:
            while True:
                i = 10
        output.append(1)
        output.append(2)

    @jump_test(3, 5, [2, 5])
    def test_jump_out_of_block_forwards(output):
        if False:
            return 10
        for i in (1, 2):
            output.append(2)
            for j in [3]:
                output.append(4)
        output.append(5)

    @jump_test(6, 1, [1, 3, 5, 1, 3, 5, 6, 7])
    def test_jump_out_of_block_backwards(output):
        if False:
            print('Hello World!')
        output.append(1)
        for i in [1]:
            output.append(3)
            for j in [2]:
                output.append(5)
            output.append(6)
        output.append(7)

    @async_jump_test(4, 5, [3, 5])
    async def test_jump_out_of_async_for_block_forwards(output):
        for i in [1]:
            async for i in asynciter([1, 2]):
                output.append(3)
                output.append(4)
            output.append(5)

    @async_jump_test(5, 2, [2, 4, 2, 4, 5, 6])
    async def test_jump_out_of_async_for_block_backwards(output):
        for i in [1]:
            output.append(2)
            async for i in asynciter([1]):
                output.append(4)
                output.append(5)
            output.append(6)

    @jump_test(1, 2, [3])
    def test_jump_to_codeless_line(output):
        if False:
            return 10
        output.append(1)
        output.append(3)

    @jump_test(2, 2, [1, 2, 3])
    def test_jump_to_same_line(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        output.append(2)
        output.append(3)

    @jump_test(4, 9, [2, 9])
    def test_jump_in_nested_finally(output):
        if False:
            i = 10
            return i + 15
        try:
            output.append(2)
        finally:
            output.append(4)
            try:
                output.append(6)
            finally:
                output.append(8)
            output.append(9)

    @jump_test(6, 7, [2, 7], (ZeroDivisionError, ''))
    def test_jump_in_nested_finally_2(output):
        if False:
            i = 10
            return i + 15
        try:
            output.append(2)
            1 / 0
            return
        finally:
            output.append(6)
            output.append(7)
        output.append(8)

    @jump_test(6, 11, [2, 11], (ZeroDivisionError, ''))
    def test_jump_in_nested_finally_3(output):
        if False:
            return 10
        try:
            output.append(2)
            1 / 0
            return
        finally:
            output.append(6)
            try:
                output.append(8)
            finally:
                output.append(10)
            output.append(11)
        output.append(12)

    @jump_test(5, 11, [2, 4], (ValueError, 'after'))
    def test_no_jump_over_return_try_finally_in_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        try:
            output.append(2)
        finally:
            output.append(4)
            output.append(5)
            return
            try:
                output.append(8)
            finally:
                output.append(10)
            pass
        output.append(12)

    @jump_test(3, 4, [1], (ValueError, 'after'))
    def test_no_jump_infinite_while_loop(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        while True:
            output.append(3)
        output.append(4)

    @jump_test(2, 4, [4, 4])
    def test_jump_forwards_into_while_block(output):
        if False:
            for i in range(10):
                print('nop')
        i = 1
        output.append(2)
        while i <= 2:
            output.append(4)
            i += 1

    @jump_test(5, 3, [3, 3, 3, 5])
    def test_jump_backwards_into_while_block(output):
        if False:
            for i in range(10):
                print('nop')
        i = 1
        while i <= 2:
            output.append(3)
            i += 1
        output.append(5)

    @jump_test(2, 3, [1, 3])
    def test_jump_forwards_out_of_with_block(output):
        if False:
            print('Hello World!')
        with tracecontext(output, 1):
            output.append(2)
        output.append(3)

    @async_jump_test(2, 3, [1, 3])
    async def test_jump_forwards_out_of_async_with_block(output):
        async with asynctracecontext(output, 1):
            output.append(2)
        output.append(3)

    @jump_test(3, 1, [1, 2, 1, 2, 3, -2])
    def test_jump_backwards_out_of_with_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        with tracecontext(output, 2):
            output.append(3)

    @async_jump_test(3, 1, [1, 2, 1, 2, 3, -2])
    async def test_jump_backwards_out_of_async_with_block(output):
        output.append(1)
        async with asynctracecontext(output, 2):
            output.append(3)

    @jump_test(2, 5, [5])
    def test_jump_forwards_out_of_try_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        try:
            output.append(2)
        finally:
            output.append(4)
        output.append(5)

    @jump_test(3, 1, [1, 1, 3, 5])
    def test_jump_backwards_out_of_try_finally_block(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        try:
            output.append(3)
        finally:
            output.append(5)

    @jump_test(2, 6, [6])
    def test_jump_forwards_out_of_try_except_block(output):
        if False:
            return 10
        try:
            output.append(2)
        except:
            output.append(4)
            raise
        output.append(6)

    @jump_test(3, 1, [1, 1, 3])
    def test_jump_backwards_out_of_try_except_block(output):
        if False:
            return 10
        output.append(1)
        try:
            output.append(3)
        except:
            output.append(5)
            raise

    @jump_test(5, 7, [4, 7, 8])
    def test_jump_between_except_blocks(output):
        if False:
            i = 10
            return i + 15
        try:
            1 / 0
        except ZeroDivisionError:
            output.append(4)
            output.append(5)
        except FloatingPointError:
            output.append(7)
        output.append(8)

    @jump_test(5, 6, [4, 6, 7])
    def test_jump_within_except_block(output):
        if False:
            while True:
                i = 10
        try:
            1 / 0
        except:
            output.append(4)
            output.append(5)
            output.append(6)
        output.append(7)

    @jump_test(2, 4, [1, 4, 5, -4])
    def test_jump_across_with(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        with tracecontext(output, 2):
            output.append(3)
        with tracecontext(output, 4):
            output.append(5)

    @async_jump_test(2, 4, [1, 4, 5, -4])
    async def test_jump_across_async_with(output):
        output.append(1)
        async with asynctracecontext(output, 2):
            output.append(3)
        async with asynctracecontext(output, 4):
            output.append(5)

    @jump_test(4, 5, [1, 3, 5, 6])
    def test_jump_out_of_with_block_within_for_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        for i in [1]:
            with tracecontext(output, 3):
                output.append(4)
            output.append(5)
        output.append(6)

    @async_jump_test(4, 5, [1, 3, 5, 6])
    async def test_jump_out_of_async_with_block_within_for_block(output):
        output.append(1)
        for i in [1]:
            async with asynctracecontext(output, 3):
                output.append(4)
            output.append(5)
        output.append(6)

    @jump_test(4, 5, [1, 2, 3, 5, -2, 6])
    def test_jump_out_of_with_block_within_with_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        with tracecontext(output, 2):
            with tracecontext(output, 3):
                output.append(4)
            output.append(5)
        output.append(6)

    @async_jump_test(4, 5, [1, 2, 3, 5, -2, 6])
    async def test_jump_out_of_async_with_block_within_with_block(output):
        output.append(1)
        with tracecontext(output, 2):
            async with asynctracecontext(output, 3):
                output.append(4)
            output.append(5)
        output.append(6)

    @jump_test(5, 6, [2, 4, 6, 7])
    def test_jump_out_of_with_block_within_finally_block(output):
        if False:
            print('Hello World!')
        try:
            output.append(2)
        finally:
            with tracecontext(output, 4):
                output.append(5)
            output.append(6)
        output.append(7)

    @async_jump_test(5, 6, [2, 4, 6, 7])
    async def test_jump_out_of_async_with_block_within_finally_block(output):
        try:
            output.append(2)
        finally:
            async with asynctracecontext(output, 4):
                output.append(5)
            output.append(6)
        output.append(7)

    @jump_test(8, 11, [1, 3, 5, 11, 12])
    def test_jump_out_of_complex_nested_blocks(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        for i in [1]:
            output.append(3)
            for j in [1, 2]:
                output.append(5)
                try:
                    for k in [1, 2]:
                        output.append(8)
                finally:
                    output.append(10)
            output.append(11)
        output.append(12)

    @jump_test(3, 5, [1, 2, 5])
    def test_jump_out_of_with_assignment(output):
        if False:
            while True:
                i = 10
        output.append(1)
        with tracecontext(output, 2) as x:
            output.append(4)
        output.append(5)

    @async_jump_test(3, 5, [1, 2, 5])
    async def test_jump_out_of_async_with_assignment(output):
        output.append(1)
        async with asynctracecontext(output, 2) as x:
            output.append(4)
        output.append(5)

    @jump_test(3, 6, [1, 6, 8, 9])
    def test_jump_over_return_in_try_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        try:
            output.append(3)
            if not output:
                return
            output.append(6)
        finally:
            output.append(8)
        output.append(9)

    @jump_test(5, 8, [1, 3, 8, 10, 11, 13])
    def test_jump_over_break_in_try_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        while True:
            output.append(3)
            try:
                output.append(5)
                if not output:
                    break
                output.append(8)
            finally:
                output.append(10)
            output.append(11)
            break
        output.append(13)

    @jump_test(1, 7, [7, 8])
    def test_jump_over_for_block_before_else(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        if not output:
            for i in [3]:
                output.append(4)
        else:
            output.append(6)
            output.append(7)
        output.append(8)

    @async_jump_test(1, 7, [7, 8])
    async def test_jump_over_async_for_block_before_else(output):
        output.append(1)
        if not output:
            async for i in asynciter([3]):
                output.append(4)
        else:
            output.append(6)
            output.append(7)
        output.append(8)

    @jump_test(2, 3, [1], (ValueError, 'after'))
    def test_no_jump_too_far_forwards(output):
        if False:
            while True:
                i = 10
        output.append(1)
        output.append(2)

    @jump_test(2, -2, [1], (ValueError, 'before'))
    def test_no_jump_too_far_backwards(output):
        if False:
            print('Hello World!')
        output.append(1)
        output.append(2)

    @jump_test(2, 3, [4], (ValueError, 'except'))
    def test_no_jump_to_except_1(output):
        if False:
            return 10
        try:
            output.append(2)
        except:
            output.append(4)
            raise

    @jump_test(2, 3, [4], (ValueError, 'except'))
    def test_no_jump_to_except_2(output):
        if False:
            while True:
                i = 10
        try:
            output.append(2)
        except ValueError:
            output.append(4)
            raise

    @jump_test(2, 3, [4], (ValueError, 'except'))
    def test_no_jump_to_except_3(output):
        if False:
            print('Hello World!')
        try:
            output.append(2)
        except ValueError as e:
            output.append(4)
            raise e

    @jump_test(2, 3, [4], (ValueError, 'except'))
    def test_no_jump_to_except_4(output):
        if False:
            print('Hello World!')
        try:
            output.append(2)
        except (ValueError, RuntimeError) as e:
            output.append(4)
            raise e

    @jump_test(1, 3, [], (ValueError, 'into'))
    def test_no_jump_forwards_into_for_block(output):
        if False:
            return 10
        output.append(1)
        for i in (1, 2):
            output.append(3)

    @async_jump_test(1, 3, [], (ValueError, 'into'))
    async def test_no_jump_forwards_into_async_for_block(output):
        output.append(1)
        async for i in asynciter([1, 2]):
            output.append(3)
        pass

    @jump_test(3, 2, [2, 2], (ValueError, 'into'))
    def test_no_jump_backwards_into_for_block(output):
        if False:
            return 10
        for i in (1, 2):
            output.append(2)
        output.append(3)

    @async_jump_test(3, 2, [2, 2], (ValueError, 'into'))
    async def test_no_jump_backwards_into_async_for_block(output):
        async for i in asynciter([1, 2]):
            output.append(2)
        output.append(3)

    @jump_test(1, 3, [], (ValueError, 'into'))
    def test_no_jump_forwards_into_with_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        with tracecontext(output, 2):
            output.append(3)

    @async_jump_test(1, 3, [], (ValueError, 'into'))
    async def test_no_jump_forwards_into_async_with_block(output):
        output.append(1)
        async with asynctracecontext(output, 2):
            output.append(3)

    @jump_test(3, 2, [1, 2, -1], (ValueError, 'into'))
    def test_no_jump_backwards_into_with_block(output):
        if False:
            i = 10
            return i + 15
        with tracecontext(output, 1):
            output.append(2)
        output.append(3)

    @async_jump_test(3, 2, [1, 2, -1], (ValueError, 'into'))
    async def test_no_jump_backwards_into_async_with_block(output):
        async with asynctracecontext(output, 1):
            output.append(2)
        output.append(3)

    @jump_test(1, 3, [], (ValueError, 'into'))
    def test_no_jump_forwards_into_try_finally_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        try:
            output.append(3)
        finally:
            output.append(5)

    @jump_test(5, 2, [2, 4], (ValueError, 'into'))
    def test_no_jump_backwards_into_try_finally_block(output):
        if False:
            print('Hello World!')
        try:
            output.append(2)
        finally:
            output.append(4)
        output.append(5)

    @jump_test(1, 3, [], (ValueError, 'into'))
    def test_no_jump_forwards_into_try_except_block(output):
        if False:
            return 10
        output.append(1)
        try:
            output.append(3)
        except:
            output.append(5)
            raise

    @jump_test(6, 2, [2], (ValueError, 'into'))
    def test_no_jump_backwards_into_try_except_block(output):
        if False:
            return 10
        try:
            output.append(2)
        except:
            output.append(4)
            raise
        output.append(6)

    @jump_test(5, 7, [4], (ValueError, 'into'))
    def test_no_jump_between_except_blocks_2(output):
        if False:
            for i in range(10):
                print('nop')
        try:
            1 / 0
        except ZeroDivisionError:
            output.append(4)
            output.append(5)
        except FloatingPointError as e:
            output.append(7)
        output.append(8)

    @jump_test(1, 5, [5])
    def test_jump_into_finally_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        try:
            output.append(3)
        finally:
            output.append(5)

    @jump_test(3, 6, [2, 6, 7])
    def test_jump_into_finally_block_from_try_block(output):
        if False:
            for i in range(10):
                print('nop')
        try:
            output.append(2)
            output.append(3)
        finally:
            output.append(5)
            output.append(6)
        output.append(7)

    @jump_test(5, 1, [1, 3, 1, 3, 5])
    def test_jump_out_of_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        try:
            output.append(3)
        finally:
            output.append(5)

    @jump_test(1, 5, [], (ValueError, "into an 'except'"))
    def test_no_jump_into_bare_except_block(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        try:
            output.append(3)
        except:
            output.append(5)

    @jump_test(1, 5, [], (ValueError, "into an 'except'"))
    def test_no_jump_into_qualified_except_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        try:
            output.append(3)
        except Exception:
            output.append(5)

    @jump_test(3, 6, [2, 5, 6], (ValueError, "into an 'except'"))
    def test_no_jump_into_bare_except_block_from_try_block(output):
        if False:
            return 10
        try:
            output.append(2)
            output.append(3)
        except:
            output.append(5)
            output.append(6)
            raise
        output.append(8)

    @jump_test(3, 6, [2], (ValueError, "into an 'except'"))
    def test_no_jump_into_qualified_except_block_from_try_block(output):
        if False:
            print('Hello World!')
        try:
            output.append(2)
            output.append(3)
        except ZeroDivisionError:
            output.append(5)
            output.append(6)
            raise
        output.append(8)

    @jump_test(7, 1, [1, 3, 6], (ValueError, "out of an 'except'"))
    def test_no_jump_out_of_bare_except_block(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        try:
            output.append(3)
            1 / 0
        except:
            output.append(6)
            output.append(7)

    @jump_test(7, 1, [1, 3, 6], (ValueError, "out of an 'except'"))
    def test_no_jump_out_of_qualified_except_block(output):
        if False:
            print('Hello World!')
        output.append(1)
        try:
            output.append(3)
            1 / 0
        except Exception:
            output.append(6)
            output.append(7)

    @jump_test(3, 5, [1, 2, 5, -2])
    def test_jump_between_with_blocks(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        with tracecontext(output, 2):
            output.append(3)
        with tracecontext(output, 4):
            output.append(5)

    @async_jump_test(3, 5, [1, 2, 5, -2])
    async def test_jump_between_async_with_blocks(output):
        output.append(1)
        async with asynctracecontext(output, 2):
            output.append(3)
        async with asynctracecontext(output, 4):
            output.append(5)

    @jump_test(5, 7, [2, 4], (ValueError, 'after'))
    def test_no_jump_over_return_out_of_finally_block(output):
        if False:
            for i in range(10):
                print('nop')
        try:
            output.append(2)
        finally:
            output.append(4)
            output.append(5)
            return
        output.append(7)

    @jump_test(7, 4, [1, 6], (ValueError, 'into'))
    def test_no_jump_into_for_block_before_else(output):
        if False:
            return 10
        output.append(1)
        if not output:
            for i in [3]:
                output.append(4)
        else:
            output.append(6)
            output.append(7)
        output.append(8)

    @async_jump_test(7, 4, [1, 6], (ValueError, 'into'))
    async def test_no_jump_into_async_for_block_before_else(output):
        output.append(1)
        if not output:
            async for i in asynciter([3]):
                output.append(4)
        else:
            output.append(6)
            output.append(7)
        output.append(8)

    def test_no_jump_to_non_integers(self):
        if False:
            while True:
                i = 10
        self.run_test(no_jump_to_non_integers, 2, 'Spam', [True])

    def test_no_jump_without_trace_function(self):
        if False:
            for i in range(10):
                print('nop')
        no_jump_without_trace_function()

    def test_large_function(self):
        if False:
            for i in range(10):
                print('nop')
        d = {}
        exec("def f(output):        # line 0\n            x = 0                     # line 1\n            y = 1                     # line 2\n            '''                       # line 3\n            %s                        # lines 4-1004\n            '''                       # line 1005\n            x += 1                    # line 1006\n            output.append(x)          # line 1007\n            return" % ('\n' * 1000,), d)
        f = d['f']
        self.run_test(f, 2, 1007, [0])

    def test_jump_to_firstlineno(self):
        if False:
            return 10
        code = compile("\n# Comments don't count.\noutput.append(2)  # firstlineno is here.\noutput.append(3)\noutput.append(4)\n", '<fake module>', 'exec')

        class fake_function:
            __code__ = code
        tracer = JumpTracer(fake_function, 4, 1)
        sys.settrace(tracer.trace)
        namespace = {'output': []}
        exec(code, namespace)
        sys.settrace(None)
        self.compare_jump_output([2, 3, 2, 3, 4], namespace['output'])

    @jump_test(2, 3, [1], event='call', error=(ValueError, "can't jump from the 'call' trace event of a new frame"))
    def test_no_jump_from_call(output):
        if False:
            return 10
        output.append(1)

        def nested():
            if False:
                return 10
            output.append(3)
        nested()
        output.append(5)

    @jump_test(2, 1, [1], event='return', error=(ValueError, "can only jump from a 'line' trace event"))
    def test_no_jump_from_return_event(output):
        if False:
            print('Hello World!')
        output.append(1)
        return

    @jump_test(2, 1, [1], event='exception', error=(ValueError, "can only jump from a 'line' trace event"))
    def test_no_jump_from_exception_event(output):
        if False:
            i = 10
            return i + 15
        output.append(1)
        1 / 0

    @jump_test(3, 2, [2, 5], event='return')
    def test_jump_from_yield(output):
        if False:
            i = 10
            return i + 15

        def gen():
            if False:
                while True:
                    i = 10
            output.append(2)
            yield 3
        next(gen())
        output.append(5)

    @jump_test(2, 3, [1, 3])
    def test_jump_forward_over_listcomp(output):
        if False:
            for i in range(10):
                print('nop')
        output.append(1)
        x = [i for i in range(10)]
        output.append(3)

    @jump_test(3, 1, [])
    def test_jump_backward_over_listcomp(output):
        if False:
            i = 10
            return i + 15
        a = 1
        x = [i for i in range(10)]
        c = 3

    @jump_test(8, 2, [2, 7, 2])
    def test_jump_backward_over_listcomp_v2(output):
        if False:
            print('Hello World!')
        flag = False
        output.append(2)
        if flag:
            return
        x = [i for i in range(5)]
        flag = 6
        output.append(7)
        output.append(8)

    @async_jump_test(2, 3, [1, 3])
    async def test_jump_forward_over_async_listcomp(output):
        output.append(1)
        x = [i async for i in asynciter(range(10))]
        output.append(3)

    @async_jump_test(3, 1, [])
    async def test_jump_backward_over_async_listcomp(output):
        a = 1
        x = [i async for i in asynciter(range(10))]
        c = 3

    @async_jump_test(8, 2, [2, 7, 2])
    async def test_jump_backward_over_async_listcomp_v2(output):
        flag = False
        output.append(2)
        if flag:
            return
        x = [i async for i in asynciter(range(5))]
        flag = 6
        output.append(7)
        output.append(8)
if __name__ == '__main__':
    unittest.main()