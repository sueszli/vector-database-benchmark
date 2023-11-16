import gc
import pprint
import sys
import unittest

class TestGetProfile(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        sys.setprofile(None)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        sys.setprofile(None)

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(sys.getprofile())

    def test_setget(self):
        if False:
            i = 10
            return i + 15

        def fn(*args):
            if False:
                return 10
            pass
        sys.setprofile(fn)
        self.assertIs(sys.getprofile(), fn)

class HookWatcher:

    def __init__(self):
        if False:
            return 10
        self.frames = []
        self.events = []

    def callback(self, frame, event, arg):
        if False:
            for i in range(10):
                print('nop')
        if event == 'call' or event == 'return' or event == 'exception':
            self.add_event(event, frame)

    def add_event(self, event, frame=None):
        if False:
            return 10
        'Add an event to the log.'
        if frame is None:
            frame = sys._getframe(1)
        try:
            frameno = self.frames.index(frame)
        except ValueError:
            frameno = len(self.frames)
            self.frames.append(frame)
        self.events.append((frameno, event, ident(frame)))

    def get_events(self):
        if False:
            return 10
        'Remove calls to add_event().'
        disallowed = [ident(self.add_event.__func__), ident(ident)]
        self.frames = None
        return [item for item in self.events if item[2] not in disallowed]

class ProfileSimulator(HookWatcher):

    def __init__(self, testcase):
        if False:
            for i in range(10):
                print('nop')
        self.testcase = testcase
        self.stack = []
        HookWatcher.__init__(self)

    def callback(self, frame, event, arg):
        if False:
            print('Hello World!')
        self.dispatch[event](self, frame)

    def trace_call(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.add_event('call', frame)
        self.stack.append(frame)

    def trace_return(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.add_event('return', frame)
        self.stack.pop()

    def trace_exception(self, frame):
        if False:
            return 10
        self.testcase.fail('the profiler should never receive exception events')

    def trace_pass(self, frame):
        if False:
            while True:
                i = 10
        pass
    dispatch = {'call': trace_call, 'exception': trace_exception, 'return': trace_return, 'c_call': trace_pass, 'c_return': trace_pass, 'c_exception': trace_pass}

class TestCaseBase(unittest.TestCase):

    def check_events(self, callable, expected):
        if False:
            while True:
                i = 10
        events = capture_events(callable, self.new_watcher())
        if events != expected:
            self.fail('Expected events:\n%s\nReceived events:\n%s' % (pprint.pformat(expected), pprint.pformat(events)))

class ProfileHookTestCase(TestCaseBase):

    def new_watcher(self):
        if False:
            while True:
                i = 10
        return HookWatcher()

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(p):
            if False:
                return 10
            pass
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_exception(self):
        if False:
            i = 10
            return i + 15

        def f(p):
            if False:
                for i in range(10):
                    print('nop')
            1 / 0
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_caught_exception(self):
        if False:
            print('Hello World!')

        def f(p):
            if False:
                i = 10
                return i + 15
            try:
                1 / 0
            except:
                pass
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_caught_nested_exception(self):
        if False:
            print('Hello World!')

        def f(p):
            if False:
                return 10
            try:
                1 / 0
            except:
                pass
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_nested_exception(self):
        if False:
            print('Hello World!')

        def f(p):
            if False:
                while True:
                    i = 10
            1 / 0
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_exception_in_except_clause(self):
        if False:
            i = 10
            return i + 15

        def f(p):
            if False:
                while True:
                    i = 10
            1 / 0

        def g(p):
            if False:
                return 10
            try:
                f(p)
            except:
                try:
                    f(p)
                except:
                    pass
        f_ident = ident(f)
        g_ident = ident(g)
        self.check_events(g, [(1, 'call', g_ident), (2, 'call', f_ident), (2, 'return', f_ident), (3, 'call', f_ident), (3, 'return', f_ident), (1, 'return', g_ident)])

    def test_exception_propagation(self):
        if False:
            return 10

        def f(p):
            if False:
                for i in range(10):
                    print('nop')
            1 / 0

        def g(p):
            if False:
                while True:
                    i = 10
            try:
                f(p)
            finally:
                p.add_event('falling through')
        f_ident = ident(f)
        g_ident = ident(g)
        self.check_events(g, [(1, 'call', g_ident), (2, 'call', f_ident), (2, 'return', f_ident), (1, 'falling through', g_ident), (1, 'return', g_ident)])

    def test_raise_twice(self):
        if False:
            print('Hello World!')

        def f(p):
            if False:
                return 10
            try:
                1 / 0
            except:
                1 / 0
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_raise_reraise(self):
        if False:
            return 10

        def f(p):
            if False:
                for i in range(10):
                    print('nop')
            try:
                1 / 0
            except:
                raise
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_raise(self):
        if False:
            for i in range(10):
                print('nop')

        def f(p):
            if False:
                while True:
                    i = 10
            raise Exception()
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_distant_exception(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                print('Hello World!')
            1 / 0

        def g():
            if False:
                print('Hello World!')
            f()

        def h():
            if False:
                while True:
                    i = 10
            g()

        def i():
            if False:
                return 10
            h()

        def j(p):
            if False:
                for i in range(10):
                    print('nop')
            i()
        f_ident = ident(f)
        g_ident = ident(g)
        h_ident = ident(h)
        i_ident = ident(i)
        j_ident = ident(j)
        self.check_events(j, [(1, 'call', j_ident), (2, 'call', i_ident), (3, 'call', h_ident), (4, 'call', g_ident), (5, 'call', f_ident), (5, 'return', f_ident), (4, 'return', g_ident), (3, 'return', h_ident), (2, 'return', i_ident), (1, 'return', j_ident)])

    def test_generator(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            for i in range(2):
                yield i

        def g(p):
            if False:
                print('Hello World!')
            for i in f():
                pass
        f_ident = ident(f)
        g_ident = ident(g)
        self.check_events(g, [(1, 'call', g_ident), (2, 'call', f_ident), (2, 'return', f_ident), (2, 'call', f_ident), (2, 'return', f_ident), (2, 'call', f_ident), (2, 'return', f_ident), (1, 'return', g_ident)])

    def test_stop_iteration(self):
        if False:
            return 10

        def f():
            if False:
                for i in range(10):
                    print('nop')
            for i in range(2):
                yield i

        def g(p):
            if False:
                while True:
                    i = 10
            for i in f():
                pass
        f_ident = ident(f)
        g_ident = ident(g)
        self.check_events(g, [(1, 'call', g_ident), (2, 'call', f_ident), (2, 'return', f_ident), (2, 'call', f_ident), (2, 'return', f_ident), (2, 'call', f_ident), (2, 'return', f_ident), (1, 'return', g_ident)])

class ProfileSimulatorTestCase(TestCaseBase):

    def new_watcher(self):
        if False:
            return 10
        return ProfileSimulator(self)

    def test_simple(self):
        if False:
            return 10

        def f(p):
            if False:
                for i in range(10):
                    print('nop')
            pass
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_basic_exception(self):
        if False:
            while True:
                i = 10

        def f(p):
            if False:
                i = 10
                return i + 15
            1 / 0
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_caught_exception(self):
        if False:
            for i in range(10):
                print('nop')

        def f(p):
            if False:
                i = 10
                return i + 15
            try:
                1 / 0
            except:
                pass
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_distant_exception(self):
        if False:
            return 10

        def f():
            if False:
                i = 10
                return i + 15
            1 / 0

        def g():
            if False:
                print('Hello World!')
            f()

        def h():
            if False:
                return 10
            g()

        def i():
            if False:
                return 10
            h()

        def j(p):
            if False:
                return 10
            i()
        f_ident = ident(f)
        g_ident = ident(g)
        h_ident = ident(h)
        i_ident = ident(i)
        j_ident = ident(j)
        self.check_events(j, [(1, 'call', j_ident), (2, 'call', i_ident), (3, 'call', h_ident), (4, 'call', g_ident), (5, 'call', f_ident), (5, 'return', f_ident), (4, 'return', g_ident), (3, 'return', h_ident), (2, 'return', i_ident), (1, 'return', j_ident)])

    def test_unbound_method(self):
        if False:
            print('Hello World!')
        kwargs = {}

        def f(p):
            if False:
                while True:
                    i = 10
            dict.get({}, 42, **kwargs)
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_unbound_method_no_args(self):
        if False:
            return 10

        def f(p):
            if False:
                while True:
                    i = 10
            dict.get()
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_unbound_method_invalid_args(self):
        if False:
            return 10

        def f(p):
            if False:
                i = 10
                return i + 15
            dict.get(print, 42)
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_unbound_method_no_keyword_args(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {}

        def f(p):
            if False:
                print('Hello World!')
            dict.get(**kwargs)
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

    def test_unbound_method_invalid_keyword_args(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {}

        def f(p):
            if False:
                while True:
                    i = 10
            dict.get(print, 42, **kwargs)
        f_ident = ident(f)
        self.check_events(f, [(1, 'call', f_ident), (1, 'return', f_ident)])

def ident(function):
    if False:
        return 10
    if hasattr(function, 'f_code'):
        code = function.f_code
    else:
        code = function.__code__
    return (code.co_firstlineno, code.co_name)

def protect(f, p):
    if False:
        while True:
            i = 10
    try:
        f(p)
    except:
        pass
protect_ident = ident(protect)

def capture_events(callable, p=None):
    if False:
        for i in range(10):
            print('nop')
    if p is None:
        p = HookWatcher()
    old_gc = gc.isenabled()
    gc.disable()
    try:
        sys.setprofile(p.callback)
        protect(callable, p)
        sys.setprofile(None)
    finally:
        if old_gc:
            gc.enable()
    return p.get_events()[1:-1]

def show_events(callable):
    if False:
        while True:
            i = 10
    import pprint
    pprint.pprint(capture_events(callable))
if __name__ == '__main__':
    unittest.main()