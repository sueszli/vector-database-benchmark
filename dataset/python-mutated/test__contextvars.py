from gevent import monkey
monkey.patch_all()
import concurrent.futures
try:
    import contextvars
except ImportError:
    from gevent import contextvars
import functools
import random
import time
import unittest
hamt = None

def isolated_context(func):
    if False:
        while True:
            i = 10
    'Needed to make reftracking test mode work.'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        ctx = contextvars.Context()
        return ctx.run(func, *args, **kwargs)
    return wrapper

class ContextTest(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertRaisesRegex'):
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    def test_context_var_new_1(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            contextvars.ContextVar()
        c = contextvars.ContextVar('aaa')
        self.assertEqual(c.name, 'aaa')
        with self.assertRaises(AttributeError):
            c.name = 'bbb'
        self.assertNotEqual(hash(c), hash('aaa'))

    @isolated_context
    def test_context_var_repr_1(self):
        if False:
            i = 10
            return i + 15
        c = contextvars.ContextVar('a')
        self.assertIn('a', repr(c))
        c = contextvars.ContextVar('a', default=123)
        self.assertIn('123', repr(c))
        lst = []
        c = contextvars.ContextVar('a', default=lst)
        lst.append(c)
        self.assertIn('...', repr(c))
        self.assertIn('...', repr(lst))
        t = c.set(1)
        self.assertIn(repr(c), repr(t))
        self.assertNotIn(' used ', repr(t))
        c.reset(t)
        self.assertIn(' used ', repr(t))

    def test_context_new_1(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            contextvars.Context(1)
        with self.assertRaises(TypeError):
            contextvars.Context(1, a=1)
        with self.assertRaises(TypeError):
            contextvars.Context(a=1)
        contextvars.Context(**{})

    def test_context_typerrors_1(self):
        if False:
            return 10
        ctx = contextvars.Context()
        with self.assertRaisesRegex(TypeError, 'ContextVar key was expected'):
            ctx[1]
        with self.assertRaisesRegex(TypeError, 'ContextVar key was expected'):
            1 in ctx
        with self.assertRaisesRegex(TypeError, 'ContextVar key was expected'):
            ctx.get(1)

    def test_context_get_context_1(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = contextvars.copy_context()
        self.assertIsInstance(ctx, contextvars.Context)

    def test_context_run_2(self):
        if False:
            i = 10
            return i + 15
        ctx = contextvars.Context()

        def func(*args, **kwargs):
            if False:
                while True:
                    i = 10
            kwargs['spam'] = 'foo'
            args += ('bar',)
            return (args, kwargs)
        for f in (func, functools.partial(func)):
            self.assertEqual(ctx.run(f), (('bar',), {'spam': 'foo'}))
            self.assertEqual(ctx.run(f, 1), ((1, 'bar'), {'spam': 'foo'}))
            self.assertEqual(ctx.run(f, a=2), (('bar',), {'a': 2, 'spam': 'foo'}))
            self.assertEqual(ctx.run(f, 11, a=2), ((11, 'bar'), {'a': 2, 'spam': 'foo'}))
            a = {}
            self.assertEqual(ctx.run(f, 11, **a), ((11, 'bar'), {'spam': 'foo'}))
            self.assertEqual(a, {})

    def test_context_run_3(self):
        if False:
            while True:
                i = 10
        ctx = contextvars.Context()

        def func(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            ctx.run(func)
        with self.assertRaises(ZeroDivisionError):
            ctx.run(func, 1, 2)
        with self.assertRaises(ZeroDivisionError):
            ctx.run(func, 1, 2, a=123)

    @isolated_context
    def test_context_run_4(self):
        if False:
            return 10
        ctx1 = contextvars.Context()
        ctx2 = contextvars.Context()
        var = contextvars.ContextVar('var')

        def func2():
            if False:
                print('Hello World!')
            self.assertIsNone(var.get(None))

        def func1():
            if False:
                return 10
            self.assertIsNone(var.get(None))
            var.set('spam')
            ctx2.run(func2)
            self.assertEqual(var.get(None), 'spam')
            cur = contextvars.copy_context()
            self.assertEqual(len(cur), 1)
            self.assertEqual(cur[var], 'spam')
            return cur
        returned_ctx = ctx1.run(func1)
        self.assertEqual(ctx1, returned_ctx)
        self.assertEqual(returned_ctx[var], 'spam')
        self.assertIn(var, returned_ctx)

    def test_context_run_5(self):
        if False:
            while True:
                i = 10
        ctx = contextvars.Context()
        var = contextvars.ContextVar('var')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsNone(var.get(None))
            var.set('spam')
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            ctx.run(func)
        self.assertIsNone(var.get(None))

    def test_context_run_6(self):
        if False:
            i = 10
            return i + 15
        ctx = contextvars.Context()
        c = contextvars.ContextVar('a', default=0)

        def fun():
            if False:
                print('Hello World!')
            self.assertEqual(c.get(), 0)
            self.assertIsNone(ctx.get(c))
            c.set(42)
            self.assertEqual(c.get(), 42)
            self.assertEqual(ctx.get(c), 42)
        ctx.run(fun)

    def test_context_run_7(self):
        if False:
            return 10
        ctx = contextvars.Context()

        def fun():
            if False:
                i = 10
                return i + 15
            with self.assertRaisesRegex(RuntimeError, 'is already entered'):
                ctx.run(fun)
        ctx.run(fun)

    @isolated_context
    def test_context_getset_1(self):
        if False:
            i = 10
            return i + 15
        c = contextvars.ContextVar('c')
        with self.assertRaises(LookupError):
            c.get()
        self.assertIsNone(c.get(None))
        t0 = c.set(42)
        self.assertEqual(c.get(), 42)
        self.assertEqual(c.get(None), 42)
        self.assertIs(t0.old_value, t0.MISSING)
        self.assertIs(t0.old_value, contextvars.Token.MISSING)
        self.assertIs(t0.var, c)
        t = c.set('spam')
        self.assertEqual(c.get(), 'spam')
        self.assertEqual(c.get(None), 'spam')
        self.assertEqual(t.old_value, 42)
        c.reset(t)
        self.assertEqual(c.get(), 42)
        self.assertEqual(c.get(None), 42)
        c.set('spam2')
        with self.assertRaisesRegex(RuntimeError, 'has already been used'):
            c.reset(t)
        self.assertEqual(c.get(), 'spam2')
        ctx1 = contextvars.copy_context()
        self.assertIn(c, ctx1)
        c.reset(t0)
        with self.assertRaisesRegex(RuntimeError, 'has already been used'):
            c.reset(t0)
        self.assertIsNone(c.get(None))
        self.assertIn(c, ctx1)
        self.assertEqual(ctx1[c], 'spam2')
        self.assertEqual(ctx1.get(c, 'aa'), 'spam2')
        self.assertEqual(len(ctx1), 1)
        self.assertEqual(list(ctx1.items()), [(c, 'spam2')])
        self.assertEqual(list(ctx1.values()), ['spam2'])
        self.assertEqual(list(ctx1.keys()), [c])
        self.assertEqual(list(ctx1), [c])
        ctx2 = contextvars.copy_context()
        self.assertNotIn(c, ctx2)
        with self.assertRaises(KeyError):
            ctx2[c]
        self.assertEqual(ctx2.get(c, 'aa'), 'aa')
        self.assertEqual(len(ctx2), 0)
        self.assertEqual(list(ctx2), [])

    @isolated_context
    def test_context_getset_2(self):
        if False:
            return 10
        v1 = contextvars.ContextVar('v1')
        v2 = contextvars.ContextVar('v2')
        t1 = v1.set(42)
        with self.assertRaisesRegex(ValueError, 'by a different'):
            v2.reset(t1)

    @isolated_context
    def test_context_getset_3(self):
        if False:
            print('Hello World!')
        c = contextvars.ContextVar('c', default=42)
        ctx = contextvars.Context()

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(c.get(), 42)
            with self.assertRaises(KeyError):
                ctx[c]
            self.assertIsNone(ctx.get(c))
            self.assertEqual(ctx.get(c, 'spam'), 'spam')
            self.assertNotIn(c, ctx)
            self.assertEqual(list(ctx.keys()), [])
            t = c.set(1)
            self.assertEqual(list(ctx.keys()), [c])
            self.assertEqual(ctx[c], 1)
            c.reset(t)
            self.assertEqual(list(ctx.keys()), [])
            with self.assertRaises(KeyError):
                ctx[c]
        ctx.run(fun)

    @isolated_context
    def test_context_getset_4(self):
        if False:
            i = 10
            return i + 15
        c = contextvars.ContextVar('c', default=42)
        ctx = contextvars.Context()
        tok = ctx.run(c.set, 1)
        with self.assertRaisesRegex(ValueError, 'different Context'):
            c.reset(tok)

    @isolated_context
    def test_context_getset_5(self):
        if False:
            for i in range(10):
                print('nop')
        c = contextvars.ContextVar('c', default=42)
        c.set([])

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            c.set([])
            c.get().append(42)
            self.assertEqual(c.get(), [42])
        contextvars.copy_context().run(fun)
        self.assertEqual(c.get(), [])

    def test_context_copy_1(self):
        if False:
            while True:
                i = 10
        ctx1 = contextvars.Context()
        c = contextvars.ContextVar('c', default=42)

        def ctx1_fun():
            if False:
                return 10
            c.set(10)
            ctx2 = ctx1.copy()
            self.assertEqual(ctx2[c], 10)
            c.set(20)
            self.assertEqual(ctx1[c], 20)
            self.assertEqual(ctx2[c], 10)
            ctx2.run(ctx2_fun)
            self.assertEqual(ctx1[c], 20)
            self.assertEqual(ctx2[c], 30)

        def ctx2_fun():
            if False:
                while True:
                    i = 10
            self.assertEqual(c.get(), 10)
            c.set(30)
            self.assertEqual(c.get(), 30)
        ctx1.run(ctx1_fun)

    @isolated_context
    def test_context_threads_1(self):
        if False:
            return 10
        cvar = contextvars.ContextVar('cvar')

        def sub(num):
            if False:
                return 10
            for i in range(10):
                cvar.set(num + i)
                time.sleep(random.uniform(0.001, 0.05))
                self.assertEqual(cvar.get(), num + i)
            return num
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as tp:
            results = list(tp.map(sub, range(10)))
        self.assertEqual(results, list(range(10)))
if __name__ == '__main__':
    unittest.main()