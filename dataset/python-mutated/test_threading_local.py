import sys
import unittest
from doctest import DocTestSuite
from test import support
from test.support import threading_helper
import weakref
import gc
import _thread
import threading
import _threading_local

class Weak(object):
    pass

def target(local, weaklist):
    if False:
        for i in range(10):
            print('nop')
    weak = Weak()
    local.weak = weak
    weaklist.append(weakref.ref(weak))

class BaseLocalTest:

    def test_local_refs(self):
        if False:
            return 10
        self._local_refs(20)
        self._local_refs(50)
        self._local_refs(100)

    def _local_refs(self, n):
        if False:
            print('Hello World!')
        local = self._local()
        weaklist = []
        for i in range(n):
            t = threading.Thread(target=target, args=(local, weaklist))
            t.start()
            t.join()
        del t
        support.gc_collect()
        self.assertEqual(len(weaklist), n)
        deadlist = [weak for weak in weaklist if weak() is None]
        self.assertIn(len(deadlist), (n - 1, n))
        local.someothervar = None
        support.gc_collect()
        deadlist = [weak for weak in weaklist if weak() is None]
        self.assertIn(len(deadlist), (n - 1, n), (n, len(deadlist)))

    def test_derived(self):
        if False:
            while True:
                i = 10
        import time

        class Local(self._local):

            def __init__(self):
                if False:
                    return 10
                time.sleep(0.01)
        local = Local()

        def f(i):
            if False:
                print('Hello World!')
            local.x = i
            self.assertEqual(local.x, i)
        with threading_helper.start_threads((threading.Thread(target=f, args=(i,)) for i in range(10))):
            pass

    def test_derived_cycle_dealloc(self):
        if False:
            i = 10
            return i + 15

        class Local(self._local):
            pass
        locals = None
        passed = False
        e1 = threading.Event()
        e2 = threading.Event()

        def f():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal passed
            cycle = [Local()]
            cycle.append(cycle)
            cycle[0].foo = 'bar'
            del cycle
            support.gc_collect()
            e1.set()
            e2.wait()
            passed = all((not hasattr(local, 'foo') for local in locals))
        t = threading.Thread(target=f)
        t.start()
        e1.wait()
        locals = [Local() for i in range(10)]
        e2.set()
        t.join()
        self.assertTrue(passed)

    def test_arguments(self):
        if False:
            i = 10
            return i + 15

        class MyLocal(self._local):

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pass
        MyLocal(a=1)
        MyLocal(1)
        self.assertRaises(TypeError, self._local, a=1)
        self.assertRaises(TypeError, self._local, 1)

    def _test_one_class(self, c):
        if False:
            for i in range(10):
                print('nop')
        self._failed = 'No error message set or cleared.'
        obj = c()
        e1 = threading.Event()
        e2 = threading.Event()

        def f1():
            if False:
                print('Hello World!')
            obj.x = 'foo'
            obj.y = 'bar'
            del obj.y
            e1.set()
            e2.wait()

        def f2():
            if False:
                print('Hello World!')
            try:
                foo = obj.x
            except AttributeError:
                self._failed = ''
            else:
                self._failed = 'Incorrectly got value %r from class %r\n' % (foo, c)
                sys.stderr.write(self._failed)
        t1 = threading.Thread(target=f1)
        t1.start()
        e1.wait()
        t2 = threading.Thread(target=f2)
        t2.start()
        t2.join()
        e2.set()
        t1.join()
        self.assertFalse(self._failed, self._failed)

    def test_threading_local(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_one_class(self._local)

    def test_threading_local_subclass(self):
        if False:
            i = 10
            return i + 15

        class LocalSubclass(self._local):
            """To test that subclasses behave properly."""
        self._test_one_class(LocalSubclass)

    def _test_dict_attribute(self, cls):
        if False:
            print('Hello World!')
        obj = cls()
        obj.x = 5
        self.assertEqual(obj.__dict__, {'x': 5})
        with self.assertRaises(AttributeError):
            obj.__dict__ = {}
        with self.assertRaises(AttributeError):
            del obj.__dict__

    def test_dict_attribute(self):
        if False:
            print('Hello World!')
        self._test_dict_attribute(self._local)

    def test_dict_attribute_subclass(self):
        if False:
            i = 10
            return i + 15

        class LocalSubclass(self._local):
            """To test that subclasses behave properly."""
        self._test_dict_attribute(LocalSubclass)

    def test_cycle_collection(self):
        if False:
            while True:
                i = 10

        class X:
            pass
        x = X()
        x.local = self._local()
        x.local.x = x
        wr = weakref.ref(x)
        del x
        support.gc_collect()
        self.assertIsNone(wr())

class ThreadLocalTest(unittest.TestCase, BaseLocalTest):
    _local = _thread._local

class PyThreadingLocalTest(unittest.TestCase, BaseLocalTest):
    _local = _threading_local.local

def test_main():
    if False:
        i = 10
        return i + 15
    suite = unittest.TestSuite()
    suite.addTest(DocTestSuite('_threading_local'))
    suite.addTest(unittest.makeSuite(ThreadLocalTest))
    suite.addTest(unittest.makeSuite(PyThreadingLocalTest))
    local_orig = _threading_local.local

    def setUp(test):
        if False:
            for i in range(10):
                print('nop')
        _threading_local.local = _thread._local

    def tearDown(test):
        if False:
            i = 10
            return i + 15
        _threading_local.local = local_orig
    suite.addTest(DocTestSuite('_threading_local', setUp=setUp, tearDown=tearDown))
    support.run_unittest(suite)
if __name__ == '__main__':
    test_main()