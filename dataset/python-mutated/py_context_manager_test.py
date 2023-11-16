"""Tests for tensorflow.python.framework._py_context_manager."""
from tensorflow.python.framework import _py_context_manager
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class TestContextManager(object):

    def __init__(self, behavior='basic'):
        if False:
            while True:
                i = 10
        self.log = []
        self.behavior = behavior

    def __enter__(self):
        if False:
            return 10
        self.log.append('__enter__()')
        if self.behavior == 'raise_from_enter':
            raise ValueError('exception in __enter__')
        return 'var'

    def __exit__(self, ex_type, ex_value, ex_tb):
        if False:
            i = 10
            return i + 15
        self.log.append('__exit__(%s, %s, %s)' % (ex_type, ex_value, ex_tb))
        if self.behavior == 'raise_from_exit':
            raise ValueError('exception in __exit__')
        if self.behavior == 'suppress_exception':
            return True
NO_EXCEPTION_LOG = "__enter__()\nbody('var')\n__exit__(None, None, None)"
EXCEPTION_LOG = "__enter__\\(\\)\nbody\\('var'\\)\n__exit__\\(<class 'ValueError'>, Foo, <traceback object.*>\\)"

class OpDefUtilTest(test_util.TensorFlowTestCase):

    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        cm = TestContextManager()

        def body(var):
            if False:
                print('Hello World!')
            cm.log.append('body(%r)' % var)
        _py_context_manager.test_py_context_manager(cm, body)
        self.assertEqual('\n'.join(cm.log), NO_EXCEPTION_LOG)

    def testBodyRaisesException(self):
        if False:
            for i in range(10):
                print('nop')
        cm = TestContextManager()

        def body(var):
            if False:
                return 10
            cm.log.append('body(%r)' % var)
            raise ValueError('Foo')
        with self.assertRaisesRegexp(ValueError, 'Foo'):
            _py_context_manager.test_py_context_manager(cm, body)
        self.assertRegex('\n'.join(cm.log), EXCEPTION_LOG)

    def testEnterRaisesException(self):
        if False:
            print('Hello World!')
        cm = TestContextManager('raise_from_enter')

        def body(var):
            if False:
                print('Hello World!')
            cm.log.append('body(%r)' % var)
        with self.assertRaisesRegexp(ValueError, 'exception in __enter__'):
            _py_context_manager.test_py_context_manager(cm, body)
        self.assertEqual('\n'.join(cm.log), '__enter__()')

    def testExitRaisesException(self):
        if False:
            i = 10
            return i + 15
        cm = TestContextManager('raise_from_exit')

        def body(var):
            if False:
                for i in range(10):
                    print('nop')
            cm.log.append('body(%r)' % var)
        _py_context_manager.test_py_context_manager(cm, body)
        self.assertEqual('\n'.join(cm.log), NO_EXCEPTION_LOG)

    def testExitSuppressesException(self):
        if False:
            while True:
                i = 10
        cm = TestContextManager('suppress_exception')

        def body(var):
            if False:
                for i in range(10):
                    print('nop')
            cm.log.append('body(%r)' % var)
            raise ValueError('Foo')
        with self.assertRaisesRegexp(ValueError, 'tensorflow::PyContextManager::Enter does not support context managers that suppress exception'):
            _py_context_manager.test_py_context_manager(cm, body)
        self.assertRegex('\n'.join(cm.log), EXCEPTION_LOG)
if __name__ == '__main__':
    googletest.main()