"""decorator_utils tests."""
import functools
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils

def _test_function(unused_arg=0):
    if False:
        while True:
            i = 10
    pass

class GetQualifiedNameTest(test.TestCase):

    def test_method(self):
        if False:
            return 10
        self.assertEqual('GetQualifiedNameTest.test_method', decorator_utils.get_qualified_name(GetQualifiedNameTest.test_method))

    def test_function(self):
        if False:
            while True:
                i = 10
        self.assertEqual('_test_function', decorator_utils.get_qualified_name(_test_function))

class AddNoticeToDocstringTest(test.TestCase):

    def _check(self, doc, expected):
        if False:
            return 10
        self.assertEqual(decorator_utils.add_notice_to_docstring(doc=doc, instructions='Instructions', no_doc_str='Nothing here', suffix_str='(suffix)', notice=['Go away']), expected)

    def test_regular(self):
        if False:
            i = 10
            return i + 15
        expected = 'Brief (suffix)\n\nWarning: Go away\nInstructions\n\nDocstring\n\nArgs:\n  arg1: desc'
        self._check('Brief\n\nDocstring\n\nArgs:\n  arg1: desc', expected)
        self._check('Brief\n\n  Docstring\n\n  Args:\n    arg1: desc', expected)
        self._check('Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc', expected)
        self._check('\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc', expected)
        self._check('\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc', expected)

    def test_brief_only(self):
        if False:
            while True:
                i = 10
        expected = 'Brief (suffix)\n\nWarning: Go away\nInstructions'
        self._check('Brief', expected)
        self._check('Brief\n', expected)
        self._check('Brief\n  ', expected)
        self._check('\nBrief\n  ', expected)
        self._check('\n  Brief\n  ', expected)

    def test_no_docstring(self):
        if False:
            return 10
        expected = 'Nothing here\n\nWarning: Go away\nInstructions'
        self._check(None, expected)
        self._check('', expected)

    def test_no_empty_line(self):
        if False:
            return 10
        expected = 'Brief (suffix)\n\nWarning: Go away\nInstructions\n\nDocstring'
        self._check('Brief\nDocstring', expected)
        self._check('Brief\n  Docstring', expected)
        self._check('\nBrief\nDocstring', expected)
        self._check('\n  Brief\n  Docstring', expected)

class ValidateCallableTest(test.TestCase):

    def test_function(self):
        if False:
            for i in range(10):
                print('nop')
        decorator_utils.validate_callable(_test_function, 'test')

    def test_method(self):
        if False:
            print('Hello World!')
        decorator_utils.validate_callable(self.test_method, 'test')

    def test_callable(self):
        if False:
            while True:
                i = 10

        class TestClass(object):

            def __call__(self):
                if False:
                    while True:
                        i = 10
                pass
        decorator_utils.validate_callable(TestClass(), 'test')

    def test_partial(self):
        if False:
            while True:
                i = 10
        partial = functools.partial(_test_function, unused_arg=7)
        decorator_utils.validate_callable(partial, 'test')

    def test_fail_non_callable(self):
        if False:
            for i in range(10):
                print('nop')
        x = 0
        self.assertRaises(ValueError, decorator_utils.validate_callable, x, 'test')

class CachedClassPropertyTest(test.TestCase):

    def testCachedClassProperty(self):
        if False:
            return 10
        log = []

        class MyClass(object):

            @decorator_utils.cached_classproperty
            def value(cls):
                if False:
                    print('Hello World!')
                log.append(cls)
                return cls.__name__

        class MySubclass(MyClass):
            pass
        self.assertLen(log, 0)
        self.assertEqual(MyClass.value, 'MyClass')
        self.assertEqual(log, [MyClass])
        self.assertEqual(MyClass.value, 'MyClass')
        self.assertEqual(MyClass.value, 'MyClass')
        self.assertEqual(log, [MyClass])
        self.assertEqual(MySubclass.value, 'MySubclass')
        self.assertEqual(log, [MyClass, MySubclass])
        self.assertEqual(MySubclass.value, 'MySubclass')
        self.assertEqual(MySubclass.value, 'MySubclass')
        self.assertEqual(log, [MyClass, MySubclass])
        self.assertEqual(MyClass().value, 'MyClass')
        self.assertEqual(MySubclass().value, 'MySubclass')
        self.assertEqual(log, [MyClass, MySubclass])
        with self.assertRaises(AttributeError):
            MyClass().value = 12
        with self.assertRaises(AttributeError):
            del MyClass().value
if __name__ == '__main__':
    test.main()