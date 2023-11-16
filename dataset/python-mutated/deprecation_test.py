"""Deprecation tests."""
import collections
import enum
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import strict_mode
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect

class DeprecatedAliasTest(test.TestCase):

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_function_alias(self, mock_warning):
        if False:
            print('Hello World!')
        deprecated_func = deprecation.deprecated_alias('deprecated.func', 'real.func', logging.error)
        logging.error('fake error logged')
        self.assertEqual(0, mock_warning.call_count)
        deprecated_func('FAKE ERROR!')
        self.assertEqual(1, mock_warning.call_count)
        self.assertRegex(mock_warning.call_args[0][1], 'deprecation_test\\.py:')
        deprecated_func('ANOTHER FAKE ERROR!')
        self.assertEqual(1, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_class_alias(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')

        class MyClass(object):
            """My docstring."""
            init_args = []

            def __init__(self, arg):
                if False:
                    return 10
                MyClass.init_args.append(arg)
        deprecated_cls = deprecation.deprecated_alias('deprecated.cls', 'real.cls', MyClass)
        print(deprecated_cls.__name__)
        print(deprecated_cls.__module__)
        print(deprecated_cls.__doc__)
        MyClass('test')
        self.assertEqual(0, mock_warning.call_count)
        deprecated_cls('deprecated')
        self.assertEqual(1, mock_warning.call_count)
        self.assertRegex(mock_warning.call_args[0][1], 'deprecation_test\\.py:')
        deprecated_cls('deprecated again')
        self.assertEqual(1, mock_warning.call_count)
        self.assertEqual(['test', 'deprecated', 'deprecated again'], MyClass.init_args)
        self.assertEqual(tf_inspect.getfullargspec(MyClass.__init__), tf_inspect.getfullargspec(deprecated_cls.__init__))

class DeprecationTest(test.TestCase):

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_once(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=True)
        def _fn():
            if False:
                return 10
            pass
        _fn()
        self.assertEqual(1, mock_warning.call_count)
        _fn()
        self.assertEqual(1, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_init_class(self, mock_warning):
        if False:
            print('Hello World!')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=True)
        class MyClass:
            """A test class."""

            def __init__(self, a):
                if False:
                    return 10
                pass
        MyClass('')
        self.assertEqual(1, mock_warning.call_count)
        MyClass('')
        self.assertEqual(1, mock_warning.call_count)
        self.assertIn('IS DEPRECATED', MyClass.__doc__)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_new_class(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=True)
        class MyStr(str):

            def __new__(cls, value):
                if False:
                    return 10
                return str.__new__(cls, value)
        MyStr('abc')
        self.assertEqual(1, mock_warning.call_count)
        MyStr('abc')
        self.assertEqual(1, mock_warning.call_count)
        self.assertIn('IS DEPRECATED', MyStr.__doc__)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_enum(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=True)
        class MyEnum(enum.Enum):
            a = 1
            b = 2
        self.assertIs(MyEnum(1), MyEnum.a)
        self.assertEqual(1, mock_warning.call_count)
        self.assertIs(MyEnum(2), MyEnum.b)
        self.assertEqual(1, mock_warning.call_count)
        self.assertIn('IS DEPRECATED', MyEnum.__doc__)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_namedtuple(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'
        mytuple = deprecation.deprecated(date, instructions, warn_once=True)(collections.namedtuple('my_tuple', ['field1', 'field2']))
        mytuple(1, 2)
        self.assertEqual(1, mock_warning.call_count)
        mytuple(3, 4)
        self.assertEqual(1, mock_warning.call_count)
        self.assertIn('IS DEPRECATED', mytuple.__doc__)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_silence(self, mock_warning):
        if False:
            return 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=False)
        def _fn():
            if False:
                return 10
            pass
        _fn()
        self.assertEqual(1, mock_warning.call_count)
        with deprecation.silence():
            _fn()
        self.assertEqual(1, mock_warning.call_count)
        _fn()
        self.assertEqual(2, mock_warning.call_count)

    def test_strict_mode_deprecation(self):
        if False:
            return 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions, warn_once=True)
        def _fn():
            if False:
                i = 10
                return i + 15
            pass
        strict_mode.enable_strict_mode()
        with self.assertRaises(RuntimeError):
            _fn()

    def _assert_subset(self, expected_subset, actual_set):
        if False:
            print('Hello World!')
        self.assertTrue(actual_set.issuperset(expected_subset), msg='%s is not a superset of %s.' % (actual_set, expected_subset))

    def test_deprecated_illegal_args(self):
        if False:
            print('Hello World!')
        instructions = 'This is how you update...'
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated('', instructions)
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated('07-04-2016', instructions)
        date = '2016-07-04'
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated(date, None)
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated(date, '')

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_no_date(self, mock_warning):
        if False:
            while True:
                i = 10
        date = None
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions)
        def _fn(arg0, arg1):
            if False:
                i = 10
                return i + 15
            'fn doc.\n\n      Args:\n        arg0: Arg 0.\n        arg1: Arg 1.\n\n      Returns:\n        Sum of args.\n      '
            return arg0 + arg1
        self.assertEqual('fn doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.\nInstructions for updating:\n%s\n\nArgs:\n  arg0: Arg 0.\n  arg1: Arg 1.\n\nReturns:\n  Sum of args.' % instructions, _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['in a future version', instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_doc(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions)
        def _fn(arg0, arg1):
            if False:
                print('Hello World!')
            'fn doc.\n\n      Args:\n        arg0: Arg 0.\n        arg1: Arg 1.\n\n      Returns:\n        Sum of args.\n      '
            return arg0 + arg1
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s\n\nArgs:\n  arg0: Arg 0.\n  arg1: Arg 1.\n\nReturns:\n  Sum of args.' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_one_line_doc(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions)
        def _fn(arg0, arg1):
            if False:
                i = 10
                return i + 15
            'fn doc.'
            return arg0 + arg1
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_no_doc(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated(date, instructions)
        def _fn(arg0, arg1):
            if False:
                for i in range(10):
                    print('nop')
            return arg0 + arg1
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('DEPRECATED FUNCTION\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_instance_fn_with_doc(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        class _Object(object):

            def __init(self):
                if False:
                    return 10
                pass

            @deprecation.deprecated(date, instructions)
            def _fn(self, arg0, arg1):
                if False:
                    i = 10
                    return i + 15
                'fn doc.\n\n        Args:\n          arg0: Arg 0.\n          arg1: Arg 1.\n\n        Returns:\n          Sum of args.\n        '
                return arg0 + arg1
        self.assertEqual('fn doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s\n\nArgs:\n  arg0: Arg 0.\n  arg1: Arg 1.\n\nReturns:\n  Sum of args.' % (date, instructions), getattr(_Object, '_fn').__doc__)
        self.assertEqual(3, _Object()._fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_instance_fn_with_one_line_doc(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        class _Object(object):

            def __init(self):
                if False:
                    print('Hello World!')
                pass

            @deprecation.deprecated(date, instructions)
            def _fn(self, arg0, arg1):
                if False:
                    i = 10
                    return i + 15
                'fn doc.'
                return arg0 + arg1
        self.assertEqual('fn doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), getattr(_Object, '_fn').__doc__)
        self.assertEqual(3, _Object()._fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_instance_fn_no_doc(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        class _Object(object):

            def __init(self):
                if False:
                    return 10
                pass

            @deprecation.deprecated(date, instructions)
            def _fn(self, arg0, arg1):
                if False:
                    for i in range(10):
                        print('nop')
                return arg0 + arg1
        self.assertEqual('DEPRECATED FUNCTION\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), getattr(_Object, '_fn').__doc__)
        self.assertEqual(3, _Object()._fn(1, 2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    def test_prop_wrong_order(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'make sure @property appears before @deprecated in your source code'):

            class _Object(object):

                def __init(self):
                    if False:
                        return 10
                    pass

                @deprecation.deprecated('2016-07-04', 'Instructions.')
                @property
                def _prop(self):
                    if False:
                        i = 10
                        return i + 15
                    return 'prop_wrong_order'

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_prop_with_doc(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'

        class _Object(object):

            def __init(self):
                if False:
                    while True:
                        i = 10
                pass

            @property
            @deprecation.deprecated(date, instructions)
            def _prop(self):
                if False:
                    return 10
                'prop doc.\n\n        Returns:\n          String.\n        '
                return 'prop_with_doc'
        self.assertEqual('prop doc. (deprecated)\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s\n\nReturns:\n  String.' % (date, instructions), getattr(_Object, '_prop').__doc__)
        self.assertEqual('prop_with_doc', _Object()._prop)
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_prop_no_doc(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'

        class _Object(object):

            def __init(self):
                if False:
                    i = 10
                    return i + 15
                pass

            @property
            @deprecation.deprecated(date, instructions)
            def _prop(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'prop_no_doc'
        self.assertEqual('DEPRECATED FUNCTION\n\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), getattr(_Object, '_prop').__doc__)
        self.assertEqual('prop_no_doc', _Object()._prop)
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

class DeprecatedArgsTest(test.TestCase):

    def _assert_subset(self, expected_subset, actual_set):
        if False:
            i = 10
            return i + 15
        self.assertTrue(actual_set.issuperset(expected_subset), msg='%s is not a superset of %s.' % (actual_set, expected_subset))

    def test_deprecated_illegal_args(self):
        if False:
            print('Hello World!')
        instructions = 'This is how you update...'
        date = '2016-07-04'
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated_args('', instructions, 'deprecated')
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated_args('07-04-2016', instructions, 'deprecated')
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated_args(date, None, 'deprecated')
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated_args(date, '', 'deprecated')
        with self.assertRaisesRegex(ValueError, 'argument'):
            deprecation.deprecated_args(date, instructions)

    def test_deprecated_missing_args(self):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        def _fn(arg0, arg1, deprecated=None):
            if False:
                print('Hello World!')
            return arg0 + arg1 if deprecated else arg1 + arg0
        with self.assertRaisesRegex(ValueError, "not present.*\\['missing'\\]"):
            deprecation.deprecated_args(date, instructions, 'missing')(_fn)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_doc(self, mock_warning):
        if False:
            print('Hello World!')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(arg0, arg1, deprecated=True):
            if False:
                return 10
            'fn doc.\n\n      Args:\n        arg0: Arg 0.\n        arg1: Arg 1.\n        deprecated: Deprecated!\n\n      Returns:\n        Sum of args.\n      '
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated arguments)\n\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. They will be removed after %s.\nInstructions for updating:\n%s\n\nArgs:\n  arg0: Arg 0.\n  arg1: Arg 1.\n  deprecated: Deprecated!\n\nReturns:\n  Sum of args.' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_one_line_doc(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(arg0, arg1, deprecated=True):
            if False:
                while True:
                    i = 10
            'fn doc.'
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated arguments)\n\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. They will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_no_doc(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(arg0, arg1, deprecated=True):
            if False:
                for i in range(10):
                    print('nop')
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('DEPRECATED FUNCTION ARGUMENTS\n\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. They will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_varargs(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(arg0, arg1, *deprecated):
            if False:
                for i in range(10):
                    print('nop')
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, True, False))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_kwargs(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(arg0, arg1, **deprecated):
            if False:
                print('Hello World!')
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, a=True, b=False))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_positional_and_named(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'd1', 'd2')
        def _fn(arg0, d1=None, arg1=2, d2=None):
            if False:
                print('Hello World!')
            return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1
        self.assertEqual(2, _fn(1, arg1=2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(2, _fn(1, None, 2, d2=False))
        self.assertEqual(2, mock_warning.call_count)
        (args1, _) = mock_warning.call_args_list[0]
        self.assertRegex(args1[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions, 'd1']), set(args1[1:]))
        (args2, _) = mock_warning.call_args_list[1]
        self.assertRegex(args2[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions, 'd2']), set(args2[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_positional_and_named_with_ok_vals(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, ('d1', None), ('d2', 'my_ok_val'))
        def _fn(arg0, d1=None, arg1=2, d2=None):
            if False:
                for i in range(10):
                    print('nop')
            return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1
        self.assertEqual(2, _fn(1, arg1=2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(2, _fn(1, False, 2, d2=False))
        self.assertEqual(2, mock_warning.call_count)
        (args1, _) = mock_warning.call_args_list[0]
        self.assertRegex(args1[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions, 'd1']), set(args1[1:]))
        (args2, _) = mock_warning.call_args_list[1]
        self.assertRegex(args2[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions, 'd2']), set(args2[1:]))
        mock_warning.reset_mock()
        self.assertEqual(3, _fn(1, None, 2, d2='my_ok_val'))
        self.assertEqual(0, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_kwonlyargs(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'deprecated')
        def _fn(*, arg0, arg1, deprecated=None):
            if False:
                i = 10
                return i + 15
            return arg0 + arg1 if deprecated is not None else arg1 + arg0
        self.assertEqual(3, _fn(arg0=1, arg1=2))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(arg0=1, arg1=2, deprecated=2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_kwonlyargs_and_args(self, mock_warning):
        if False:
            return 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, ('deprecated_arg1', 'deprecated_arg2'))
        def _fn(arg0, arg1, *, kw1, deprecated_arg1=None, deprecated_arg2=None):
            if False:
                return 10
            res = arg0 + arg1 + kw1
            if deprecated_arg1 is not None:
                res += deprecated_arg1
            if deprecated_arg2 is not None:
                res += deprecated_arg2
            return res
        self.assertEqual(6, _fn(1, 2, kw1=3))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(8, _fn(1, 2, kw1=3, deprecated_arg1=2))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))
        self.assertEqual(12, _fn(1, 2, kw1=3, deprecated_arg1=2, deprecated_arg2=4))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_deprecated_args_once(self, mock_warning):
        if False:
            for i in range(10):
                print('nop')
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'arg', warn_once=True)
        def _fn(arg=0):
            if False:
                return 10
            pass
        _fn()
        self.assertEqual(0, mock_warning.call_count)
        _fn(arg=0)
        self.assertEqual(1, mock_warning.call_count)
        _fn(arg=1)
        self.assertEqual(1, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_deprecated_multiple_args_once_each(self, mock_warning):
        if False:
            return 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_args(date, instructions, 'arg0', 'arg1', warn_once=True)
        def _fn(arg0=0, arg1=0):
            if False:
                for i in range(10):
                    print('nop')
            pass
        _fn(arg0=0)
        self.assertEqual(1, mock_warning.call_count)
        _fn(arg0=0)
        self.assertEqual(1, mock_warning.call_count)
        _fn(arg1=0)
        self.assertEqual(2, mock_warning.call_count)
        _fn(arg0=0)
        self.assertEqual(2, mock_warning.call_count)
        _fn(arg1=0)
        self.assertEqual(2, mock_warning.call_count)

class DeprecatedArgValuesTest(test.TestCase):

    def _assert_subset(self, expected_subset, actual_set):
        if False:
            i = 10
            return i + 15
        self.assertTrue(actual_set.issuperset(expected_subset), msg='%s is not a superset of %s.' % (actual_set, expected_subset))

    def test_deprecated_illegal_args(self):
        if False:
            while True:
                i = 10
        instructions = 'This is how you update...'
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated_arg_values('', instructions, deprecated=True)
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            deprecation.deprecated_arg_values('07-04-2016', instructions, deprecated=True)
        date = '2016-07-04'
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated_arg_values(date, None, deprecated=True)
        with self.assertRaisesRegex(ValueError, 'instructions'):
            deprecation.deprecated_arg_values(date, '', deprecated=True)
        with self.assertRaisesRegex(ValueError, 'argument'):
            deprecation.deprecated_arg_values(date, instructions)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_doc(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_arg_values(date, instructions, warn_once=False, deprecated=True)
        def _fn(arg0, arg1, deprecated=True):
            if False:
                for i in range(10):
                    print('nop')
            'fn doc.\n\n      Args:\n        arg0: Arg 0.\n        arg1: Arg 1.\n        deprecated: Deprecated!\n\n      Returns:\n        Sum of args.\n      '
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated argument values)\n\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. They will be removed after %s.\nInstructions for updating:\n%s\n\nArgs:\n  arg0: Arg 0.\n  arg1: Arg 1.\n  deprecated: Deprecated!\n\nReturns:\n  Sum of args.' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2, deprecated=False))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, deprecated=True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(2, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_with_one_line_doc(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_arg_values(date, instructions, warn_once=False, deprecated=True)
        def _fn(arg0, arg1, deprecated=True):
            if False:
                print('Hello World!')
            'fn doc.'
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('fn doc. (deprecated argument values)\n\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. They will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2, deprecated=False))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, deprecated=True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(2, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_deprecated_v1
    def test_static_fn_no_doc(self, mock_warning):
        if False:
            i = 10
            return i + 15
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_arg_values(date, instructions, warn_once=False, deprecated=True)
        def _fn(arg0, arg1, deprecated=True):
            if False:
                for i in range(10):
                    print('nop')
            return arg0 + arg1 if deprecated else arg1 + arg0
        self.assertEqual('_fn', _fn.__name__)
        self.assertEqual('DEPRECATED FUNCTION ARGUMENT VALUES\n\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. They will be removed after %s.\nInstructions for updating:\n%s' % (date, instructions), _fn.__doc__)
        self.assertEqual(3, _fn(1, 2, deprecated=False))
        self.assertEqual(0, mock_warning.call_count)
        self.assertEqual(3, _fn(1, 2, deprecated=True))
        self.assertEqual(1, mock_warning.call_count)
        (args, _) = mock_warning.call_args
        self.assertRegex(args[0], 'deprecated and will be removed')
        self._assert_subset(set(['after ' + date, instructions]), set(args[1:]))
        self.assertEqual(3, _fn(1, 2))
        self.assertEqual(2, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_arg_values_once(self, mock_warning):
        if False:
            while True:
                i = 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_arg_values(date, instructions, warn_once=True, deprecated=True)
        def _fn(deprecated):
            if False:
                return 10
            pass
        _fn(deprecated=False)
        self.assertEqual(0, mock_warning.call_count)
        _fn(deprecated=True)
        self.assertEqual(1, mock_warning.call_count)
        _fn(deprecated=True)
        self.assertEqual(1, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def test_deprecated_multiple_arg_values_once_each(self, mock_warning):
        if False:
            return 10
        date = '2016-07-04'
        instructions = 'This is how you update...'

        @deprecation.deprecated_arg_values(date, instructions, warn_once=True, arg0='forbidden', arg1='disallowed')
        def _fn(arg0, arg1):
            if False:
                return 10
            pass
        _fn(arg0='allowed', arg1='also allowed')
        self.assertEqual(0, mock_warning.call_count)
        _fn(arg0='forbidden', arg1='disallowed')
        self.assertEqual(2, mock_warning.call_count)
        _fn(arg0='forbidden', arg1='allowed')
        self.assertEqual(2, mock_warning.call_count)
        _fn(arg0='forbidden', arg1='disallowed')
        self.assertEqual(2, mock_warning.call_count)

    @test.mock.patch.object(logging, 'warning', autospec=True)
    @test_util.run_in_graph_and_eager_modes
    def test_deprecated_arg_values_when_value_is_none(self, mock_warning):
        if False:
            print('Hello World!')

        @deprecation.deprecated_arg_values('2016-07-04', 'This is how you update...', warn_once=True, arg0=None)
        def _fn(arg0):
            if False:
                i = 10
                return i + 15
            pass
        tensor.enable_tensor_equality()
        initial_count = mock_warning.call_count
        _fn(arg0=variables.Variable(0))
        self.assertEqual(initial_count, mock_warning.call_count)
        _fn(arg0=None)
        self.assertEqual(initial_count + 1, mock_warning.call_count)
        tensor.disable_tensor_equality()

class DeprecationArgumentsTest(test.TestCase):

    def testDeprecatedArgumentLookup(self):
        if False:
            i = 10
            return i + 15
        good_value = 3
        self.assertEqual(deprecation.deprecated_argument_lookup('val_new', good_value, 'val_old', None), good_value)
        self.assertEqual(deprecation.deprecated_argument_lookup('val_new', None, 'val_old', good_value), good_value)
        with self.assertRaisesRegex(ValueError, "Cannot specify both 'val_old' and 'val_new'"):
            deprecation.deprecated_argument_lookup('val_new', good_value, 'val_old', good_value)

    def testRewriteArgumentDocstring(self):
        if False:
            i = 10
            return i + 15
        docs = 'Add `a` and `b`\n\n    Args:\n      a: first arg\n      b: second arg\n    '
        new_docs = deprecation.rewrite_argument_docstring(deprecation.rewrite_argument_docstring(docs, 'a', 'left'), 'b', 'right')
        new_docs_ref = 'Add `left` and `right`\n\n    Args:\n      left: first arg\n      right: second arg\n    '
        self.assertEqual(new_docs, new_docs_ref)

class DeprecatedEndpointsTest(test.TestCase):

    def testSingleDeprecatedEndpoint(self):
        if False:
            print('Hello World!')

        @deprecation.deprecated_endpoints('foo1')
        def foo():
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(('foo1',), foo._tf_deprecated_api_names)

    def testMultipleDeprecatedEndpoint(self):
        if False:
            while True:
                i = 10

        @deprecation.deprecated_endpoints('foo1', 'foo2')
        def foo():
            if False:
                print('Hello World!')
            pass
        self.assertEqual(('foo1', 'foo2'), foo._tf_deprecated_api_names)

    def testCannotSetDeprecatedEndpointsTwice(self):
        if False:
            return 10
        with self.assertRaises(deprecation.DeprecatedNamesAlreadySetError):

            @deprecation.deprecated_endpoints('foo1')
            @deprecation.deprecated_endpoints('foo2')
            def foo():
                if False:
                    while True:
                        i = 10
                pass

class DeprecateMovedModuleTest(test.TestCase):

    @test.mock.patch.object(logging, 'warning', autospec=True)
    def testCallDeprecatedModule(self, mock_warning):
        if False:
            return 10
        from tensorflow.python.util import deprecated_module
        self.assertEqual(0, mock_warning.call_count)
        result = deprecated_module.a()
        self.assertEqual(1, mock_warning.call_count)
        self.assertEqual(1, result)
        deprecated_module.a()
        self.assertEqual(1, mock_warning.call_count)
if __name__ == '__main__':
    test.main()