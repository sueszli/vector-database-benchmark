"""Adds support for parameterized tests to Python's unittest TestCase class.

A parameterized test is a method in a test case that is invoked with different
argument tuples.

A simple example:

  class AdditionExample(parameterized.ParameterizedTestCase):
    @parameterized.Parameters(
       (1, 2, 3),
       (4, 5, 9),
       (1, 1, 3))
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)


Each invocation is a separate test case and properly isolated just
like a normal test method, with its own setUp/tearDown cycle. In the
example above, there are three separate testcases, one of which will
fail due to an assertion error (1 + 1 != 3).

Parameters for invididual test cases can be tuples (with positional parameters)
or dictionaries (with named parameters):

  class AdditionExample(parameterized.ParameterizedTestCase):
    @parameterized.Parameters(
       {'op1': 1, 'op2': 2, 'result': 3},
       {'op1': 4, 'op2': 5, 'result': 9},
    )
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)

If a parameterized test fails, the error message will show the
original test name (which is modified internally) and the arguments
for the specific invocation, which are part of the string returned by
the shortDescription() method on test cases.

The id method of the test, used internally by the unittest framework,
is also modified to show the arguments. To make sure that test names
stay the same across several invocations, object representations like

  >>> class Foo(object):
  ...  pass
  >>> repr(Foo())
  '<__main__.Foo object at 0x23d8610>'

are turned into '<__main__.Foo>'. For even more descriptive names,
especially in test logs, you can use the NamedParameters decorator. In
this case, only tuples are supported, and the first parameters has to
be a string (or an object that returns an apt name when converted via
str()):

  class NamedExample(parameterized.ParameterizedTestCase):
    @parameterized.NamedParameters(
       ('Normal', 'aa', 'aaa', True),
       ('EmptyPrefix', '', 'abc', True),
       ('BothEmpty', '', '', True))
    def testStartsWith(self, prefix, string, result):
      self.assertEqual(result, strings.startswith(prefix))

Named tests also have the benefit that they can be run individually
from the command line:

  $ testmodule.py NamedExample.testStartsWithNormal
  .
  --------------------------------------------------------------------
  Ran 1 test in 0.000s

  OK

Parameterized Classes
=====================
If invocation arguments are shared across test methods in a single
ParameterizedTestCase class, instead of decorating all test methods
individually, the class itself can be decorated:

  @parameterized.Parameters(
    (1, 2, 3)
    (4, 5, 9))
  class ArithmeticTest(parameterized.ParameterizedTestCase):
    def testAdd(self, arg1, arg2, result):
      self.assertEqual(arg1 + arg2, result)

    def testSubtract(self, arg2, arg2, result):
      self.assertEqual(result - arg1, arg2)

Inputs from Iterables
=====================
If parameters should be shared across several test cases, or are dynamically
created from other sources, a single non-tuple iterable can be passed into
the decorator. This iterable will be used to obtain the test cases:

  class AdditionExample(parameterized.ParameterizedTestCase):
    @parameterized.Parameters(
      c.op1, c.op2, c.result for c in testcases
    )
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)


Single-Argument Test Methods
============================
If a test method takes only one argument, the single argument does not need to
be wrapped into a tuple:

  class NegativeNumberExample(parameterized.ParameterizedTestCase):
    @parameterized.Parameters(
       -1, -3, -4, -5
    )
    def testIsNegative(self, arg):
      self.assertTrue(IsNegative(arg))
"""
__author__ = 'tmarek@google.com (Torsten Marek)'
import collections
import functools
import re
import types
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import uuid
import six
ADDR_RE = re.compile('\\<([a-zA-Z0-9_\\-\\.]+) object at 0x[a-fA-F0-9]+\\>')
_SEPARATOR = uuid.uuid1().hex
_FIRST_ARG = object()
_ARGUMENT_REPR = object()

def _CleanRepr(obj):
    if False:
        while True:
            i = 10
    return ADDR_RE.sub('<\\1>', repr(obj))

def _StrClass(cls):
    if False:
        print('Hello World!')
    return '%s.%s' % (cls.__module__, cls.__name__)

def _NonStringIterable(obj):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(obj, collections.Iterable) and (not isinstance(obj, six.string_types))

def _FormatParameterList(testcase_params):
    if False:
        print('Hello World!')
    if isinstance(testcase_params, collections.Mapping):
        return ', '.join(('%s=%s' % (argname, _CleanRepr(value)) for (argname, value) in testcase_params.items()))
    elif _NonStringIterable(testcase_params):
        return ', '.join(map(_CleanRepr, testcase_params))
    else:
        return _FormatParameterList((testcase_params,))

class _ParameterizedTestIter(object):
    """Callable and iterable class for producing new test cases."""

    def __init__(self, test_method, testcases, naming_type):
        if False:
            return 10
        'Returns concrete test functions for a test and a list of parameters.\n\n    The naming_type is used to determine the name of the concrete\n    functions as reported by the unittest framework. If naming_type is\n    _FIRST_ARG, the testcases must be tuples, and the first element must\n    have a string representation that is a valid Python identifier.\n\n    Args:\n      test_method: The decorated test method.\n      testcases: (list of tuple/dict) A list of parameter\n                 tuples/dicts for individual test invocations.\n      naming_type: The test naming type, either _NAMED or _ARGUMENT_REPR.\n    '
        self._test_method = test_method
        self.testcases = testcases
        self._naming_type = naming_type

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise RuntimeError('You appear to be running a parameterized test case without having inherited from parameterized.ParameterizedTestCase. This is bad because none of your test cases are actually being run.')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        test_method = self._test_method
        naming_type = self._naming_type

        def MakeBoundParamTest(testcase_params):
            if False:
                return 10

            @functools.wraps(test_method)
            def BoundParamTest(self):
                if False:
                    i = 10
                    return i + 15
                if isinstance(testcase_params, collections.Mapping):
                    test_method(self, **testcase_params)
                elif _NonStringIterable(testcase_params):
                    test_method(self, *testcase_params)
                else:
                    test_method(self, testcase_params)
            if naming_type is _FIRST_ARG:
                BoundParamTest.__x_use_name__ = True
                BoundParamTest.__name__ += str(testcase_params[0])
                testcase_params = testcase_params[1:]
            elif naming_type is _ARGUMENT_REPR:
                BoundParamTest.__x_extra_id__ = '(%s)' % (_FormatParameterList(testcase_params),)
            else:
                raise RuntimeError('%s is not a valid naming type.' % (naming_type,))
            BoundParamTest.__doc__ = '%s(%s)' % (BoundParamTest.__name__, _FormatParameterList(testcase_params))
            if test_method.__doc__:
                BoundParamTest.__doc__ += '\n%s' % (test_method.__doc__,)
            return BoundParamTest
        return (MakeBoundParamTest(c) for c in self.testcases)

def _IsSingletonList(testcases):
    if False:
        return 10
    'True iff testcases contains only a single non-tuple element.'
    return len(testcases) == 1 and (not isinstance(testcases[0], tuple))

def _ModifyClass(class_object, testcases, naming_type):
    if False:
        i = 10
        return i + 15
    assert not getattr(class_object, '_id_suffix', None), 'Cannot add parameters to %s, which already has parameterized methods.' % (class_object,)
    class_object._id_suffix = id_suffix = {}
    for (name, obj) in class_object.__dict__.copy().items():
        if name.startswith(unittest.TestLoader.testMethodPrefix) and isinstance(obj, types.FunctionType):
            delattr(class_object, name)
            methods = {}
            _UpdateClassDictForParamTestCase(methods, id_suffix, name, _ParameterizedTestIter(obj, testcases, naming_type))
            for (name, meth) in methods.items():
                setattr(class_object, name, meth)

def _ParameterDecorator(naming_type, testcases):
    if False:
        i = 10
        return i + 15
    'Implementation of the parameterization decorators.\n\n  Args:\n    naming_type: The naming type.\n    testcases: Testcase parameters.\n\n  Returns:\n    A function for modifying the decorated object.\n  '

    def _Apply(obj):
        if False:
            return 10
        if isinstance(obj, type):
            _ModifyClass(obj, list(testcases) if not isinstance(testcases, collections.Sequence) else testcases, naming_type)
            return obj
        else:
            return _ParameterizedTestIter(obj, testcases, naming_type)
    if _IsSingletonList(testcases):
        assert _NonStringIterable(testcases[0]), 'Single parameter argument must be a non-string iterable'
        testcases = testcases[0]
    return _Apply

def Parameters(*testcases):
    if False:
        for i in range(10):
            print('nop')
    'A decorator for creating parameterized tests.\n\n  See the module docstring for a usage example.\n  Args:\n    *testcases: Parameters for the decorated method, either a single\n                iterable, or a list of tuples/dicts/objects (for tests\n                with only one argument).\n\n  Returns:\n     A test generator to be handled by TestGeneratorMetaclass.\n  '
    return _ParameterDecorator(_ARGUMENT_REPR, testcases)

def NamedParameters(*testcases):
    if False:
        i = 10
        return i + 15
    'A decorator for creating parameterized tests.\n\n  See the module docstring for a usage example. The first element of\n  each parameter tuple should be a string and will be appended to the\n  name of the test method.\n\n  Args:\n    *testcases: Parameters for the decorated method, either a single\n                iterable, or a list of tuples.\n\n  Returns:\n     A test generator to be handled by TestGeneratorMetaclass.\n  '
    return _ParameterDecorator(_FIRST_ARG, testcases)

class TestGeneratorMetaclass(type):
    """Metaclass for test cases with test generators.

  A test generator is an iterable in a testcase that produces callables. These
  callables must be single-argument methods. These methods are injected into
  the class namespace and the original iterable is removed. If the name of the
  iterable conforms to the test pattern, the injected methods will be picked
  up as tests by the unittest framework.

  In general, it is supposed to be used in conjuction with the
  Parameters decorator.
  """

    def __new__(mcs, class_name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        dct['_id_suffix'] = id_suffix = {}
        for (name, obj) in dct.items():
            if name.startswith(unittest.TestLoader.testMethodPrefix) and _NonStringIterable(obj):
                iterator = iter(obj)
                dct.pop(name)
                _UpdateClassDictForParamTestCase(dct, id_suffix, name, iterator)
        return type.__new__(mcs, class_name, bases, dct)

def _UpdateClassDictForParamTestCase(dct, id_suffix, name, iterator):
    if False:
        return 10
    'Adds individual test cases to a dictionary.\n\n  Args:\n    dct: The target dictionary.\n    id_suffix: The dictionary for mapping names to test IDs.\n    name: The original name of the test case.\n    iterator: The iterator generating the individual test cases.\n  '
    for (idx, func) in enumerate(iterator):
        assert callable(func), 'Test generators must yield callables, got %r' % (func,)
        if getattr(func, '__x_use_name__', False):
            new_name = func.__name__
        else:
            new_name = '%s%s%d' % (name, _SEPARATOR, idx)
        assert new_name not in dct, 'Name of parameterized test case "%s" not unique' % (new_name,)
        dct[new_name] = func
        id_suffix[new_name] = getattr(func, '__x_extra_id__', '')

class ParameterizedTestCase(unittest.TestCase):
    """Base class for test cases using the Parameters decorator."""
    __metaclass__ = TestGeneratorMetaclass

    def _OriginalName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._testMethodName.split(_SEPARATOR)[0]

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s (%s)' % (self._OriginalName(), _StrClass(self.__class__))

    def id(self):
        if False:
            return 10
        'Returns the descriptive ID of the test.\n\n    This is used internally by the unittesting framework to get a name\n    for the test to be used in reports.\n\n    Returns:\n      The test id.\n    '
        return '%s.%s%s' % (_StrClass(self.__class__), self._OriginalName(), self._id_suffix.get(self._testMethodName, ''))

def CoopParameterizedTestCase(other_base_class):
    if False:
        i = 10
        return i + 15
    'Returns a new base class with a cooperative metaclass base.\n\n  This enables the ParameterizedTestCase to be used in combination\n  with other base classes that have custom metaclasses, such as\n  mox.MoxTestBase.\n\n  Only works with metaclasses that do not override type.__new__.\n\n  Example:\n\n    import google3\n    import mox\n\n    from google3.testing.pybase import parameterized\n\n    class ExampleTest(parameterized.CoopParameterizedTestCase(mox.MoxTestBase)):\n      ...\n\n  Args:\n    other_base_class: (class) A test case base class.\n\n  Returns:\n    A new class object.\n  '
    metaclass = type('CoopMetaclass', (other_base_class.__metaclass__, TestGeneratorMetaclass), {})
    return metaclass('CoopParameterizedTestCase', (other_base_class, ParameterizedTestCase), {})