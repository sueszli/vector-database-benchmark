"""Unit tests for tf_decorator."""
import functools
import inspect
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

def test_tfdecorator(decorator_name, decorator_doc=None):
    if False:
        while True:
            i = 10

    def make_tf_decorator(target):
        if False:
            for i in range(10):
                print('nop')
        return tf_decorator.TFDecorator(decorator_name, target, decorator_doc)
    return make_tf_decorator

def test_decorator_increment_first_int_arg(target):
    if False:
        for i in range(10):
            print('nop')
    'This test decorator skips past `self` as args[0] in the bound case.'

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        new_args = []
        found = False
        for arg in args:
            if not found and isinstance(arg, int):
                new_args.append(arg + 1)
                found = True
            else:
                new_args.append(arg)
        return target(*new_args, **kwargs)
    return tf_decorator.make_decorator(target, wrapper)

def test_injectable_decorator_square(target):
    if False:
        i = 10
        return i + 15

    def wrapper(x):
        if False:
            print('Hello World!')
        return wrapper.__wrapped__(x) ** 2
    return tf_decorator.make_decorator(target, wrapper)

def test_injectable_decorator_increment(target):
    if False:
        return 10

    def wrapper(x):
        if False:
            i = 10
            return i + 15
        return wrapper.__wrapped__(x) + 1
    return tf_decorator.make_decorator(target, wrapper)

def test_function(x):
    if False:
        print('Hello World!')
    'Test Function Docstring.'
    return x + 1

@test_tfdecorator('decorator 1')
@test_decorator_increment_first_int_arg
@test_tfdecorator('decorator 3', 'decorator 3 documentation')
def test_decorated_function(x):
    if False:
        print('Hello World!')
    'Test Decorated Function Docstring.'
    return x * 2

@test_tfdecorator('decorator')
class TestDecoratedClass(object):
    """Test Decorated Class."""

    def __init__(self, two_attr=2):
        if False:
            i = 10
            return i + 15
        self.two_attr = two_attr

    @property
    def two_prop(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    def two_func(self):
        if False:
            return 10
        return 2

    @test_decorator_increment_first_int_arg
    def return_params(self, a, b, c):
        if False:
            print('Hello World!')
        'Return parameters.'
        return [a, b, c]

class TfDecoratorTest(test.TestCase):

    def testInitCapturesTarget(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(test_function, tf_decorator.TFDecorator('', test_function).decorated_target)

    def testInitCapturesDecoratorName(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('decorator name', tf_decorator.TFDecorator('decorator name', test_function).decorator_name)

    def testInitCapturesDecoratorDoc(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('decorator doc', tf_decorator.TFDecorator('', test_function, 'decorator doc').decorator_doc)

    def testInitCapturesNonNoneArgspec(self):
        if False:
            print('Hello World!')
        argspec = tf_inspect.FullArgSpec(args=['a', 'b', 'c'], varargs=None, varkw=None, defaults=(1, 'hello'), kwonlyargs=[], kwonlydefaults=None, annotations=None)
        self.assertIs(argspec, tf_decorator.TFDecorator('', test_function, '', argspec).decorator_argspec)

    def testInitSetsDecoratorNameToTargetName(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('test_function', tf_decorator.TFDecorator('', test_function).__name__)

    def testInitSetsDecoratorQualNameToTargetQualName(self):
        if False:
            while True:
                i = 10
        if hasattr(tf_decorator.TFDecorator('', test_function), '__qualname__'):
            self.assertEqual('test_function', tf_decorator.TFDecorator('', test_function).__qualname__)

    def testInitSetsDecoratorDocToTargetDoc(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('Test Function Docstring.', tf_decorator.TFDecorator('', test_function).__doc__)

    def testCallingATFDecoratorCallsTheTarget(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(124, tf_decorator.TFDecorator('', test_function)(123))

    def testCallingADecoratedFunctionCallsTheTarget(self):
        if False:
            while True:
                i = 10
        self.assertEqual((2 + 1) * 2, test_decorated_function(2))

    def testInitializingDecoratedClassWithInitParamsDoesntRaise(self):
        if False:
            return 10
        try:
            TestDecoratedClass(2)
        except TypeError:
            self.assertFail()

    def testReadingClassAttributeOnDecoratedClass(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(2, TestDecoratedClass().two_attr)

    def testCallingClassMethodOnDecoratedClass(self):
        if False:
            while True:
                i = 10
        self.assertEqual(2, TestDecoratedClass().two_func())

    def testReadingClassPropertyOnDecoratedClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(2, TestDecoratedClass().two_prop)

    def testNameOnBoundProperty(self):
        if False:
            return 10
        self.assertEqual('return_params', TestDecoratedClass().return_params.__name__)

    def testQualNameOnBoundProperty(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(TestDecoratedClass().return_params, '__qualname__'):
            self.assertEqual('TestDecoratedClass.return_params', TestDecoratedClass().return_params.__qualname__)

    def testDocstringOnBoundProperty(self):
        if False:
            return 10
        self.assertEqual('Return parameters.', TestDecoratedClass().return_params.__doc__)

    def testTarget__get__IsProxied(self):
        if False:
            while True:
                i = 10

        class Descr(object):

            def __get__(self, instance, owner):
                if False:
                    i = 10
                    return i + 15
                return self

        class Foo(object):
            foo = tf_decorator.TFDecorator('Descr', Descr())
        self.assertIsInstance(Foo.foo, Descr)

def test_wrapper(*args, **kwargs):
    if False:
        return 10
    return test_function(*args, **kwargs)

class TfMakeDecoratorTest(test.TestCase):

    def testAttachesATFDecoratorAttr(self):
        if False:
            i = 10
            return i + 15
        decorated = tf_decorator.make_decorator(test_function, test_wrapper)
        decorator = getattr(decorated, '_tf_decorator')
        self.assertIsInstance(decorator, tf_decorator.TFDecorator)

    def testAttachesWrappedAttr(self):
        if False:
            return 10
        decorated = tf_decorator.make_decorator(test_function, test_wrapper)
        wrapped_attr = getattr(decorated, '__wrapped__')
        self.assertIs(test_function, wrapped_attr)

    def testSetsTFDecoratorNameToDecoratorNameArg(self):
        if False:
            return 10
        decorated = tf_decorator.make_decorator(test_function, test_wrapper, 'test decorator name')
        decorator = getattr(decorated, '_tf_decorator')
        self.assertEqual('test decorator name', decorator.decorator_name)

    def testSetsTFDecoratorDocToDecoratorDocArg(self):
        if False:
            i = 10
            return i + 15
        decorated = tf_decorator.make_decorator(test_function, test_wrapper, decorator_doc='test decorator doc')
        decorator = getattr(decorated, '_tf_decorator')
        self.assertEqual('test decorator doc', decorator.decorator_doc)

    def testUpdatesDictWithMissingEntries(self):
        if False:
            for i in range(10):
                print('nop')
        test_function.foobar = True
        decorated = tf_decorator.make_decorator(test_function, test_wrapper)
        self.assertTrue(decorated.foobar)
        del test_function.foobar

    def testUpdatesDict_doesNotOverridePresentEntries(self):
        if False:
            return 10
        test_function.foobar = True
        test_wrapper.foobar = False
        decorated = tf_decorator.make_decorator(test_function, test_wrapper)
        self.assertFalse(decorated.foobar)
        del test_function.foobar
        del test_wrapper.foobar

    def testSetsTFDecoratorArgSpec(self):
        if False:
            return 10
        argspec = tf_inspect.FullArgSpec(args=['a', 'b', 'c'], varargs='args', kwonlyargs={}, defaults=(1, 'hello'), kwonlydefaults=None, varkw='kwargs', annotations=None)
        decorated = tf_decorator.make_decorator(test_function, test_wrapper, '', '', argspec)
        decorator = getattr(decorated, '_tf_decorator')
        self.assertEqual(argspec, decorator.decorator_argspec)
        self.assertEqual(inspect.signature(decorated), inspect.Signature([inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD), inspect.Parameter('b', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1), inspect.Parameter('c', inspect.Parameter.POSITIONAL_OR_KEYWORD, default='hello'), inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL), inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)]))

    def testSetsDecoratorNameToFunctionThatCallsMakeDecoratorIfAbsent(self):
        if False:
            return 10

        def test_decorator_name(wrapper):
            if False:
                i = 10
                return i + 15
            return tf_decorator.make_decorator(test_function, wrapper)
        decorated = test_decorator_name(test_wrapper)
        decorator = getattr(decorated, '_tf_decorator')
        self.assertEqual('test_decorator_name', decorator.decorator_name)

    def testCompatibleWithNamelessCallables(self):
        if False:
            print('Hello World!')

        class Callable(object):

            def __call__(self):
                if False:
                    while True:
                        i = 10
                pass
        callable_object = Callable()
        _ = tf_decorator.make_decorator(callable_object, test_wrapper)
        partial = functools.partial(test_function, x=1)
        _ = tf_decorator.make_decorator(partial, test_wrapper)

class TfDecoratorRewrapTest(test.TestCase):

    def testRewrapMutatesAffectedFunction(self):
        if False:
            for i in range(10):
                print('nop')

        @test_injectable_decorator_square
        @test_injectable_decorator_increment
        def test_rewrappable_decorated(x):
            if False:
                while True:
                    i = 10
            return x * 2

        def new_target(x):
            if False:
                while True:
                    i = 10
            return x * 3
        self.assertEqual((1 * 2 + 1) ** 2, test_rewrappable_decorated(1))
        (prev_target, _) = tf_decorator.unwrap(test_rewrappable_decorated)
        tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
        self.assertEqual((1 * 3 + 1) ** 2, test_rewrappable_decorated(1))

    def testRewrapOfDecoratorFunction(self):
        if False:
            for i in range(10):
                print('nop')

        @test_injectable_decorator_square
        @test_injectable_decorator_increment
        def test_rewrappable_decorated(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * 2

        def new_target(x):
            if False:
                while True:
                    i = 10
            return x * 3
        prev_target = test_rewrappable_decorated._tf_decorator._decorated_target
        tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
        self.assertEqual((1 * 3) ** 2, test_rewrappable_decorated(1))

class TfDecoratorUnwrapTest(test.TestCase):

    def testUnwrapReturnsEmptyArrayForUndecoratedFunction(self):
        if False:
            for i in range(10):
                print('nop')
        (decorators, _) = tf_decorator.unwrap(test_function)
        self.assertEqual(0, len(decorators))

    def testUnwrapReturnsUndecoratedFunctionAsTarget(self):
        if False:
            i = 10
            return i + 15
        (_, target) = tf_decorator.unwrap(test_function)
        self.assertIs(test_function, target)

    def testUnwrapReturnsFinalFunctionAsTarget(self):
        if False:
            while True:
                i = 10
        self.assertEqual((4 + 1) * 2, test_decorated_function(4))
        (_, target) = tf_decorator.unwrap(test_decorated_function)
        self.assertTrue(tf_inspect.isfunction(target))
        self.assertEqual(4 * 2, target(4))

    def testUnwrapReturnsListOfUniqueTFDecorators(self):
        if False:
            for i in range(10):
                print('nop')
        (decorators, _) = tf_decorator.unwrap(test_decorated_function)
        self.assertEqual(3, len(decorators))
        self.assertTrue(isinstance(decorators[0], tf_decorator.TFDecorator))
        self.assertTrue(isinstance(decorators[1], tf_decorator.TFDecorator))
        self.assertTrue(isinstance(decorators[2], tf_decorator.TFDecorator))
        self.assertIsNot(decorators[0], decorators[1])
        self.assertIsNot(decorators[1], decorators[2])
        self.assertIsNot(decorators[2], decorators[0])

    def testUnwrapReturnsDecoratorListFromOutermostToInnermost(self):
        if False:
            print('Hello World!')
        (decorators, _) = tf_decorator.unwrap(test_decorated_function)
        self.assertEqual('decorator 1', decorators[0].decorator_name)
        self.assertEqual('test_decorator_increment_first_int_arg', decorators[1].decorator_name)
        self.assertEqual('decorator 3', decorators[2].decorator_name)
        self.assertEqual('decorator 3 documentation', decorators[2].decorator_doc)

    def testUnwrapBoundMethods(self):
        if False:
            for i in range(10):
                print('nop')
        test_decorated_class = TestDecoratedClass()
        self.assertEqual([2, 2, 3], test_decorated_class.return_params(1, 2, 3))
        (decorators, target) = tf_decorator.unwrap(test_decorated_class.return_params)
        self.assertEqual('test_decorator_increment_first_int_arg', decorators[0].decorator_name)
        self.assertEqual([1, 2, 3], target(test_decorated_class, 1, 2, 3))
if __name__ == '__main__':
    test.main()