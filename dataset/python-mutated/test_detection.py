from hypothesis import given
from hypothesis.internal.detection import is_hypothesis_test
from hypothesis.stateful import RuleBasedStateMachine, rule
from hypothesis.strategies import integers

def test_functions_default_to_not_tests():
    if False:
        for i in range(10):
            print('nop')

    def foo():
        if False:
            print('Hello World!')
        pass
    assert not is_hypothesis_test(foo)

def test_methods_default_to_not_tests():
    if False:
        print('Hello World!')

    class Foo:

        def foo(self):
            if False:
                print('Hello World!')
            pass
    assert not is_hypothesis_test(Foo().foo)

def test_detection_of_functions():
    if False:
        return 10

    @given(integers())
    def test(i):
        if False:
            while True:
                i = 10
        pass
    assert is_hypothesis_test(test)

def test_detection_of_methods():
    if False:
        return 10

    class Foo:

        @given(integers())
        def test(self, i):
            if False:
                return 10
            pass
    assert is_hypothesis_test(Foo().test)

def test_detection_of_stateful_tests():
    if False:
        i = 10
        return i + 15

    class Stuff(RuleBasedStateMachine):

        @rule(x=integers())
        def a_rule(self, x):
            if False:
                for i in range(10):
                    print('nop')
            pass
    assert is_hypothesis_test(Stuff.TestCase().runTest)