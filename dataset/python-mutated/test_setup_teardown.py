import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import integers, text

class HasSetup:

    def setup_example(self):
        if False:
            return 10
        self.setups = getattr(self, 'setups', 0)
        self.setups += 1

class HasTeardown:

    def teardown_example(self, ex):
        if False:
            print('Hello World!')
        self.teardowns = getattr(self, 'teardowns', 0)
        self.teardowns += 1

class SomeGivens:

    @given(integers())
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def give_me_an_int(self, x):
        if False:
            i = 10
            return i + 15
        pass

    @given(text())
    def give_me_a_string(self, x):
        if False:
            i = 10
            return i + 15
        pass

    @given(integers())
    @settings(max_examples=1000)
    def give_me_a_positive_int(self, x):
        if False:
            i = 10
            return i + 15
        assert x >= 0

    @given(integers().map(lambda x: x.nope))
    def fail_in_reify(self, x):
        if False:
            while True:
                i = 10
        pass

    @given(integers())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def assume_some_stuff(self, x):
        if False:
            return 10
        assume(x > 0)

    @given(integers().filter(lambda x: x > 0))
    def assume_in_reify(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass

class HasSetupAndTeardown(HasSetup, HasTeardown, SomeGivens):
    pass

def test_calls_setup_and_teardown_on_self_as_first_argument():
    if False:
        i = 10
        return i + 15
    x = HasSetupAndTeardown()
    x.give_me_an_int()
    x.give_me_a_string()
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_calls_setup_and_teardown_on_self_unbound():
    if False:
        while True:
            i = 10
    x = HasSetupAndTeardown()
    HasSetupAndTeardown.give_me_an_int(x)
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_calls_setup_and_teardown_on_failure():
    if False:
        return 10
    x = HasSetupAndTeardown()
    with pytest.raises(AssertionError):
        x.give_me_a_positive_int()
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_still_tears_down_on_error_in_generation():
    if False:
        return 10
    x = HasSetupAndTeardown()
    with pytest.raises(AttributeError):
        x.fail_in_reify()
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_still_tears_down_on_failed_assume():
    if False:
        i = 10
        return i + 15
    x = HasSetupAndTeardown()
    x.assume_some_stuff()
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_still_tears_down_on_failed_assume_in_reify():
    if False:
        i = 10
        return i + 15
    x = HasSetupAndTeardown()
    x.assume_in_reify()
    assert x.setups > 0
    assert x.teardowns == x.setups

def test_sets_up_without_teardown():
    if False:
        while True:
            i = 10

    class Foo(HasSetup, SomeGivens):
        pass
    x = Foo()
    x.give_me_an_int()
    assert x.setups > 0
    assert not hasattr(x, 'teardowns')

def test_tears_down_without_setup():
    if False:
        while True:
            i = 10

    class Foo(HasTeardown, SomeGivens):
        pass
    x = Foo()
    x.give_me_an_int()
    assert x.teardowns > 0
    assert not hasattr(x, 'setups')