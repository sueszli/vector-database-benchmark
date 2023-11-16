from unittest import TestCase
import pytest
from hypothesis import find, given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.stateful import RuleBasedStateMachine, rule

def test_cannot_use_without_a_runner():
    if False:
        while True:
            i = 10

    @given(st.runner())
    def f(x):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument):
        f()

def test_cannot_use_in_find_without_default():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        find(st.runner(), lambda x: True)

def test_is_default_in_find():
    if False:
        for i in range(10):
            print('nop')
    t = object()
    assert find(st.runner(default=t), lambda x: True) == t

@given(st.runner(default=1))
def test_is_default_without_self(runner):
    if False:
        i = 10
        return i + 15
    assert runner == 1

class TestStuff(TestCase):

    @given(st.runner())
    def test_runner_is_self(self, runner):
        if False:
            return 10
        assert runner is self

    @given(st.runner(default=3))
    def test_runner_is_self_even_with_default(self, runner):
        if False:
            print('Hello World!')
        assert runner is self

class RunnerStateMachine(RuleBasedStateMachine):

    @rule(runner=st.runner())
    def step(self, runner):
        if False:
            return 10
        assert runner is self
TestState = RunnerStateMachine.TestCase