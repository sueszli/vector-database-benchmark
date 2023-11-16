import pytest
from stanza.models.constituency.tree_stack import TreeStack
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_simple():
    if False:
        i = 10
        return i + 15
    stack = TreeStack(value=5, parent=None, length=1)
    stack = stack.push(3)
    stack = stack.push(1)
    expected_values = [1, 3, 5]
    for value in expected_values:
        assert stack.value == value
        stack = stack.pop()
    assert stack is None

def test_iter():
    if False:
        for i in range(10):
            print('nop')
    stack = TreeStack(value=5, parent=None, length=1)
    stack = stack.push(3)
    stack = stack.push(1)
    stack_list = list(stack)
    assert list(stack) == [1, 3, 5]

def test_str():
    if False:
        i = 10
        return i + 15
    stack = TreeStack(value=5, parent=None, length=1)
    stack = stack.push(3)
    stack = stack.push(1)
    assert str(stack) == 'TreeStack(1, 3, 5)'

def test_len():
    if False:
        i = 10
        return i + 15
    stack = TreeStack(value=5, parent=None, length=1)
    assert len(stack) == 1
    stack = stack.push(3)
    stack = stack.push(1)
    assert len(stack) == 3

def test_long_len():
    if False:
        print('Hello World!')
    '\n    Original stack had a bug where this took exponential time...\n    '
    stack = TreeStack(value=0, parent=None, length=1)
    for i in range(1, 40):
        stack = stack.push(i)
    assert len(stack) == 40