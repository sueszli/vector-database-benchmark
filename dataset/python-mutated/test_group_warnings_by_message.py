import warnings
import pytest

def func(msg):
    if False:
        return 10
    warnings.warn(UserWarning(msg))

@pytest.mark.parametrize('i', range(5))
def test_foo(i):
    if False:
        return 10
    func('foo')

def test_foo_1():
    if False:
        return 10
    func('foo')

@pytest.mark.parametrize('i', range(5))
def test_bar(i):
    if False:
        print('Hello World!')
    func('bar')