import warnings
import pytest

def func(msg):
    if False:
        print('Hello World!')
    warnings.warn(UserWarning(msg))

@pytest.mark.parametrize('i', range(20))
def test_foo(i):
    if False:
        while True:
            i = 10
    func('foo')

def test_foo_1():
    if False:
        for i in range(10):
            print('nop')
    func('foo')

@pytest.mark.parametrize('i', range(20))
def test_bar(i):
    if False:
        print('Hello World!')
    func('bar')