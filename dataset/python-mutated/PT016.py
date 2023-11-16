import pytest

def f():
    if False:
        for i in range(10):
            print('nop')
    pytest.fail('this is a failure')

def f():
    if False:
        for i in range(10):
            print('nop')
    pytest.fail(msg='this is a failure')

def f():
    if False:
        for i in range(10):
            print('nop')
    pytest.fail(reason='this is a failure')

def f():
    if False:
        i = 10
        return i + 15
    pytest.fail()
    pytest.fail('')
    pytest.fail(f'')
    pytest.fail(msg='')
    pytest.fail(msg=f'')
    pytest.fail(reason='')
    pytest.fail(reason=f'')