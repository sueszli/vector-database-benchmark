import pprint
from typing import List
from typing import Tuple
import pytest

def pytest_generate_tests(metafunc):
    if False:
        i = 10
        return i + 15
    if 'arg1' in metafunc.fixturenames:
        metafunc.parametrize('arg1', ['arg1v1', 'arg1v2'], scope='module')
    if 'arg2' in metafunc.fixturenames:
        metafunc.parametrize('arg2', ['arg2v1', 'arg2v2'], scope='function')

@pytest.fixture(scope='session')
def checked_order():
    if False:
        return 10
    order: List[Tuple[str, str, str]] = []
    yield order
    pprint.pprint(order)
    assert order == [('issue_519.py', 'fix1', 'arg1v1'), ('test_one[arg1v1-arg2v1]', 'fix2', 'arg2v1'), ('test_one[arg1v1-arg2v2]', 'fix2', 'arg2v2'), ('test_two[arg1v1-arg2v1]', 'fix2', 'arg2v1'), ('test_two[arg1v1-arg2v2]', 'fix2', 'arg2v2'), ('issue_519.py', 'fix1', 'arg1v2'), ('test_one[arg1v2-arg2v1]', 'fix2', 'arg2v1'), ('test_one[arg1v2-arg2v2]', 'fix2', 'arg2v2'), ('test_two[arg1v2-arg2v1]', 'fix2', 'arg2v1'), ('test_two[arg1v2-arg2v2]', 'fix2', 'arg2v2')]

@pytest.fixture(scope='module')
def fix1(request, arg1, checked_order):
    if False:
        i = 10
        return i + 15
    checked_order.append((request.node.name, 'fix1', arg1))
    yield ('fix1-' + arg1)

@pytest.fixture(scope='function')
def fix2(request, fix1, arg2, checked_order):
    if False:
        return 10
    checked_order.append((request.node.name, 'fix2', arg2))
    yield ('fix2-' + arg2 + fix1)

def test_one(fix2):
    if False:
        print('Hello World!')
    pass

def test_two(fix2):
    if False:
        for i in range(10):
            print('nop')
    pass