import pytest
from conda.common.toposort import pop_key, toposort

def test_pop_key():
    if False:
        while True:
            i = 10
    key = pop_key({'a': {'b', 'c'}, 'b': {'c'}})
    assert key == 'b'
    key = pop_key({'a': {'b'}, 'b': {'c', 'a'}})
    assert key == 'a'
    key = pop_key({'a': {'b'}, 'b': {'a'}})
    assert key == 'a'

def test_simple():
    if False:
        for i in range(10):
            print('nop')
    data = {'a': 'bc', 'b': 'c'}
    results = toposort(data, safe=True)
    assert results == ['c', 'b', 'a']
    results = toposort(data, safe=False)
    assert results == ['c', 'b', 'a']

def test_cycle():
    if False:
        for i in range(10):
            print('nop')
    data = {'a': 'b', 'b': 'a'}
    with pytest.raises(ValueError):
        toposort(data, False)
    results = toposort(data)
    assert set(results) == {'b', 'a'}

def test_cycle_best_effort():
    if False:
        print('Hello World!')
    data = {'a': 'bc', 'b': 'c', '1': '2', '2': '1'}
    results = toposort(data)
    assert results[:3] == ['c', 'b', 'a']
    assert set(results[3:]) == {'1', '2'}

def test_python_is_prioritized():
    if False:
        return 10
    "\n    This test checks a special invariant related to 'python' specifically.\n    Python is part of a cycle (pip <--> python), which can cause it to be\n    installed *after* packages that need python (possibly in\n    post-install.sh).\n\n    A special case in toposort() breaks the cycle, to ensure that python\n    isn't installed too late.  Here, we verify that it works.\n    "
    data = {'python': ['pip', 'openssl', 'readline', 'sqlite', 'tk', 'xz', 'zlib'], 'pip': ['python', 'setuptools', 'wheel'], 'setuptools': ['python'], 'wheel': ['python'], 'openssl': [], 'readline': [], 'sqlite': [], 'tk': [], 'xz': [], 'zlib': []}
    data.update({'psutil': ['python'], 'greenlet': ['python'], 'futures': ['python'], 'six': ['python']})
    results = toposort(data)
    assert results.index('python') < results.index('setuptools')
    assert results.index('python') < results.index('wheel')
    assert results.index('python') < results.index('pip')
    assert results.index('python') < results.index('psutil')
    assert results.index('python') < results.index('greenlet')
    assert results.index('python') < results.index('futures')
    assert results.index('python') < results.index('six')

def test_degenerate():
    if False:
        while True:
            i = 10
    'Edge cases.'
    assert toposort({}) == []
    assert toposort({}, safe=False) == []