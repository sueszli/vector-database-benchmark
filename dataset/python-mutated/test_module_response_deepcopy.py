from __future__ import annotations
from ansible.vars.clean import module_response_deepcopy

def test_module_response_deepcopy_basic():
    if False:
        while True:
            i = 10
    x = 42
    y = module_response_deepcopy(x)
    assert y == x

def test_module_response_deepcopy_atomic():
    if False:
        print('Hello World!')
    tests = [None, 42, 2 ** 100, 3.14, True, False, 1j, 'hello', u'helloሴ']
    for x in tests:
        assert module_response_deepcopy(x) is x

def test_module_response_deepcopy_list():
    if False:
        while True:
            i = 10
    x = [[1, 2], 3]
    y = module_response_deepcopy(x)
    assert y == x
    assert x is not y
    assert x[0] is not y[0]

def test_module_response_deepcopy_empty_tuple():
    if False:
        return 10
    x = ()
    y = module_response_deepcopy(x)
    assert x is y

def test_module_response_deepcopy_tuple_of_immutables():
    if False:
        return 10
    x = ((1, 2), 3)
    y = module_response_deepcopy(x)
    assert x is y

def test_module_response_deepcopy_dict():
    if False:
        while True:
            i = 10
    x = {'foo': [1, 2], 'bar': 3}
    y = module_response_deepcopy(x)
    assert y == x
    assert x is not y
    assert x['foo'] is not y['foo']