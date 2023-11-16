from openai._utils import deepcopy_minimal

def assert_different_identities(obj1: object, obj2: object) -> None:
    if False:
        i = 10
        return i + 15
    assert obj1 == obj2
    assert id(obj1) != id(obj2)

def test_simple_dict() -> None:
    if False:
        while True:
            i = 10
    obj1 = {'foo': 'bar'}
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)

def test_nested_dict() -> None:
    if False:
        i = 10
        return i + 15
    obj1 = {'foo': {'bar': True}}
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)
    assert_different_identities(obj1['foo'], obj2['foo'])

def test_complex_nested_dict() -> None:
    if False:
        i = 10
        return i + 15
    obj1 = {'foo': {'bar': [{'hello': 'world'}]}}
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)
    assert_different_identities(obj1['foo'], obj2['foo'])
    assert_different_identities(obj1['foo']['bar'], obj2['foo']['bar'])
    assert_different_identities(obj1['foo']['bar'][0], obj2['foo']['bar'][0])

def test_simple_list() -> None:
    if False:
        while True:
            i = 10
    obj1 = ['a', 'b', 'c']
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)

def test_nested_list() -> None:
    if False:
        return 10
    obj1 = ['a', [1, 2, 3]]
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)
    assert_different_identities(obj1[1], obj2[1])

class MyObject:
    ...

def test_ignores_other_types() -> None:
    if False:
        i = 10
        return i + 15
    my_obj = MyObject()
    obj1 = {'foo': my_obj}
    obj2 = deepcopy_minimal(obj1)
    assert_different_identities(obj1, obj2)
    assert obj1['foo'] is my_obj
    obj3 = ('a', 'b')
    obj4 = deepcopy_minimal(obj3)
    assert obj3 is obj4