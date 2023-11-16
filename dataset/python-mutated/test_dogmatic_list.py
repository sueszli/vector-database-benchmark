import pytest
from sacred.config.custom_containers import DogmaticDict, DogmaticList

def test_isinstance_of_list():
    if False:
        return 10
    assert isinstance(DogmaticList(), list)

def test_init():
    if False:
        for i in range(10):
            print('nop')
    l = DogmaticList()
    assert l == []
    l2 = DogmaticList([2, 3, 1])
    assert l2 == [2, 3, 1]

def test_append():
    if False:
        i = 10
        return i + 15
    l = DogmaticList([1, 2])
    l.append(3)
    l.append(4)
    assert l == [1, 2]

def test_extend():
    if False:
        i = 10
        return i + 15
    l = DogmaticList([1, 2])
    l.extend([3, 4])
    assert l == [1, 2]

def test_insert():
    if False:
        print('Hello World!')
    l = DogmaticList([1, 2])
    l.insert(1, 17)
    assert l == [1, 2]

def test_pop():
    if False:
        for i in range(10):
            print('nop')
    l = DogmaticList([1, 2, 3])
    with pytest.raises(TypeError):
        l.pop()
    assert l == [1, 2, 3]

def test_sort():
    if False:
        i = 10
        return i + 15
    l = DogmaticList([3, 1, 2])
    l.sort()
    assert l == [3, 1, 2]

def test_reverse():
    if False:
        print('Hello World!')
    l = DogmaticList([1, 2, 3])
    l.reverse()
    assert l == [1, 2, 3]

def test_setitem():
    if False:
        print('Hello World!')
    l = DogmaticList([1, 2, 3])
    l[1] = 23
    assert l == [1, 2, 3]

def test_setslice():
    if False:
        return 10
    l = DogmaticList([1, 2, 3])
    l[1:3] = [4, 5]
    assert l == [1, 2, 3]

def test_delitem():
    if False:
        print('Hello World!')
    l = DogmaticList([1, 2, 3])
    del l[1]
    assert l == [1, 2, 3]

def test_delslice():
    if False:
        return 10
    l = DogmaticList([1, 2, 3])
    del l[1:]
    assert l == [1, 2, 3]

def test_iadd():
    if False:
        print('Hello World!')
    l = DogmaticList([1, 2])
    l += [3, 4]
    assert l == [1, 2]

def test_imul():
    if False:
        for i in range(10):
            print('nop')
    l = DogmaticList([1, 2])
    l *= 4
    assert l == [1, 2]

def test_list_interface_getitem():
    if False:
        for i in range(10):
            print('nop')
    l = DogmaticList([0, 1, 2])
    assert l[0] == 0
    assert l[1] == 1
    assert l[2] == 2
    assert l[-1] == 2
    assert l[-2] == 1
    assert l[-3] == 0

def test_list_interface_len():
    if False:
        while True:
            i = 10
    l = DogmaticList()
    assert len(l) == 0
    l = DogmaticList([0, 1, 2])
    assert len(l) == 3

def test_list_interface_count():
    if False:
        while True:
            i = 10
    l = DogmaticList([1, 2, 4, 4, 5])
    assert l.count(1) == 1
    assert l.count(3) == 0
    assert l.count(4) == 2

def test_list_interface_index():
    if False:
        for i in range(10):
            print('nop')
    l = DogmaticList([1, 2, 4, 4, 5])
    assert l.index(1) == 0
    assert l.index(4) == 2
    assert l.index(5) == 4
    with pytest.raises(ValueError):
        l.index(3)

def test_empty_revelation():
    if False:
        while True:
            i = 10
    l = DogmaticList([1, 2, 3])
    assert l.revelation() == set()

def test_nested_dict_revelation():
    if False:
        i = 10
        return i + 15
    d1 = DogmaticDict({'a': 7, 'b': 12})
    d2 = DogmaticDict({'c': 7})
    l = DogmaticList([d1, 2, d2])
    l.revelation()
    assert 'a' in l[0]
    assert 'b' in l[0]
    assert 'c' in l[2]