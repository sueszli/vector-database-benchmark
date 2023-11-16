import sys
from random import randint, choice
from perspective.table import Table

class CustomObjectStore(object):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self._value = value

    def _psp_dtype_(self):
        if False:
            print('Hello World!')
        return 'object'

    def __int__(self):
        if False:
            while True:
                i = 10
        return int(self._value)

    def __repr__(self):
        if False:
            return 10
        return 'test' if self._value == 1 else 'test{}'.format(self._value)

def run():
    if False:
        i = 10
        return i + 15
    t = CustomObjectStore(1)
    t2 = CustomObjectStore(2)
    assert sys.getrefcount(t) == 2
    assert sys.getrefcount(t2) == 2
    data = {'a': [0], 'b': [t]}
    assert sys.getrefcount(t) == 3
    tbl = Table(data, index='a')
    assert sys.getrefcount(t) == 4
    assert tbl.schema() == {'a': int, 'b': object}
    assert tbl.size() == 1
    assert tbl.view().to_dict() == {'a': [0], 'b': [t]}
    print(sys.getrefcount(t), 'should be', 4)
    print('t:', id(t))
    assert sys.getrefcount(t) == 4
    print(sys.getrefcount(t2), 'should be', 2)
    print('t2:', id(t2))
    assert sys.getrefcount(t2) == 2
    tbl.update([{'a': i, 'b': None} for i in range(1, 6)])
    assert tbl.size() == 6
    print(sys.getrefcount(t), 'should be', 4)
    assert sys.getrefcount(t) == 4
    print()
    tbl.update([data])
    print(sys.getrefcount(t), 'should be', 4)
    assert sys.getrefcount(t) == 4
    print()
    tbl.update([data])
    print(sys.getrefcount(t), 'should be', 4)
    assert sys.getrefcount(t) == 4
    print()
    tbl.update([data])
    print(sys.getrefcount(t), 'should be', 4)
    assert sys.getrefcount(t) == 4
    print()
    tbl.update([{'a': 1, 'b': t}])
    print(sys.getrefcount(t), 'should be', 5)
    assert sys.getrefcount(t) == 5
    print()
    tbl.update([{'a': 1, 'b': t}])
    print(sys.getrefcount(t), 'should be', 5)
    assert sys.getrefcount(t) == 5
    print()
    tbl.update([{'a': 3, 'b': t}])
    print(sys.getrefcount(t), 'should be', 6)
    assert sys.getrefcount(t) == 6
    print()
    tbl.update([{'a': 3, 'b': t}])
    print(sys.getrefcount(t), 'should be', 6)
    assert sys.getrefcount(t) == 6
    print()
    tbl.update([{'a': 5, 'b': t}])
    print(sys.getrefcount(t), 'should be', 7)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 0, 'b': t}])
    print(sys.getrefcount(t), 'should be', 7)
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 1, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 1, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 1, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 1, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 2, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 4)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, True, True, False, True]
    print()
    tbl.update([{'a': 2, 'b': None}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 2, 'b': t2}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 4)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, True, True, False, True]
    print()
    tbl.update([{'a': 2, 'b': None}])
    print(sys.getrefcount(t), 'should be', 6)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, True, False, True]
    print()
    tbl.update([{'a': 3, 'b': None}])
    print(sys.getrefcount(t), 'should be', 5)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, False, False, True]
    print()
    tbl.update([{'a': 3, 'b': None}])
    print(sys.getrefcount(t), 'should be', 5)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, False, False, True]
    print()
    tbl.update([{'a': 5, 'b': None}])
    print(sys.getrefcount(t), 'should be', 4)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, False, False, False]
    print()
    tbl.update([{'a': 5, 'b': None}])
    print(sys.getrefcount(t), 'should be', 4)
    print(sys.getrefcount(t2), 'should be', 3)
    print(tbl.view().to_dict()['b'])
    assert list((_ is not None for _ in tbl.view().to_dict()['b'])) == [True, True, False, False, False, False]
    print()
    tbl.clear()
    assert tbl.size() == 0
    assert tbl.view().to_dict() == {}
    print(sys.getrefcount(t), 'should be', 3)
    assert sys.getrefcount(t) == 3

def run2():
    if False:
        print('Hello World!')
    t = CustomObjectStore(1)
    t_ref_count = 2
    assert sys.getrefcount(t) == t_ref_count
    indexes = set([0])
    tbl = Table({'a': [0], 'b': [t]}, index='a')
    assert sys.getrefcount(t) == 3
    t_ref_count += 1
    assert tbl.schema() == {'a': int, 'b': object}
    assert tbl.size() == 1
    assert tbl.view().to_dict() == {'a': [0], 'b': [t]}
    tbl.remove([1])
    tbl.remove([1])
    tbl.remove([1])
    for _ in range(10):
        pick = randint(1, 2) if indexes else 1
        if pick == 1:
            ind = randint(1, 10)
            while ind in indexes:
                ind = randint(1, 100)
            print('adding', ind, 'refcount', t_ref_count, 'should be', sys.getrefcount(t))
            tbl.update({'a': [ind], 'b': [t]})
            t_ref_count += 1
            indexes.add(ind)
            assert sys.getrefcount(t) == t_ref_count
        else:
            ind = choice(list(indexes))
            indexes.remove(ind)
            tbl.remove([ind])
            t_ref_count -= 1
            print('removing', ind, 'refcount', t_ref_count, 'should be', sys.getrefcount(t))
            assert sys.getrefcount(t) == t_ref_count
        print(t_ref_count)
        print(tbl.view().to_dict())
    assert sys.getrefcount(t) == t_ref_count
    print()
    tbl.clear()
    assert tbl.size() == 0
    assert tbl.view().to_dict() == {}
    print(sys.getrefcount(t), 'should be', 2)
    assert sys.getrefcount(t) == 2