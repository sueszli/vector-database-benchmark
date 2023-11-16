from __future__ import annotations
import pickle
import types
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from operator import add, setitem
from random import random
from typing import NamedTuple
import cloudpickle
import pytest
from tlz import merge
import dask
import dask.bag as db
from dask import compute
from dask.delayed import Delayed, delayed, to_task_dask
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import inc

class Tuple:
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __init__(self, dsk, keys):
        if False:
            for i in range(10):
                print('nop')
        self._dask = dsk
        self._keys = keys

    def __dask_tokenize__(self):
        if False:
            return 10
        return self._keys

    def __dask_graph__(self):
        if False:
            i = 10
            return i + 15
        return self._dask

    def __dask_keys__(self):
        if False:
            print('Hello World!')
        return self._keys

    def __dask_postcompute__(self):
        if False:
            return 10
        return (tuple, ())

@pytest.mark.filterwarnings('ignore:The dask.delayed:UserWarning')
def test_to_task_dask():
    if False:
        return 10
    a = delayed(1, name='a')
    b = delayed(2, name='b')
    (task, dask) = to_task_dask([a, b, 3])
    assert task == ['a', 'b', 3]
    (task, dask) = to_task_dask((a, b, 3))
    assert task == (tuple, ['a', 'b', 3])
    assert dict(dask) == merge(a.dask, b.dask)
    (task, dask) = to_task_dask({a: 1, b: 2})
    assert task == (dict, [['b', 2], ['a', 1]]) or task == (dict, [['a', 1], ['b', 2]])
    assert dict(dask) == merge(a.dask, b.dask)
    f = namedtuple('f', ['a', 'b', 'c'])
    x = f(a, b, 3)
    (task, dask) = to_task_dask(x)
    assert task == (f, 'a', 'b', 3)
    assert dict(dask) == merge(a.dask, b.dask)
    (task, dask) = to_task_dask(slice(a, b, 3))
    assert task == (slice, 'a', 'b', 3)
    assert dict(dask) == merge(a.dask, b.dask)

    class MyClass(dict):
        pass
    (task, dask) = to_task_dask(MyClass())
    assert type(task) is MyClass
    assert dict(dask) == {}
    x = Tuple({'a': 1, 'b': 2, 'c': (add, 'a', 'b')}, ['a', 'b', 'c'])
    (task, dask) = to_task_dask(x)
    assert task in dask
    f = dask.pop(task)
    assert f == (tuple, ['a', 'b', 'c'])
    assert dask == x._dask

def test_delayed():
    if False:
        while True:
            i = 10
    add2 = delayed(add)
    assert add2(1, 2).compute() == 3
    assert (add2(1, 2) + 3).compute() == 6
    assert add2(add2(1, 2), 3).compute() == 6
    a = delayed(1)
    assert a.compute() == 1
    assert 1 in a.dask.values()
    b = add2(add2(a, 2), 3)
    assert a.key in b.dask

def test_delayed_with_namedtuple():
    if False:
        return 10

    class ANamedTuple(NamedTuple):
        a: int
    literal = dask.delayed(3)
    with_class = dask.delayed({'a': ANamedTuple(a=literal)})

    def return_nested(obj):
        if False:
            while True:
                i = 10
        return obj['a'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3

@dataclass
class ANonFrozenDataClass:
    a: int

@dataclass(frozen=True)
class AFrozenDataClass:
    a: int

@pytest.mark.parametrize('cls', (ANonFrozenDataClass, AFrozenDataClass))
def test_delayed_with_dataclass(cls):
    if False:
        i = 10
        return i + 15
    literal = delayed(3)
    with_class = delayed({'data': cls(a=literal)})

    def return_nested(obj):
        if False:
            print('Hello World!')
        return obj['data'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3

def test_delayed_with_dataclass_with_custom_init():
    if False:
        return 10

    @dataclass()
    class ADataClass:
        a: int

        def __init__(self, b: int):
            if False:
                for i in range(10):
                    print('nop')
            self.a = b
    literal = dask.delayed(3)
    with pytest.raises(TypeError) as e:
        dask.delayed({'data': ADataClass(b=literal)})
    e.match('ADataClass')
    e.match('custom __init__ is not supported')

def test_delayed_with_dataclass_with_eager_custom_init():
    if False:
        while True:
            i = 10

    @dataclass()
    class ADataClass:
        a: int

        def __init__(self, b: int):
            if False:
                for i in range(10):
                    print('nop')
            self.a = b
    with_class = delayed({'data': ADataClass(b=3)})

    def return_nested(obj):
        if False:
            while True:
                i = 10
        return obj['data'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3

def test_delayed_with_eager_dataclass_with_set_init_false_field():
    if False:
        print('Hello World!')

    @dataclass
    class ADataClass:
        a: int
        b: int = field(init=False)

    def prep_dataclass(a):
        if False:
            print('Hello World!')
        data = ADataClass(a=a)
        data.b = 4
        return data
    with_class = delayed({'data': prep_dataclass(3)})

    def return_nested(obj):
        if False:
            for i in range(10):
                print('nop')
        return obj['data'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3

def test_delayed_with_dataclass_with_set_init_false_field():
    if False:
        i = 10
        return i + 15

    @dataclass
    class ADataClass:
        a: int
        b: int = field(init=False)
    literal = dask.delayed(3)

    def prep_dataclass(a):
        if False:
            return 10
        data = ADataClass(a=a)
        data.b = 4
        return data
    with pytest.raises(ValueError) as e:
        dask.delayed(prep_dataclass(literal))
    e.match('ADataClass')
    e.match('`init=False` are not supported')

def test_delayed_with_dataclass_with_unset_init_false_field():
    if False:
        while True:
            i = 10

    @dataclass
    class ADataClass:
        a: int
        b: int = field(init=False)
    literal = dask.delayed(3)
    with_class = delayed({'data': ADataClass(a=literal)})

    def return_nested(obj):
        if False:
            while True:
                i = 10
        return obj['data'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3

def test_operators():
    if False:
        i = 10
        return i + 15
    a = delayed([1, 2, 3])
    assert a[0].compute() == 1
    assert (a + a).compute() == [1, 2, 3, 1, 2, 3]
    b = delayed(2)
    assert a[:b].compute() == [1, 2]
    a = delayed(10)
    assert (a + 1).compute() == 11
    assert (1 + a).compute() == 11
    assert (a >> 1).compute() == 5
    assert (a > 2).compute()
    assert (a ** 2).compute() == 100

    class dummy:

        def __matmul__(self, other):
            if False:
                return 10
            return 4
    c = delayed(dummy())
    d = delayed(dummy())
    assert (c @ d).compute() == 4

def test_methods():
    if False:
        return 10
    a = delayed('a b c d e')
    assert a.split(' ').compute() == ['a', 'b', 'c', 'd', 'e']
    assert a.upper().replace('B', 'A').split().count('A').compute() == 2
    assert a.split(' ', pure=True).key == a.split(' ', pure=True).key
    o = a.split(' ', dask_key_name='test')
    assert o.key == 'test'

def test_attributes():
    if False:
        return 10
    a = delayed(2 + 1j)
    assert a.real._key == a.real._key
    assert a.real.compute() == 2
    assert a.imag.compute() == 1
    assert (a.real + a.imag).compute() == 3

def test_method_getattr_call_same_task():
    if False:
        i = 10
        return i + 15
    a = delayed([1, 2, 3])
    o = a.index(1)
    assert getattr not in {v[0] for v in o.__dask_graph__().values()}

def test_np_dtype_of_delayed():
    if False:
        for i in range(10):
            print('nop')
    np = pytest.importorskip('numpy')
    x = delayed(1)
    with pytest.raises(TypeError):
        np.dtype(x)
    assert delayed(np.array([1], dtype='f8')).dtype.compute() == np.dtype('f8')

def test_delayed_visualise_warn():
    if False:
        for i in range(10):
            print('nop')

    def inc(x):
        if False:
            while True:
                i = 10
        return x + 1
    z = dask.delayed(inc)(1)
    z.compute()
    with pytest.warns(UserWarning, match='dask.delayed objects have no `visualise` method'):
        z.visualise(file_name='desk_graph.svg')
    with pytest.warns(UserWarning, match='dask.delayed objects have no `visualise` method'):
        z.visualise()

def test_delayed_errors():
    if False:
        i = 10
        return i + 15
    a = delayed([1, 2, 3])
    pytest.raises(TypeError, lambda : setattr(a, 'foo', 1))
    pytest.raises(TypeError, lambda : setitem(a, 1, 0))
    pytest.raises(TypeError, lambda : 1 in a)
    pytest.raises(TypeError, lambda : list(a))
    pytest.raises(AttributeError, lambda : a._hidden())
    pytest.raises(TypeError, lambda : bool(a))

def test_common_subexpressions():
    if False:
        while True:
            i = 10
    a = delayed([1, 2, 3])
    res = a[0] + a[0]
    assert a[0].key in res.dask
    assert a.key in res.dask
    assert len(res.dask) == 3

def test_delayed_optimize():
    if False:
        print('Hello World!')
    x = Delayed('b', {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')})
    (x2,) = dask.optimize(x)
    assert sorted(x2.dask.keys()) == ['a', 'b']
    assert x2._layer != x2._key

def test_lists():
    if False:
        return 10
    a = delayed(1)
    b = delayed(2)
    c = delayed(sum)([a, b])
    assert c.compute() == 3

def test_literates():
    if False:
        return 10
    a = delayed(1)
    b = a + 1
    lit = (a, b, 3)
    assert delayed(lit).compute() == (1, 2, 3)
    lit = [a, b, 3]
    assert delayed(lit).compute() == [1, 2, 3]
    lit = {a, b, 3}
    assert delayed(lit).compute() == {1, 2, 3}
    lit = {a: 'a', b: 'b', 3: 'c'}
    assert delayed(lit).compute() == {1: 'a', 2: 'b', 3: 'c'}
    assert delayed(lit)[a].compute() == 'a'
    lit = {'a': a, 'b': b, 'c': 3}
    assert delayed(lit).compute() == {'a': 1, 'b': 2, 'c': 3}
    assert delayed(lit)['a'].compute() == 1

def test_literates_keys():
    if False:
        while True:
            i = 10
    a = delayed(1)
    b = a + 1
    lit = (a, b, 3)
    assert delayed(lit).key != delayed(lit).key
    assert delayed(lit, pure=True).key == delayed(lit, pure=True).key

def test_lists_are_concrete():
    if False:
        for i in range(10):
            print('nop')
    a = delayed(1)
    b = delayed(2)
    c = delayed(max)([[a, 10], [b, 20]], key=lambda x: x[0])[1]
    assert c.compute() == 20

@pytest.mark.parametrize('typ', [list, tuple, set])
def test_iterators(typ):
    if False:
        return 10
    a = delayed(1)
    b = delayed(2)
    c = delayed(sum)(iter(typ([a, b])))
    x = c.compute()
    assert x == 3

    def f(seq):
        if False:
            print('Hello World!')
        return sum(seq)
    c = delayed(f)(iter(typ([a, b])))
    assert c.compute() == 3

def test_traverse_false():
    if False:
        return 10

    def fail(*args):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError("shouldn't have computed")
    a = delayed(fail)()
    x = [a, 1, 2, 3]
    res = delayed(x, traverse=False).compute()
    assert len(res) == 4
    assert res[0] is a
    assert res[1:] == x[1:]
    x = (fail, a, (fail, a))
    res = delayed(x, traverse=False).compute()
    assert isinstance(res, tuple)
    assert res[0] == fail
    assert res[1] is a
    x = [1, (fail, a), a]
    res = delayed(x, traverse=False).compute()
    assert isinstance(res, list)
    assert res[0] == 1
    assert res[1][0] == fail and res[1][1] is a
    assert res[2] is a
    b = delayed(1)
    x = delayed(b, traverse=False)
    assert x.compute() == 1

def test_pure():
    if False:
        for i in range(10):
            print('nop')
    v1 = delayed(add, pure=True)(1, 2)
    v2 = delayed(add, pure=True)(1, 2)
    assert v1.key == v2.key
    myrand = delayed(random)
    assert myrand().key != myrand().key

def test_pure_global_setting():
    if False:
        for i in range(10):
            print('nop')
    func = delayed(add)
    with dask.config.set(delayed_pure=True):
        assert func(1, 2).key == func(1, 2).key
    with dask.config.set(delayed_pure=False):
        assert func(1, 2).key != func(1, 2).key
    func = delayed(add, pure=True)
    with dask.config.set(delayed_pure=False):
        assert func(1, 2).key == func(1, 2).key
    assert delayed(1).key != delayed(1).key
    with dask.config.set(delayed_pure=True):
        assert delayed(1).key == delayed(1).key
    with dask.config.set(delayed_pure=False):
        assert delayed(1, pure=True).key == delayed(1, pure=True).key
    data = delayed([1, 2, 3])
    assert data.index(1).key != data.index(1).key
    with dask.config.set(delayed_pure=True):
        assert data.index(1).key == data.index(1).key
        assert data.index(1, pure=False).key != data.index(1, pure=False).key
    with dask.config.set(delayed_pure=False):
        assert data.index(1, pure=True).key == data.index(1, pure=True).key
    with dask.config.set(delayed_pure=False):
        assert data.index.key == data.index.key
        element = data[0]
        assert (element + element).key == (element + element).key

def test_nout():
    if False:
        for i in range(10):
            print('nop')
    func = delayed(lambda x: (x, -x), nout=2, pure=True)
    x = func(1)
    assert len(x) == 2
    (a, b) = x
    assert compute(a, b) == (1, -1)
    assert a._length is None
    assert b._length is None
    pytest.raises(TypeError, lambda : len(a))
    pytest.raises(TypeError, lambda : list(a))
    pytest.raises(ValueError, lambda : delayed(add, nout=-1))
    pytest.raises(ValueError, lambda : delayed(add, nout=True))
    func = delayed(add, nout=None)
    a = func(1)
    assert a._length is None
    pytest.raises(TypeError, lambda : list(a))
    pytest.raises(TypeError, lambda : len(a))
    func = delayed(lambda x: (x,), nout=1, pure=True)
    x = func(1)
    assert len(x) == 1
    (a,) = x
    assert a.compute() == 1
    assert a._length is None
    pytest.raises(TypeError, lambda : len(a))
    func = delayed(lambda x: tuple(), nout=0, pure=True)
    x = func(1)
    assert len(x) == 0
    assert x.compute() == tuple()

@pytest.mark.parametrize('x', [[1, 2], (1, 2), (add, 1, 2), [], ()])
def test_nout_with_tasks(x):
    if False:
        i = 10
        return i + 15
    length = len(x)
    d = delayed(x, nout=length)
    assert len(d) == len(list(d)) == length
    assert d.compute() == x

def test_kwargs():
    if False:
        return 10

    def mysum(a, b, c=(), **kwargs):
        if False:
            return 10
        return a + b + sum(c) + sum(kwargs.values())
    dmysum = delayed(mysum)
    ten = dmysum(1, 2, c=[delayed(3), 0], four=dmysum(2, 2))
    assert ten.compute() == 10
    dmysum = delayed(mysum, pure=True)
    c = [delayed(3), 0]
    ten = dmysum(1, 2, c=c, four=dmysum(2, 2))
    assert ten.compute() == 10
    assert dmysum(1, 2, c=c, four=dmysum(2, 2)).key == ten.key
    assert dmysum(1, 2, c=c, four=dmysum(2, 3)).key != ten.key
    assert dmysum(1, 2, c=c, four=4).key != ten.key
    assert dmysum(1, 2, c=c, four=4).key != dmysum(2, 2, c=c, four=4).key

def test_custom_delayed():
    if False:
        return 10
    x = Tuple({'a': 1, 'b': 2, 'c': (add, 'a', 'b')}, ['a', 'b', 'c'])
    x2 = delayed(add, pure=True)(x, (4, 5, 6))
    n = delayed(len, pure=True)(x)
    assert delayed(len, pure=True)(x).key == n.key
    assert x2.compute() == (1, 2, 3, 4, 5, 6)
    assert compute(n, x2, x) == (3, (1, 2, 3, 4, 5, 6), (1, 2, 3))

@pytest.mark.filterwarnings('ignore:The dask.delayed:UserWarning')
def test_array_delayed():
    if False:
        return 10
    np = pytest.importorskip('numpy')
    da = pytest.importorskip('dask.array')
    arr = np.arange(100).reshape((10, 10))
    darr = da.from_array(arr, chunks=(5, 5))
    val = delayed(sum)([arr, darr, 1])
    assert isinstance(val, Delayed)
    assert np.allclose(val.compute(), arr + arr + 1)
    assert val.sum().compute() == (arr + arr + 1).sum()
    assert val[0, 0].compute() == (arr + arr + 1)[0, 0]
    (task, dsk) = to_task_dask(darr)
    assert not darr.dask.keys() - dsk.keys()
    diff = dsk.keys() - darr.dask.keys()
    assert len(diff) == 1
    delayed_arr = delayed(darr)
    assert (delayed_arr.compute() == arr).all()

def test_array_bag_delayed():
    if False:
        for i in range(10):
            print('nop')
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')
    arr1 = np.arange(100).reshape((10, 10))
    arr2 = arr1.dot(arr1.T)
    darr1 = da.from_array(arr1, chunks=(5, 5))
    darr2 = da.from_array(arr2, chunks=(5, 5))
    b = db.from_sequence([1, 2, 3])
    seq = [arr1, arr2, darr1, darr2, b]
    out = delayed(sum)([i.sum() for i in seq])
    assert out.compute() == 2 * arr1.sum() + 2 * arr2.sum() + sum([1, 2, 3])

def test_delayed_picklable():
    if False:
        return 10
    x = delayed(divmod, nout=2, pure=True)(1, 2)
    y = pickle.loads(pickle.dumps(x))
    assert x.dask == y.dask
    assert x._key == y._key
    assert x._length == y._length
    x = delayed(1j + 2)
    y = pickle.loads(pickle.dumps(x))
    assert x.dask == y.dask
    assert x._key == y._key
    assert x._nout == y._nout
    assert x._pure == y._pure
    x = x.real
    y = pickle.loads(pickle.dumps(x))
    assert x._obj._key == y._obj._key
    assert x._obj.dask == y._obj.dask
    assert x._attr == y._attr
    assert x._key == y._key

def test_delayed_compute_forward_kwargs():
    if False:
        print('Hello World!')
    x = delayed(1) + 2
    x.compute(bogus_keyword=10)

def test_delayed_method_descriptor():
    if False:
        return 10
    delayed(bytes.decode)(b'')

def test_delayed_callable():
    if False:
        i = 10
        return i + 15
    f = delayed(add, pure=True)
    v = f(1, 2)
    assert v.dask == {v.key: (add, 1, 2)}
    assert f.dask == {f.key: add}
    assert f.compute() == add

def test_delayed_name_on_call():
    if False:
        print('Hello World!')
    f = delayed(add, pure=True)
    assert f(1, 2, dask_key_name='foo')._key == 'foo'

def test_callable_obj():
    if False:
        while True:
            i = 10

    class Foo:

        def __init__(self, a):
            if False:
                while True:
                    i = 10
            self.a = a

        def __call__(self):
            if False:
                print('Hello World!')
            return 2
    foo = Foo(1)
    f = delayed(foo)
    assert f.compute() is foo
    assert f.a.compute() == 1
    assert f().compute() == 2

def identity(x):
    if False:
        return 10
    return x

def test_name_consistent_across_instances():
    if False:
        i = 10
        return i + 15
    func = delayed(identity, pure=True)
    data = {'x': 1, 'y': 25, 'z': [1, 2, 3]}
    assert func(data)._key == 'identity-4f318f3c27b869239e97c3ac07f7201a'
    data = {'x': 1, 1: 'x'}
    assert func(data)._key == func(data)._key
    assert func(1)._key == 'identity-7258833899272585e16d0ec36b21a3de'

def test_sensitive_to_partials():
    if False:
        for i in range(10):
            print('nop')
    assert delayed(partial(add, 10), pure=True)(2)._key != delayed(partial(add, 20), pure=True)(2)._key

def test_delayed_name():
    if False:
        print('Hello World!')
    assert delayed(1)._key.startswith('int-')
    assert delayed(1, pure=True)._key.startswith('int-')
    assert delayed(1, name='X')._key == 'X'

    def myfunc(x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1
    assert delayed(myfunc)(1).key.startswith('myfunc')

def test_finalize_name():
    if False:
        print('Hello World!')
    da = pytest.importorskip('dask.array')
    x = da.ones(10, chunks=5)
    v = delayed([x])
    assert set(x.dask).issubset(v.dask)

    def key(s):
        if False:
            print('Hello World!')
        if isinstance(s, tuple):
            s = s[0]
        return s.split('-')[0].replace('_', '')
    assert all((key(k).isalpha() for k in v.dask))

def test_keys_from_array():
    if False:
        print('Hello World!')
    da = pytest.importorskip('dask.array')
    from dask.array.utils import _check_dsk
    X = da.ones((10, 10), chunks=5).to_delayed().flatten()
    xs = [delayed(inc)(x) for x in X]
    _check_dsk(xs[0].dask)

def test_delayed_decorator_on_method():
    if False:
        while True:
            i = 10

    class A:
        BASE = 10

        def __init__(self, base):
            if False:
                return 10
            self.BASE = base

        @delayed
        def addmethod(self, x, y):
            if False:
                print('Hello World!')
            return self.BASE + x + y

        @classmethod
        @delayed
        def addclass(cls, x, y):
            if False:
                return 10
            return cls.BASE + x + y

        @staticmethod
        @delayed
        def addstatic(x, y):
            if False:
                return 10
            return x + y
    a = A(100)
    assert a.addmethod(3, 4).compute() == 107
    assert A.addmethod(a, 3, 4).compute() == 107
    assert a.addclass(3, 4).compute() == 17
    assert A.addclass(3, 4).compute() == 17
    assert a.addstatic(3, 4).compute() == 7
    assert A.addstatic(3, 4).compute() == 7
    assert isinstance(a.addmethod, types.MethodType)
    assert isinstance(A.addclass, types.MethodType)
    assert isinstance(A.addstatic, Delayed)

def test_attribute_of_attribute():
    if False:
        for i in range(10):
            print('nop')
    x = delayed(123)
    assert isinstance(x.a, Delayed)
    assert isinstance(x.a.b, Delayed)
    assert isinstance(x.a.b.c, Delayed)

def test_check_meta_flag():
    if False:
        for i in range(10):
            print('nop')
    dd = pytest.importorskip('dask.dataframe')
    from pandas import Series
    a = Series(['a', 'b', 'a'], dtype='category')
    b = Series(['a', 'c', 'a'], dtype='category')
    da = delayed(lambda x: x)(a)
    db = delayed(lambda x: x)(b)
    c = dd.from_delayed([da, db], verify_meta=False)
    dd.utils.assert_eq(c, c)

def modlevel_eager(x):
    if False:
        i = 10
        return i + 15
    return x + 1

@delayed
def modlevel_delayed1(x):
    if False:
        print('Hello World!')
    return x + 1

@delayed(pure=False)
def modlevel_delayed2(x):
    if False:
        return 10
    return x + 1

@pytest.mark.parametrize('f', [delayed(modlevel_eager), pytest.param(modlevel_delayed1, marks=pytest.mark.xfail(reason='#3369')), pytest.param(modlevel_delayed2, marks=pytest.mark.xfail(reason='#3369'))])
def test_pickle(f):
    if False:
        print('Hello World!')
    d = f(2)
    d = pickle.loads(pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
    assert d.compute() == 3

@pytest.mark.parametrize('f', [delayed(modlevel_eager), modlevel_delayed1, modlevel_delayed2])
def test_cloudpickle(f):
    if False:
        i = 10
        return i + 15
    d = f(2)
    d = cloudpickle.loads(cloudpickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
    assert d.compute() == 3

def test_dask_layers():
    if False:
        print('Hello World!')
    d1 = delayed(1)
    assert d1.dask.layers.keys() == {d1.key}
    assert d1.dask.dependencies == {d1.key: set()}
    assert d1.__dask_layers__() == (d1.key,)
    d2 = modlevel_delayed1(d1)
    assert d2.dask.layers.keys() == {d1.key, d2.key}
    assert d2.dask.dependencies == {d1.key: set(), d2.key: {d1.key}}
    assert d2.__dask_layers__() == (d2.key,)
    hlg = HighLevelGraph.from_collections('foo', {'alias': d2.key}, dependencies=[d2])
    with pytest.raises(ValueError, match='not in'):
        Delayed('alias', hlg)
    explicit = Delayed('alias', hlg, layer='foo')
    assert explicit.__dask_layers__() == ('foo',)
    explicit.dask.validate()

def test_annotations_survive_optimization():
    if False:
        for i in range(10):
            print('nop')
    with dask.annotate(foo='bar'):
        graph = HighLevelGraph.from_collections('b', {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')}, [])
        d = Delayed('b', graph)
    assert type(d.dask) is HighLevelGraph
    assert len(d.dask.layers) == 1
    assert len(d.dask.layers['b']) == 3
    assert d.dask.layers['b'].annotations == {'foo': 'bar'}
    (d_opt,) = dask.optimize(d)
    assert type(d_opt.dask) is HighLevelGraph
    assert len(d_opt.dask.layers) == 1
    assert len(d_opt.dask.layers['b']) == 2
    assert d_opt.dask.layers['b'].annotations == {'foo': 'bar'}

def test_delayed_function_attributes_forwarded():
    if False:
        print('Hello World!')

    @delayed
    def add(x, y):
        if False:
            i = 10
            return i + 15
        'This is a docstring'
        return x + y
    assert add.__name__ == 'add'
    assert add.__doc__ == 'This is a docstring'