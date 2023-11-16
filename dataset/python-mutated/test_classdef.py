import dill
import sys
dill.settings['recurse'] = True

class _class:

    def _method(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ok(self):
        if False:
            while True:
                i = 10
        return True

class _class2:

    def __call__(self):
        if False:
            while True:
                i = 10
        pass

    def ok(self):
        if False:
            print('Hello World!')
        return True

class _newclass(object):

    def _method(self):
        if False:
            i = 10
            return i + 15
        pass

    def ok(self):
        if False:
            i = 10
            return i + 15
        return True

class _newclass2(object):

    def __call__(self):
        if False:
            return 10
        pass

    def ok(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class _meta(type):
    pass

def __call__(self):
    if False:
        print('Hello World!')
    pass

def ok(self):
    if False:
        for i in range(10):
            print('nop')
    return True
_mclass = _meta('_mclass', (object,), {'__call__': __call__, 'ok': ok})
del __call__
del ok
o = _class()
oc = _class2()
n = _newclass()
nc = _newclass2()
m = _mclass()

def test_class_instances():
    if False:
        i = 10
        return i + 15
    assert dill.pickles(o)
    assert dill.pickles(oc)
    assert dill.pickles(n)
    assert dill.pickles(nc)
    assert dill.pickles(m)

def test_class_objects():
    if False:
        print('Hello World!')
    clslist = [_class, _class2, _newclass, _newclass2, _mclass]
    objlist = [o, oc, n, nc, m]
    _clslist = [dill.dumps(obj) for obj in clslist]
    _objlist = [dill.dumps(obj) for obj in objlist]
    for obj in clslist:
        globals().pop(obj.__name__)
    del clslist
    for obj in ['o', 'oc', 'n', 'nc']:
        globals().pop(obj)
    del objlist
    del obj
    for (obj, cls) in zip(_objlist, _clslist):
        _cls = dill.loads(cls)
        _obj = dill.loads(obj)
        assert _obj.ok()
        assert _cls.ok(_cls())
        if _cls.__name__ == '_mclass':
            assert type(_cls).__name__ == '_meta'

def test_none():
    if False:
        return 10
    assert dill.pickles(type(None))
if hex(sys.hexversion) >= '0x20600f0':
    from collections import namedtuple
    Z = namedtuple('Z', ['a', 'b'])
    Zi = Z(0, 1)
    X = namedtuple('Y', ['a', 'b'])
    X.__name__ = 'X'
    if hex(sys.hexversion) >= '0x30300f0':
        X.__qualname__ = 'X'
    Xi = X(0, 1)
    Bad = namedtuple('FakeName', ['a', 'b'])
    Badi = Bad(0, 1)
else:
    Z = Zi = X = Xi = Bad = Badi = None

def test_namedtuple():
    if False:
        print('Hello World!')
    assert Z is dill.loads(dill.dumps(Z))
    assert Zi == dill.loads(dill.dumps(Zi))
    assert X is dill.loads(dill.dumps(X))
    assert Xi == dill.loads(dill.dumps(Xi))
    assert Bad is not dill.loads(dill.dumps(Bad))
    assert Bad._fields == dill.loads(dill.dumps(Bad))._fields
    assert tuple(Badi) == tuple(dill.loads(dill.dumps(Badi)))

def test_array_nested():
    if False:
        return 10
    try:
        import numpy as np
        x = np.array([1])
        y = (x,)
        dill.dumps(x)
        assert y == dill.loads(dill.dumps(y))
    except ImportError:
        pass

def test_array_subclass():
    if False:
        while True:
            i = 10
    try:
        import numpy as np

        class TestArray(np.ndarray):

            def __new__(cls, input_array, color):
                if False:
                    i = 10
                    return i + 15
                obj = np.asarray(input_array).view(cls)
                obj.color = color
                return obj

            def __array_finalize__(self, obj):
                if False:
                    for i in range(10):
                        print('nop')
                if obj is None:
                    return
                if isinstance(obj, type(self)):
                    self.color = obj.color

            def __getnewargs__(self):
                if False:
                    while True:
                        i = 10
                return (np.asarray(self), self.color)
        a1 = TestArray(np.zeros(100), color='green')
        assert dill.pickles(a1)
        assert a1.__dict__ == dill.copy(a1).__dict__
        a2 = a1[0:9]
        assert dill.pickles(a2)
        assert a2.__dict__ == dill.copy(a2).__dict__

        class TestArray2(np.ndarray):
            color = 'blue'
        a3 = TestArray2([1, 2, 3, 4, 5])
        a3.color = 'green'
        assert dill.pickles(a3)
        assert a3.__dict__ == dill.copy(a3).__dict__
    except ImportError:
        pass

def test_method_decorator():
    if False:
        for i in range(10):
            print('nop')

    class A(object):

        @classmethod
        def test(cls):
            if False:
                for i in range(10):
                    print('nop')
            pass
    a = A()
    res = dill.dumps(a)
    new_obj = dill.loads(res)
    new_obj.__class__.test()

class Y(object):
    __slots__ = ['y']

    def __init__(self, y):
        if False:
            print('Hello World!')
        self.y = y
value = 123
y = Y(value)

def test_slots():
    if False:
        print('Hello World!')
    assert dill.pickles(Y)
    assert dill.pickles(y)
    assert dill.pickles(Y.y)
    assert dill.copy(y).y == value
if __name__ == '__main__':
    test_class_instances()
    test_class_objects()
    test_none()
    test_namedtuple()
    test_array_nested()
    test_array_subclass()
    test_method_decorator()
    test_slots()