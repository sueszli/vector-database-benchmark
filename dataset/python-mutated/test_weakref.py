import dill
dill.settings['recurse'] = True
import weakref

class _class:

    def _method(self):
        if False:
            print('Hello World!')
        pass

class _class2:

    def __call__(self):
        if False:
            print('Hello World!')
        pass

class _newclass(object):

    def _method(self):
        if False:
            print('Hello World!')
        pass

class _newclass2(object):

    def __call__(self):
        if False:
            while True:
                i = 10
        pass

def _function():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_weakref():
    if False:
        i = 10
        return i + 15
    o = _class()
    oc = _class2()
    n = _newclass()
    nc = _newclass2()
    f = _function
    z = _class
    x = _newclass
    r = weakref.ref(o)
    dr = weakref.ref(_class())
    p = weakref.proxy(o)
    dp = weakref.proxy(_class())
    c = weakref.proxy(oc)
    dc = weakref.proxy(_class2())
    m = weakref.ref(n)
    dm = weakref.ref(_newclass())
    t = weakref.proxy(n)
    dt = weakref.proxy(_newclass())
    d = weakref.proxy(nc)
    dd = weakref.proxy(_newclass2())
    fr = weakref.ref(f)
    fp = weakref.proxy(f)
    xr = weakref.ref(x)
    xp = weakref.proxy(x)
    objlist = [r, dr, m, dm, fr, xr, p, dp, t, dt, c, dc, d, dd, fp, xp]
    for obj in objlist:
        res = dill.detect.errors(obj)
        if res:
            print('%s' % res)
        assert not res

def test_dictproxy():
    if False:
        print('Hello World!')
    from dill._dill import DictProxyType
    try:
        m = DictProxyType({'foo': 'bar'})
    except:
        m = type.__dict__
    mp = dill.copy(m)
    assert mp.items() == m.items()
if __name__ == '__main__':
    test_weakref()
    from dill._dill import IS_PYPY
    if not IS_PYPY:
        test_dictproxy()