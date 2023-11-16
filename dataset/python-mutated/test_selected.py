"""
testing some selected object types
"""
import dill
dill.settings['recurse'] = True
verbose = False

def test_dict_contents():
    if False:
        while True:
            i = 10
    c = type.__dict__
    for (i, j) in c.items():
        ok = dill.pickles(j)
        if verbose:
            print('%s: %s, %s' % (ok, type(j), j))
        assert ok
    if verbose:
        print('')

def _g(x):
    if False:
        return 10
    yield x

def _f():
    if False:
        while True:
            i = 10
    try:
        raise
    except:
        from sys import exc_info
        (e, er, tb) = exc_info()
        return (er, tb)

class _d(object):

    def _method(self):
        if False:
            for i in range(10):
                print('nop')
        pass
from dill import objects
from dill import load_types
load_types(pickleable=True, unpickleable=False)
_newclass = objects['ClassObjectType']
del objects

def test_class_descriptors():
    if False:
        return 10
    d = _d.__dict__
    for i in d.values():
        ok = dill.pickles(i)
        if verbose:
            print('%s: %s, %s' % (ok, type(i), i))
        assert ok
    if verbose:
        print('')
    od = _newclass.__dict__
    for i in od.values():
        ok = dill.pickles(i)
        if verbose:
            print('%s: %s, %s' % (ok, type(i), i))
        assert ok
    if verbose:
        print('')

def test_class():
    if False:
        return 10
    o = _d()
    oo = _newclass()
    ok = dill.pickles(o)
    if verbose:
        print('%s: %s, %s' % (ok, type(o), o))
    assert ok
    ok = dill.pickles(oo)
    if verbose:
        print('%s: %s, %s' % (ok, type(oo), oo))
    assert ok
    if verbose:
        print('')

def test_frame_related():
    if False:
        while True:
            i = 10
    g = _g(1)
    f = g.gi_frame
    (e, t) = _f()
    _is = lambda ok: not ok if dill._dill.IS_PYPY else ok
    ok = dill.pickles(f)
    if verbose:
        print('%s: %s, %s' % (ok, type(f), f))
    assert _is(not ok)
    ok = dill.pickles(g)
    if verbose:
        print('%s: %s, %s' % (ok, type(g), g))
    assert _is(not ok)
    ok = dill.pickles(t)
    if verbose:
        print('%s: %s, %s' % (ok, type(t), t))
    assert not ok
    ok = dill.pickles(e)
    if verbose:
        print('%s: %s, %s' % (ok, type(e), e))
    assert ok
    if verbose:
        print('')
if __name__ == '__main__':
    test_frame_related()
    test_dict_contents()
    test_class()
    test_class_descriptors()