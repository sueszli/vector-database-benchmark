import importlib
import sys
import types
import pytest
import networkx.lazy_imports as lazy

def test_lazy_import_basics():
    if False:
        for i in range(10):
            print('nop')
    math = lazy._lazy_import('math')
    anything_not_real = lazy._lazy_import('anything_not_real')
    assert math.sin(math.pi) == pytest.approx(0, 1e-06)
    try:
        anything_not_real.pi
        assert False
    except ModuleNotFoundError:
        pass
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    try:
        anything_not_real.pi
        assert False
    except ModuleNotFoundError:
        pass

def test_lazy_import_impact_on_sys_modules():
    if False:
        print('Hello World!')
    math = lazy._lazy_import('math')
    anything_not_real = lazy._lazy_import('anything_not_real')
    assert type(math) == types.ModuleType
    assert 'math' in sys.modules
    assert type(anything_not_real) == lazy.DelayedImportErrorModule
    assert 'anything_not_real' not in sys.modules
    np_test = pytest.importorskip('numpy')
    np = lazy._lazy_import('numpy')
    assert type(np) == types.ModuleType
    assert 'numpy' in sys.modules
    np.pi
    assert type(np) == types.ModuleType
    assert 'numpy' in sys.modules

def test_lazy_import_nonbuiltins():
    if False:
        for i in range(10):
            print('nop')
    sp = lazy._lazy_import('scipy')
    np = lazy._lazy_import('numpy')
    if isinstance(sp, lazy.DelayedImportErrorModule):
        try:
            sp.special.erf
            assert False
        except ModuleNotFoundError:
            pass
    elif isinstance(np, lazy.DelayedImportErrorModule):
        try:
            np.sin(np.pi)
            assert False
        except ModuleNotFoundError:
            pass
    else:
        assert sp.special.erf(np.pi) == pytest.approx(1, 0.0001)

def test_lazy_attach():
    if False:
        return 10
    name = 'mymod'
    submods = ['mysubmodule', 'anothersubmodule']
    myall = {'not_real_submod': ['some_var_or_func']}
    locls = {'attach': lazy.attach, 'name': name, 'submods': submods, 'myall': myall}
    s = '__getattr__, __lazy_dir__, __all__ = attach(name, submods, myall)'
    exec(s, {}, locls)
    expected = {'attach': lazy.attach, 'name': name, 'submods': submods, 'myall': myall, '__getattr__': None, '__lazy_dir__': None, '__all__': None}
    assert locls.keys() == expected.keys()
    for (k, v) in expected.items():
        if v is not None:
            assert locls[k] == v