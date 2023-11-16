import pytest
from spyder_kernels.utils.lazymodules import LazyModule, FakeObject

def test_non_existent_module():
    if False:
        print('Hello World!')
    "Test that we retun FakeObject's for non-existing modules."
    mod = LazyModule('no_module', second_level_attrs=['a'])
    assert mod.foo is FakeObject
    assert mod.foo.a is FakeObject
    with pytest.raises(AttributeError):
        mod.foo.b

def test_existing_modules():
    if False:
        return 10
    'Test that lazy modules work for existing modules.'
    np = LazyModule('numpy')
    import numpy
    assert np.ndarray == numpy.ndarray
    assert np.__spy_mod__
    assert np.__spy_modname__