"""
Tests for the initializer classes.
"""
import itertools as itt
import numpy as np
from neon import NervanaObject
from neon.initializers.initializer import Array, Constant, Uniform, Gaussian, GlorotUniform

def pytest_generate_tests(metafunc):
    if False:
        while True:
            i = 10
    if 'args' in metafunc.fixturenames:
        fargs = []
        dim1 = [1, 5]
        dim2 = [2, 10]
        fargs = itt.product(dim1, dim2)
        metafunc.parametrize('args', fargs)

def test_constant(backend_default, args):
    if False:
        i = 10
        return i + 15
    be = NervanaObject.be
    (dim1, dim2) = args
    shape = (dim1, dim2)
    const_arg = 3
    Wdev = be.empty(shape)
    const_init = Constant(const_arg)
    const_init.fill(Wdev)
    Whost = Wdev.get()
    flat = Whost.flatten()
    for elt in flat:
        assert elt == const_arg
    return

def test_array(backend_default, args):
    if False:
        return 10
    be = NervanaObject.be
    (dim1, dim2) = args
    shape = (dim1, dim2)
    Wloc = be.array(np.arange(shape[0] * shape[1]).reshape(shape))
    Wdev = be.empty(shape)
    init = Array(Wdev)
    init.fill(Wloc)
    assert np.all(np.equal(Wdev.get(), Wloc.get()))
    return

def test_uniform(backend_default, args):
    if False:
        for i in range(10):
            print('nop')
    be = NervanaObject.be
    (dim1, dim2) = args
    shape = (dim1, dim2)
    Wdev = be.empty(shape)
    uniform_init = Uniform(low=-5, high=15)
    uniform_init.fill(Wdev)
    Whost = Wdev.get()
    flat = Whost.flatten()
    for elt in flat:
        assert elt <= 15 and elt >= -5
    return

def test_gaussian(backend_default, args):
    if False:
        while True:
            i = 10
    be = NervanaObject.be
    (dim1, dim2) = args
    shape = (dim1, dim2)
    Wdev = be.empty(shape)
    gaussian_init = Gaussian(loc=10000, scale=1)
    gaussian_init.fill(Wdev)
    Whost = Wdev.get()
    flat = Whost.flatten()
    for elt in flat:
        assert elt >= 0
    return

def test_glorot(backend_default, args):
    if False:
        while True:
            i = 10
    be = NervanaObject.be
    shape_1 = (1, 2)
    shape_2 = (1000, 10000)
    Wdev_1 = be.empty(shape_1)
    Wdev_2 = be.empty(shape_2)
    glorot_init = GlorotUniform()
    glorot_init.fill(Wdev_1)
    glorot_init.fill(Wdev_2)
    Whost_1 = Wdev_1.get()
    Whost_2 = Wdev_2.get()
    mean_1 = np.mean(Whost_1)
    mean_2 = np.mean(Whost_2)
    assert np.abs(mean_1) > np.abs(mean_2)
    return