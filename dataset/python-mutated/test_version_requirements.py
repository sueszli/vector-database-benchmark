"""Tests for the version requirement functions.

"""
import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing

def test_get_module_version():
    if False:
        return 10
    assert version_req.get_module_version('numpy')
    assert version_req.get_module_version('scipy')
    with testing.raises(ImportError):
        version_req.get_module_version('fakenumpy')

def test_is_installed():
    if False:
        for i in range(10):
            print('nop')
    assert version_req.is_installed('python', '>=2.7')
    assert not version_req.is_installed('numpy', '<1.0')

def test_require():
    if False:
        i = 10
        return i + 15

    @version_req.require('python', '>2.7')
    @version_req.require('numpy', '>1.5')
    def foo():
        if False:
            i = 10
            return i + 15
        return 1
    assert_equal(foo(), 1)

    @version_req.require('scipy', '<0.1')
    def bar():
        if False:
            while True:
                i = 10
        return 0
    with testing.raises(ImportError):
        bar()

def test_get_module():
    if False:
        print('Hello World!')
    assert version_req.get_module('numpy') is np