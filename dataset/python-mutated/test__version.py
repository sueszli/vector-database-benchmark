"""Tests for the NumpyVersion class.

"""
from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion

def test_main_versions():
    if False:
        for i in range(10):
            print('nop')
    assert_(NumpyVersion('1.8.0') == '1.8.0')
    for ver in ['1.9.0', '2.0.0', '1.8.1', '10.0.1']:
        assert_(NumpyVersion('1.8.0') < ver)
    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert_(NumpyVersion('1.8.0') > ver)

def test_version_1_point_10():
    if False:
        for i in range(10):
            print('nop')
    assert_(NumpyVersion('1.9.0') < '1.10.0')
    assert_(NumpyVersion('1.11.0') < '1.11.1')
    assert_(NumpyVersion('1.11.0') == '1.11.0')
    assert_(NumpyVersion('1.99.11') < '1.99.12')

def test_alpha_beta_rc():
    if False:
        i = 10
        return i + 15
    assert_(NumpyVersion('1.8.0rc1') == '1.8.0rc1')
    for ver in ['1.8.0', '1.8.0rc2']:
        assert_(NumpyVersion('1.8.0rc1') < ver)
    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert_(NumpyVersion('1.8.0rc1') > ver)
    assert_(NumpyVersion('1.8.0b1') > '1.8.0a2')

def test_dev_version():
    if False:
        print('Hello World!')
    assert_(NumpyVersion('1.9.0.dev-Unknown') < '1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev-ffffffff']:
        assert_(NumpyVersion('1.9.0.dev-f16acvda') < ver)
    assert_(NumpyVersion('1.9.0.dev-f16acvda') == '1.9.0.dev-11111111')

def test_dev_a_b_rc_mixed():
    if False:
        for i in range(10):
            print('nop')
    assert_(NumpyVersion('1.9.0a2.dev-f16acvda') == '1.9.0a2.dev-11111111')
    assert_(NumpyVersion('1.9.0a2.dev-6acvda54') < '1.9.0a2')

def test_dev0_version():
    if False:
        while True:
            i = 10
    assert_(NumpyVersion('1.9.0.dev0+Unknown') < '1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert_(NumpyVersion('1.9.0.dev0+f16acvda') < ver)
    assert_(NumpyVersion('1.9.0.dev0+f16acvda') == '1.9.0.dev0+11111111')

def test_dev0_a_b_rc_mixed():
    if False:
        for i in range(10):
            print('nop')
    assert_(NumpyVersion('1.9.0a2.dev0+f16acvda') == '1.9.0a2.dev0+11111111')
    assert_(NumpyVersion('1.9.0a2.dev0+6acvda54') < '1.9.0a2')

def test_raises():
    if False:
        while True:
            i = 10
    for ver in ['1.9', '1,9.0', '1.7.x']:
        assert_raises(ValueError, NumpyVersion, ver)