from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os

@pytest.fixture(scope='module')
def x():
    if False:
        for i in range(10):
            print('nop')
    return np.random.randn(512, 128)

@pytest.mark.parametrize('func', [fft.fft, fft.ifft, fft.fft2, fft.ifft2, fft.fftn, fft.ifftn, fft.rfft, fft.irfft, fft.rfft2, fft.irfft2, fft.rfftn, fft.irfftn, fft.hfft, fft.ihfft, fft.hfft2, fft.ihfft2, fft.hfftn, fft.ihfftn, fft.dct, fft.idct, fft.dctn, fft.idctn, fft.dst, fft.idst, fft.dstn, fft.idstn])
@pytest.mark.parametrize('workers', [2, -1])
def test_threaded_same(x, func, workers):
    if False:
        for i in range(10):
            print('nop')
    expected = func(x, workers=1)
    actual = func(x, workers=workers)
    assert_allclose(actual, expected)

def _mt_fft(x):
    if False:
        for i in range(10):
            print('nop')
    return fft.fft(x, workers=2)

def test_mixed_threads_processes(x):
    if False:
        while True:
            i = 10
    expect = fft.fft(x, workers=2)
    with multiprocessing.Pool(2) as p:
        res = p.map(_mt_fft, [x for _ in range(4)])
    for r in res:
        assert_allclose(r, expect)
    fft.fft(x, workers=2)

def test_invalid_workers(x):
    if False:
        for i in range(10):
            print('nop')
    cpus = os.cpu_count()
    fft.ifft([1], workers=-cpus)
    with pytest.raises(ValueError, match='workers must not be zero'):
        fft.fft(x, workers=0)
    with pytest.raises(ValueError, match='workers value out of range'):
        fft.ifft(x, workers=-cpus - 1)

def test_set_get_workers():
    if False:
        for i in range(10):
            print('nop')
    cpus = os.cpu_count()
    assert fft.get_workers() == 1
    with fft.set_workers(4):
        assert fft.get_workers() == 4
        with fft.set_workers(-1):
            assert fft.get_workers() == cpus
        assert fft.get_workers() == 4
    assert fft.get_workers() == 1
    with fft.set_workers(-cpus):
        assert fft.get_workers() == 1

def test_set_workers_invalid():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='workers must not be zero'):
        with fft.set_workers(0):
            pass
    with pytest.raises(ValueError, match='workers value out of range'):
        with fft.set_workers(-os.cpu_count() - 1):
            pass