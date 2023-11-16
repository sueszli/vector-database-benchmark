import numpy as np
import pytest
from vispy.testing import run_tests_if_main
from vispy.visuals._scalable_textures import get_default_clim_from_dtype, get_default_clim_from_data, CPUScaledTextureMixin, GPUScaledTextureMixin

class Stub:
    _ndim = 2

    def __init__(self, data, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def set_data(self, data, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._data = data

class CPUScaledStub(CPUScaledTextureMixin, Stub):
    pass

class GPUScaledStub(GPUScaledTextureMixin, Stub):
    internalformat = 'r32f'

    def _get_texture_format_for_data(self, data, internalformat=None):
        if False:
            i = 10
            return i + 15
        return None

def test_default_clim():
    if False:
        return 10
    ref_data = np.array([10, 5, 15, 25, 15])
    data = ref_data.astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)
    data = ref_data.astype(np.int32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (-2 ** 31, 2 ** 31 - 1)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)
    data = ref_data.astype(np.uint8)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 255)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)

def test_default_clim_non_finite():
    if False:
        for i in range(10):
            print('nop')
    data = np.array([10, np.nan, 5, 15, 25, 15]).astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)
    data = np.array([10, np.inf, 5, 15, 25, 15]).astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)
    data = np.array([10, -np.inf, 5, 15, 25, 15]).astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (5, 25)
    data = np.array([np.nan, np.nan, np.nan]).astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (0, 0)
    data = np.array([np.nan, np.inf, -np.inf]).astype(np.float32)
    clim = get_default_clim_from_dtype(data.dtype)
    assert clim == (0, 1)
    clim = get_default_clim_from_data(data)
    assert clim == (0, 0)

def test_clim_handling_cpu():
    if False:
        i = 10
        return i + 15
    ref_data = np.array([[10, 10, 5], [15, 25, 15]])
    st = CPUScaledStub()
    st.set_clim('auto')
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (5, 25)
    assert st.clim_normalized == (0, 1)
    assert np.all(st._data == (ref_data - 5) / 20)
    st.set_clim((0, 20))
    assert st.clim == (0, 20)
    assert st.clim_normalized == (-0.25, 0.75)
    assert np.all(st._data == (ref_data - 5) / 20)
    st = CPUScaledStub()
    st.set_clim((0, 20))
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (0, 20)
    assert st.clim_normalized == (0, 1)
    assert np.all(st._data == ref_data / 20)
    st = CPUScaledStub()
    st.set_clim((10, 10))
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (10, 10)
    assert st.clim_normalized == (0, np.inf)
    st = CPUScaledStub()
    st.set_clim('auto')
    assert st.clim == 'auto'
    pytest.raises(RuntimeError, getattr, st, 'clim_normalized')
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (5, 25)
    assert st.clim_normalized == (0, 1)

def test_clim_handling_gpu():
    if False:
        while True:
            i = 10
    ref_data = np.array([[10, 10, 5], [15, 25, 15]])
    st = GPUScaledStub()
    st.set_clim('auto')
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (5, 25)
    assert st.clim_normalized == (5, 25)
    assert np.all(st._data == ref_data)
    st.set_clim((0, 20))
    assert st.clim == (0, 20)
    assert st.clim_normalized == (0, 20)
    assert np.all(st._data == ref_data)
    st = GPUScaledStub()
    st.set_clim((0, 20))
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (0, 20)
    assert st.clim_normalized == (0, 20)
    assert np.all(st._data == ref_data)
    st = GPUScaledStub()
    st.set_clim((10, 10))
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (10, 10)
    assert st.clim_normalized == (10, np.inf)
    st = GPUScaledStub()
    st.set_clim('auto')
    assert st.clim == 'auto'
    pytest.raises(RuntimeError, getattr, st, 'clim_normalized')
    st.scale_and_set_data(ref_data.astype(np.float32))
    assert st.clim == (5.0, 25.0)
    assert st.clim_normalized == (5.0, 25.0)
run_tests_if_main()