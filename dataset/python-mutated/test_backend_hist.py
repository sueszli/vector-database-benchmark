from builtins import str
import numpy as np
import itertools as itt
import pytest
from neon.backends.util import check_gpu
from utils import tensors_allclose

def ref_hist(inp, nbins=64, offset=-48):
    if False:
        print('Hello World!')
    '\n    Implement a log2 histogram geared towards visualizing neural net parameters.\n\n    Bins are computed as the log2 of the magnitude of a tensor value.  Bins are\n    rounded to the nearest int.\n\n    Smallest value bin extends to -Inf to enable visualization of zeros.\n\n    Log2 computation is always done in fp32 regardless of input dtype to give\n    rounding a consistent behavior.\n    '
    bins = np.arange(nbins + 1) + float(offset)
    bins[0] = -float('Inf')
    np_inp_log_abs = np.rint(np.log2(np.abs(inp.astype(np.float32))))
    (np_hist, edges) = np.histogram(np_inp_log_abs, density=False, bins=bins)
    if np_hist.ndim < 2:
        np_hist = np_hist.reshape(1, np_hist.size)
    return np_hist

def pytest_generate_tests(metafunc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build a list of test arguments for test_hist.\n\n    Run a full but slow set if --all is specified as a py.test arg, or just\n    run sanity tests otherwise.\n    '
    bin_offs = [(64, -48), (32, 0)]
    dims = [(64, 32768), (64, 1)]
    dtypes = [np.float32, np.uint8]
    inputs = [('normal dist', lambda dim: np.random.normal(64, 4, dim[0] * dim[1]).reshape(dim))]
    if metafunc.config.option.all:
        bin_offs.extend([(64, -32), (32, -16)])
        dims.extend([(64, 387200), (128, 128), (2, 32), (1, 1)])
        dtypes.extend([np.float16, np.int8])
    if 'nbin_offset_dim_dtype_inp' in metafunc.fixturenames:
        fargs = itt.product(bin_offs, dims, dtypes, inputs)
        metafunc.parametrize('nbin_offset_dim_dtype_inp', fargs)

def test_edge_cases_mkl(backend_pair_mkl):
    if False:
        return 10
    '\n    Test several edge cases related to min/max bin, and rounding.\n\n    Also test backend dump_hist_data functionality.\n    '
    (nm, nc) = backend_pair_mkl
    np_ref = dict()
    inputs = [('edges', np.array([2 ** (-48), 2 ** 15], dtype=np.float32)), ('rounding', np.array([2 ** 5, 63.99998856, 2 ** 6, 2 ** (-3), 2 ** (-4), 0.11262291, 92.22483826], dtype=np.float32)), ('fp16 rounding', np.array([45.21875], dtype=np.float16))]
    for (tag, inp) in inputs:
        np_ref[tag] = ref_hist(inp)
        for be in [nm, nc]:
            be_inp = be.array(inp)
            be_hist = be_inp.hist(tag)
            assert tensors_allclose(np_ref[tag], be_hist), tag + str(be)
    for be in [nm, nc]:
        (be_hist_data, be_hist_map) = be.dump_hist_data()
        for (tag, inp) in inputs:
            be_data = be_hist_data[be_hist_map[tag]]
            assert tensors_allclose(np_ref[tag], be_data), tag + str(be)

def test_hist_mkl(nbin_offset_dim_dtype_inp, backend_pair_mkl):
    if False:
        i = 10
        return i + 15
    '\n    Compare the nervanamkl and nervanacpu hist implementation to the reference\n    implementation above.\n\n    Parameterized test case, uses pytest_generate_test to enumerate dim_dtype_inp\n    tuples that drive the test.\n    '
    ((nbins, offset), dim, dtype, (name, inp_gen)) = nbin_offset_dim_dtype_inp
    (nm, nc) = backend_pair_mkl
    nm.set_hist_buffers(nbins, offset)
    nc.set_hist_buffers(nbins, offset)
    np_inp = inp_gen(dim).astype(dtype)
    np_hist = ref_hist(np_inp, nbins=nbins, offset=offset)
    for be in [nm, nc]:
        be_inp = be.array(np_inp, dtype=dtype)
        be_hist = be_inp.hist(name)
        assert tensors_allclose(np_hist, be_hist)

@pytest.mark.hasgpu
def test_edge_cases(backend_pair):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test several edge cases related to min/max bin, and rounding.\n\n    Also test backend dump_hist_data functionality.\n    '
    gpuflag = check_gpu.get_compute_capability(0) >= 3.0
    if gpuflag is False:
        raise RuntimeError('Device does not have CUDA compute capability 3.0 or greater')
    (ng, nc) = backend_pair
    np_ref = dict()
    inputs = [('edges', np.array([2 ** (-48), 2 ** 15], dtype=np.float32)), ('rounding', np.array([2 ** 5, 63.99998856, 2 ** 6, 2 ** (-3), 2 ** (-4), 0.11262291, 92.22483826], dtype=np.float32)), ('fp16 rounding', np.array([45.21875], dtype=np.float16))]
    for (tag, inp) in inputs:
        np_ref[tag] = ref_hist(inp)
        for be in [ng, nc]:
            be_inp = be.array(inp)
            be_hist = be_inp.hist(tag)
            assert tensors_allclose(np_ref[tag], be_hist), tag + str(be)
    for be in [ng, nc]:
        (be_hist_data, be_hist_map) = be.dump_hist_data()
        for (tag, inp) in inputs:
            be_data = be_hist_data[be_hist_map[tag]]
            assert tensors_allclose(np_ref[tag], be_data), tag + str(be)

@pytest.mark.hasgpu
def test_hist(nbin_offset_dim_dtype_inp, backend_pair):
    if False:
        print('Hello World!')
    '\n    Compare the nervanagpu and nervanacpu hist implementation to the reference\n    implementation above.\n\n    Parameterized test case, uses pytest_generate_test to enumerate dim_dtype_inp\n    tuples that drive the test.\n    '
    ((nbins, offset), dim, dtype, (name, inp_gen)) = nbin_offset_dim_dtype_inp
    gpuflag = check_gpu.get_compute_capability(0) >= 3.0
    if gpuflag is False:
        raise RuntimeError('Device does not have CUDA compute capability 3.0 or greater')
    (ng, nc) = backend_pair
    ng.set_hist_buffers(nbins, offset)
    nc.set_hist_buffers(nbins, offset)
    np_inp = inp_gen(dim).astype(dtype)
    np_hist = ref_hist(np_inp, nbins=nbins, offset=offset)
    for be in [ng, nc]:
        be_inp = be.array(np_inp, dtype=dtype)
        be_hist = be_inp.hist(name)
        assert tensors_allclose(np_hist, be_hist)