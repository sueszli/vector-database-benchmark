from itertools import product
import pytest
import numpy as np
from cupy import testing
from cupyx.scipy.signal._upfirdn import _output_len

def upfirdn_naive(x, h, up=1, down=1):
    if False:
        for i in range(10):
            print('nop')
    'Naive upfirdn processing in Python.\n\n    Note: arg order (x, h) differs to facilitate apply_along_axis use.\n    '
    h = np.asarray(h)
    out = np.zeros(len(x) * up, x.dtype)
    out[::up] = x
    out = np.convolve(h, out)[::down][:_output_len(len(h), len(x), up, down)]
    return out

def make_case(up, down, h, x_dtype, case):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(17)
    h = np.atleast_1d(h)
    x = {'tiny': np.ones(1, dtype=x_dtype), 'ones': np.ones(10, dtype=x_dtype), 'randn': rng.randn(10).astype(x_dtype), 'ramp': np.arange(10).astype(x_dtype)}[case]
    if 'case' == 'randn' and x_dtype in (np.complex64, np.complex128):
        x += 1j * rng.randn(10)
    return (x, h)

def make_case_2D(up, down, h, x_dtype, case):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(17)
    h = np.atleast_1d(h)
    if case == '2D':
        size = (3, 5)
        x = rng.randn(*size).astype(x_dtype)
        if x_dtype in (np.complex64, np.complex128):
            x += 1j * rng.randn(*size)
        return (x, h)
    elif case == '2D_noncontig':
        size = (3, 7)
        x = rng.randn(*size).astype(x_dtype)
        if x_dtype in (np.complex64, np.complex128):
            x += 1j * rng.randn(*size)
        x = x[::2, 1::3].T
        return (x, h)
    else:
        raise ValueError(f'unknown 2D_case, {case}.')
_UPFIRDN_TYPES = (int, np.float32, np.complex64, float, complex)
_upfirdn_modes = ['constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect', 'antisymmetric', 'antireflect', 'line']

@testing.with_requires('scipy')
class TestUpfirdn:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('len_h', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('len_x', [1, 2, 3, 4, 5])
    def test_singleton(self, xp, scp, len_h, len_x):
        if False:
            i = 10
            return i + 15
        h = xp.zeros(len_h)
        h[len_h // 2] = 1.0
        x = xp.ones(len_x)
        y = scp.signal.upfirdn(h, x, 1, 1)
        return y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shift_x(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        y = scp.signal.upfirdn(xp.asarray([1, 1]), xp.asarray([1.0]), 1, 1)
        y1 = scp.signal.upfirdn(xp.asarray([1, 1]), xp.asarray([0.0, 1.0]), 1, 1)
        return (y, y1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('len_h, len_x, up, down, expected', [(2, 2, 5, 2, [1, 0, 0, 0]), (2, 3, 6, 3, [1, 0, 1, 0, 1]), (2, 4, 4, 3, [1, 0, 0, 0, 1]), (3, 2, 6, 2, [1, 0, 0, 1, 0]), (4, 11, 3, 5, [1, 0, 0, 1, 0, 0, 1])])
    def test_length_factors(self, xp, scp, len_h, len_x, up, down, expected):
        if False:
            return 10
        h = xp.zeros(len_h)
        h[0] = 1.0
        x = xp.ones(len_x)
        y = scp.signal.upfirdn(h, x, up, down)
        return y

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    @pytest.mark.parametrize('down, want_len', [(2, 5015), (11, 912), (79, 127)])
    @pytest.mark.parametrize('dtype', _UPFIRDN_TYPES)
    def test_vs_convolve(self, xp, scp, dtype, down, want_len):
        if False:
            return 10
        random_state = np.random.RandomState(17)
        size = 10000
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        x = xp.asarray(x)
        h = scp.signal.firwin(31, 1.0 / down, window='hamming')
        y = scp.signal.upfirdn(h, x, up=1, down=down)
        return y

    @pytest.mark.xfail(reason='upfirdn `mode=...` not implemented')
    @pytest.mark.parametrize('size, h_len, mode, dtype', product([8], [4, 5, 26], _upfirdn_modes, [np.float32, np.float64, np.complex64, np.complex128]))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_modes(self, xp, scp, size, h_len, mode, dtype):
        if False:
            return 10
        random_state = np.random.RandomState(5)
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        x = xp.asarray(x)
        h = xp.arange(1, 1 + h_len, dtype=x.real.dtype)
        y = scp.signal.upfirdn(h, x, up=1, down=1, mode=mode)
        return y

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h', (1.0, 1j))
    @pytest.mark.parametrize('up, down', [(1, 1), (2, 2), (3, 2), (2, 3)])
    @pytest.mark.parametrize('case', ['tiny', 'ones', 'randn', 'ramp'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vs_naive_delta(self, x_dtype, h, up, down, case, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, h) = make_case(up, down, h, x_dtype, case)
        x = xp.asarray(x)
        h = xp.asarray(h)
        y = scp.signal.upfirdn(h, x, up, down)
        return y

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h', (1.0, 1j))
    @pytest.mark.parametrize('up, down', [(1, 1), (2, 2), (3, 2), (2, 3)])
    @pytest.mark.parametrize('case', ['2D', '2D_noncontig'])
    @pytest.mark.parametrize('axis', [0, 1, -1])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vs_naive_delta_2D(self, axis, x_dtype, h, up, down, case, xp, scp):
        if False:
            print('Hello World!')
        (x, h) = make_case_2D(up, down, h, x_dtype, case)
        x = xp.asarray(x)
        h = xp.asarray(h)
        y = scp.signal.upfirdn(h, x, up, down, axis=axis)
        return y

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('p_max, q_max', list(product((10, 100), (10, 100))))
    @pytest.mark.parametrize('case', ['tiny', 'ones', 'randn', 'ramp'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-07)
    def test_vs_naive(self, xp, scp, case, x_dtype, h_dtype, p_max, q_max):
        if False:
            print('Hello World!')
        n_reps = 3
        longest_h = 25
        random_state = np.random.RandomState(17)
        cases = []
        for _ in range(n_reps):
            p_add = q_max if p_max > q_max else 1
            q_add = p_max if q_max > p_max else 1
            p = random_state.randint(p_max) + p_add
            q = random_state.randint(q_max) + q_add
            len_h = random_state.randint(longest_h) + 1
            h = np.atleast_1d(random_state.randint(len_h))
            h = h.astype(h_dtype)
            if h_dtype == complex:
                h += 1j * random_state.randint(len_h)
            (x, h) = make_case(p, q, h, x_dtype, case)
            x = xp.asarray(x)
            h = xp.asarray(x)
            y = scp.signal.upfirdn(h, x, p, q)
            cases.append(y)
        return cases

def test_output_len_long_input():
    if False:
        print('Hello World!')
    len_h = 1001
    in_len = 10 ** 8
    up = 320
    down = 441
    out_len = _output_len(len_h, in_len, up, down)
    assert out_len == 72562360