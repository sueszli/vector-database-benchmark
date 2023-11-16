import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pytest import raises as assert_raises
from numpy.fft import fft, ifft
from scipy.signal import max_len_seq

class TestMLS:

    def test_mls_inputs(self):
        if False:
            return 10
        assert_raises(ValueError, max_len_seq, 10, state=np.zeros(10))
        assert_raises(ValueError, max_len_seq, 10, state=np.ones(3))
        assert_raises(ValueError, max_len_seq, 10, length=-1)
        assert_array_equal(max_len_seq(10, length=0)[0], [])
        assert_raises(ValueError, max_len_seq, 64)
        assert_raises(ValueError, max_len_seq, 10, taps=[-1, 1])

    def test_mls_output(self):
        if False:
            for i in range(10):
                print('nop')
        alt_taps = {2: [1], 3: [2], 4: [3], 5: [4, 3, 2], 6: [5, 4, 1], 7: [4], 8: [7, 5, 3]}
        for nbits in range(2, 8):
            for state in [None, np.round(np.random.rand(nbits))]:
                for taps in [None, alt_taps[nbits]]:
                    if state is not None and np.all(state == 0):
                        state[0] = 1
                    orig_m = max_len_seq(nbits, state=state, taps=taps)[0]
                    m = 2.0 * orig_m - 1.0
                    err_msg = 'mls had non binary terms'
                    assert_array_equal(np.abs(m), np.ones_like(m), err_msg=err_msg)
                    tester = np.real(ifft(fft(m) * np.conj(fft(m))))
                    out_len = 2 ** nbits - 1
                    err_msg = 'mls impulse has incorrect value'
                    assert_allclose(tester[0], out_len, err_msg=err_msg)
                    err_msg = 'mls steady-state has incorrect value'
                    assert_allclose(tester[1:], np.full(out_len - 1, -1), err_msg=err_msg)
                    for n in (1, 2 ** (nbits - 1)):
                        (m1, s1) = max_len_seq(nbits, state=state, taps=taps, length=n)
                        (m2, s2) = max_len_seq(nbits, state=s1, taps=taps, length=1)
                        (m3, s3) = max_len_seq(nbits, state=s2, taps=taps, length=out_len - n - 1)
                        new_m = np.concatenate((m1, m2, m3))
                        assert_array_equal(orig_m, new_m)