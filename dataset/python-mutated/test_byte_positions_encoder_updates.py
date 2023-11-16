import numpy as np
from deeplake.core.meta.encode.byte_positions import BytePositionsEncoder
from .common import assert_encoded

def _validate_bp(enc: BytePositionsEncoder):
    if False:
        print('Hello World!')
    'Helps validate that tests have proper initial encoder states.'
    expected = np.array(enc._encoded)
    enc._post_process_state(0)
    actual = enc._encoded
    np.testing.assert_array_equal(expected, actual)

def test_update_no_change():
    if False:
        return 10
    enc = BytePositionsEncoder([[8, 0, 10], [4, 88, 15]])
    _validate_bp(enc)
    enc[0] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    enc[1] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    enc[10] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    enc[11] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    enc[12] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    enc[15] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])
    assert enc.num_samples == 16

def test_update_squeeze_trivial():
    if False:
        for i in range(10):
            print('nop')
    enc = BytePositionsEncoder([[4, 0, 9], [2, 40, 10], [4, 42, 29]])
    _validate_bp(enc)
    enc[10] = 4
    assert_encoded(enc, [[4, 0, 29]])
    assert enc.num_samples == 30

def test_update_squeeze_complex():
    if False:
        print('Hello World!')
    enc = BytePositionsEncoder([[2, 0, 5], [4, 12, 9], [2, 28, 10], [4, 30, 29], [2, 106, 100]])
    _validate_bp(enc)
    enc[10] = 4
    assert_encoded(enc, [[2, 0, 5], [4, 12, 29], [2, 108, 100]])
    assert enc.num_samples == 101