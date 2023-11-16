import numpy as np
from hypothesis import given
from hypothesis.strategies import none

def test_numpy_prng_is_seeded():
    if False:
        for i in range(10):
            print('nop')
    first = []
    prng_state = np.random.get_state()

    @given(none())
    def inner(_):
        if False:
            print('Hello World!')
        val = np.random.bytes(10)
        if not first:
            first.append(val)
        assert val == first[0], 'Numpy random module should be reproducible'
    inner()
    np.testing.assert_array_equal(np.random.get_state()[1], prng_state[1], 'State was not restored.')