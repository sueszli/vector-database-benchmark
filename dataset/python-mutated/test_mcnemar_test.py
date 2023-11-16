import numpy as np
from numpy.testing import assert_almost_equal
from mlxtend.evaluate import mcnemar
from mlxtend.utils import assert_raises

def test_input_dimensions():
    if False:
        while True:
            i = 10
    t = np.ones((3, 3))
    assert_raises(ValueError, 'Input array must be a 2x2 array.', mcnemar, t)

def test_defaults():
    if False:
        print('Hello World!')
    tb = np.array([[101, 121], [59, 33]])
    (chi2, p) = (20.67222222222222, 5.450094825427117e-06)
    (chi2p, pp) = mcnemar(tb)
    assert_almost_equal(chi2, chi2p, decimal=7)
    assert_almost_equal(p, pp, decimal=7)

def test_corrected_false():
    if False:
        i = 10
        return i + 15
    tb = np.array([[101, 121], [59, 33]])
    (chi2, p) = (21.355555555555554, 3.815135865112594e-06)
    (chi2p, pp) = mcnemar(tb, corrected=False)
    assert_almost_equal(chi2, chi2p, decimal=7)
    assert_almost_equal(p, pp, decimal=7)

def test_exact():
    if False:
        while True:
            i = 10
    tb = np.array([[101, 121], [59, 33]])
    p = 4.43444926375551e-06
    (chi2p, pp) = mcnemar(tb, exact=True)
    assert chi2p is None
    assert_almost_equal(p, pp, decimal=7)
    assert p < 4.45e-06

def test_exact_corrected():
    if False:
        print('Hello World!')
    tb = np.array([[101, 121], [59, 33]])
    p = 4.43444926375551e-06
    (chi2p, pp) = mcnemar(tb, exact=True, corrected=False)
    assert chi2p is None
    assert_almost_equal(p, pp, decimal=7)
    assert p < 4.45e-06