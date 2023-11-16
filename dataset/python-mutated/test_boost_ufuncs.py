import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import _boost
type_char_to_type_tol = {'f': (np.float32, 32 * np.finfo(np.float32).eps), 'd': (np.float64, 32 * np.finfo(np.float64).eps)}
test_data = [(_boost._beta_cdf, (0.5, 2, 3), 0.6875), (_boost._beta_ppf, (0.6875, 2, 3), 0.5), (_boost._beta_pdf, (0.5, 2, 3), 1.5), (_boost._beta_pdf, (0, 1, 5), 5.0), (_boost._beta_pdf, (1, 5, 1), 5.0), (_boost._beta_sf, (0.5, 2, 1), 0.75), (_boost._beta_isf, (0.75, 2, 1), 0.5), (_boost._binom_cdf, (1, 3, 0.5), 0.5), (_boost._binom_pdf, (1, 4, 0.5), 0.25), (_boost._hypergeom_cdf, (2, 3, 5, 6), 0.5), (_boost._nbinom_cdf, (1, 4, 0.25), 0.015625), (_boost._ncf_mean, (10, 12, 2.5), 1.5)]

@pytest.mark.parametrize('func, args, expected', test_data)
def test_stats_boost_ufunc(func, args, expected):
    if False:
        print('Hello World!')
    type_sigs = func.types
    type_chars = [sig.split('->')[-1] for sig in type_sigs]
    for type_char in type_chars:
        (typ, rtol) = type_char_to_type_tol[type_char]
        args = [typ(arg) for arg in args]
        with np.errstate(over='ignore'):
            value = func(*args)
        assert isinstance(value, typ)
        assert_allclose(value, expected, rtol=rtol)