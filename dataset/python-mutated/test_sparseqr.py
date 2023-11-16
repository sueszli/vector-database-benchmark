import pytest
from pytest_pyodide import run_in_pyodide

@pytest.mark.driver_timeout(40)
@run_in_pyodide(packages=['scipy', 'sparseqr'])
def test_scipy_linalg(selenium):
    if False:
        print('Hello World!')
    import numpy
    import scipy.sparse.linalg
    import sparseqr
    from numpy.testing import assert_allclose
    n_test = 10
    M = scipy.sparse.rand(n_test, n_test, density=0.1)
    (Q, R, E, rank) = sparseqr.qr(M)
    for ib in range(n_test):
        b = numpy.zeros(10)
        b[ib] = 1
        v = Q.dot(R.dot(b))
        w = M.dot(sparseqr.permutation_vector_to_matrix(E).dot(b))
        assert_allclose(v, w, rtol=1e-07, atol=1e-09)