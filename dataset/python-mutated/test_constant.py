import numpy as np
import scipy.sparse.linalg as sparla
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import psd_wrap

def test_is_psd() -> None:
    if False:
        i = 10
        return i + 15
    n = 50
    psd = np.eye(n)
    nsd = -np.eye(n)
    assert cp.Constant(psd).is_psd()
    assert not cp.Constant(psd).is_nsd()
    assert cp.Constant(nsd).is_nsd()
    assert not cp.Constant(nsd).is_psd()
    failures = set()
    for seed in range(95, 100):
        np.random.seed(seed)
        P = np.random.randn(n, n)
        P = P.T @ P
        try:
            cp.Constant(P).is_psd()
        except sparla.ArpackNoConvergence as e:
            assert 'CVXPY note' in str(e)
            failures.add(seed)
    assert failures == {97}
    assert psd_wrap(cp.Constant(P)).is_psd()

def test_print():
    if False:
        print('Hello World!')
    A = cp.Constant(np.ones((3, 3)))
    assert str(A) == '[[1.00 1.00 1.00]\n [1.00 1.00 1.00]\n [1.00 1.00 1.00]]'
    B = cp.Constant(np.ones((5, 2)))
    assert str(B) == '[[1.00 1.00]\n [1.00 1.00]\n ...\n [1.00 1.00]\n [1.00 1.00]]'
    default = s.PRINT_EDGEITEMS
    s.PRINT_EDGEITEMS = 10
    assert str(B) == '[[1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]]'
    s.PRINT_EDGEITEMS = default