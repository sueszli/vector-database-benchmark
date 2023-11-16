from cleanlab.internal import latent_algebra
import numpy as np
import pytest
s = [0] * 10 + [1] * 5 + [2] * 15
nm = np.array([[1.0, 0.0, 0.2], [0.0, 0.7, 0.2], [0.0, 0.3, 0.6]])

def test_latent_py_ps_inv():
    if False:
        while True:
            i = 10
    (ps, py, inv) = latent_algebra.compute_ps_py_inv_noise_matrix(s, nm)
    assert all(abs(np.dot(inv, ps) - py) < 0.001)
    assert all(abs(np.dot(nm, py) - ps) < 0.001)

def get_latent_py_ps_inv():
    if False:
        print('Hello World!')
    (ps, py, inv) = latent_algebra.compute_ps_py_inv_noise_matrix(s, nm)
    return (ps, py, inv)

def test_latent_inv():
    if False:
        for i in range(10):
            print('nop')
    (ps, py, inv) = get_latent_py_ps_inv()
    inv2 = latent_algebra.compute_inv_noise_matrix(py, nm)
    assert np.all(abs(inv - inv2) < 0.001)

def test_latent_nm():
    if False:
        return 10
    (ps, py, inv) = get_latent_py_ps_inv()
    nm2 = latent_algebra.compute_noise_matrix_from_inverse(ps, inv, py=py)
    assert np.all(abs(nm - nm2) < 0.001)

def test_latent_py():
    if False:
        i = 10
        return i + 15
    (ps, py, inv) = get_latent_py_ps_inv()
    py2 = latent_algebra.compute_py(ps, nm, inv)
    assert np.all(abs(py - py2) < 0.001)

def test_latent_py_warning():
    if False:
        return 10
    (ps, py, inv) = get_latent_py_ps_inv()
    with pytest.raises(TypeError) as e:
        with pytest.warns(UserWarning) as w:
            py2 = latent_algebra.compute_py(ps=np.array([[[0.1, 0.3, 0.6]]]), noise_matrix=nm, inverse_noise_matrix=inv)
            py2 = latent_algebra.compute_py(ps=np.array([[0.1], [0.2], [0.7]]), noise_matrix=nm, inverse_noise_matrix=inv)

def test_compute_py_err():
    if False:
        return 10
    (ps, py, inv) = get_latent_py_ps_inv()
    try:
        py = latent_algebra.compute_py(ps=ps, noise_matrix=nm, inverse_noise_matrix=inv, py_method='marginal_ps')
    except ValueError as e:
        assert 'true_labels_class_counts' in str(e)
        with pytest.raises(ValueError) as e:
            py = latent_algebra.compute_py(ps=ps, noise_matrix=nm, inverse_noise_matrix=inv, py_method='marginal_ps')

def test_compute_py_marginal_ps():
    if False:
        i = 10
        return i + 15
    (ps, py, inv) = get_latent_py_ps_inv()
    cj = nm * ps * len(s)
    true_labels_class_counts = cj.sum(axis=0)
    py2 = latent_algebra.compute_py(ps=ps, noise_matrix=nm, inverse_noise_matrix=inv, py_method='marginal_ps', true_labels_class_counts=true_labels_class_counts)
    assert all(abs(py - py2) < 0.01)

def test_pyx():
    if False:
        return 10
    pred_probs = np.array([[0.1, 0.3, 0.6], [0.1, 0.0, 0.9], [0.1, 0.0, 0.9], [1.0, 0.0, 0.0], [0.1, 0.8, 0.1]])
    (ps, py, inv) = get_latent_py_ps_inv()
    pyx = latent_algebra.compute_pyx(pred_probs, nm, inv)
    assert np.all(np.sum(pyx, axis=1) - 1 < 0.0001)

def test_pyx_error():
    if False:
        i = 10
        return i + 15
    pred_probs = np.array([0.1, 0.3, 0.6])
    (ps, py, inv) = get_latent_py_ps_inv()
    try:
        pyx = latent_algebra.compute_pyx(pred_probs, nm, inverse_noise_matrix=inv)
    except ValueError as e:
        assert 'should be (N, K)' in str(e)
    with pytest.raises(ValueError) as e:
        pyx = latent_algebra.compute_pyx(pred_probs, nm, inverse_noise_matrix=inv)

def test_compute_py_method_marginal_true_labels_class_counts_none_error():
    if False:
        while True:
            i = 10
    (ps, py, inv) = get_latent_py_ps_inv()
    with pytest.raises(ValueError) as e:
        _ = latent_algebra.compute_py(ps=ps, noise_matrix=nm, inverse_noise_matrix=inv, py_method='marginal', true_labels_class_counts=None)