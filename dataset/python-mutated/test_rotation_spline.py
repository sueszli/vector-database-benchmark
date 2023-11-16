from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial.transform._rotation_spline import _angular_rate_to_rotvec_dot_matrix, _rotvec_dot_to_angular_rate_matrix, _matrix_vector_product_of_stacks, _angular_acceleration_nonlinear_term, _create_block_3_diagonal_matrix

def test_angular_rate_to_rotvec_conversions():
    if False:
        print('Hello World!')
    np.random.seed(0)
    rv = np.random.randn(4, 3)
    A = _angular_rate_to_rotvec_dot_matrix(rv)
    A_inv = _rotvec_dot_to_angular_rate_matrix(rv)
    assert_allclose(_matrix_vector_product_of_stacks(A, rv), rv)
    assert_allclose(_matrix_vector_product_of_stacks(A_inv, rv), rv)
    I_stack = np.empty((4, 3, 3))
    I_stack[:] = np.eye(3)
    assert_allclose(np.matmul(A, A_inv), I_stack, atol=1e-15)

def test_angular_rate_nonlinear_term():
    if False:
        print('Hello World!')
    np.random.seed(0)
    rv = np.random.rand(4, 3)
    assert_allclose(_angular_acceleration_nonlinear_term(rv, rv), 0, atol=1e-19)

def test_create_block_3_diagonal_matrix():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(0)
    A = np.empty((4, 3, 3))
    A[:] = np.arange(1, 5)[:, None, None]
    B = np.empty((4, 3, 3))
    B[:] = -np.arange(1, 5)[:, None, None]
    d = 10 * np.arange(10, 15)
    banded = _create_block_3_diagonal_matrix(A, B, d)
    (k, l) = list(zip(*product(np.arange(banded.shape[0]), np.arange(banded.shape[1]))))
    k = np.asarray(k)
    l = np.asarray(l)
    i = k - 5 + l
    j = l
    values = banded.ravel()
    mask = (i >= 0) & (i < 15)
    i = i[mask]
    j = j[mask]
    values = values[mask]
    full = np.zeros((15, 15))
    full[i, j] = values
    zero = np.zeros((3, 3))
    eye = np.eye(3)
    ref = np.block([[d[0] * eye, B[0], zero, zero, zero], [A[0], d[1] * eye, B[1], zero, zero], [zero, A[1], d[2] * eye, B[2], zero], [zero, zero, A[2], d[3] * eye, B[3]], [zero, zero, zero, A[3], d[4] * eye]])
    assert_allclose(full, ref, atol=1e-19)

def test_spline_2_rotations():
    if False:
        while True:
            i = 10
    times = [0, 10]
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [10, -20, 30]], degrees=True)
    spline = RotationSpline(times, rotations)
    rv = (rotations[0].inv() * rotations[1]).as_rotvec()
    rate = rv / (times[1] - times[0])
    times_check = np.array([-1, 5, 12])
    dt = times_check - times[0]
    rv_ref = rate * dt[:, None]
    assert_allclose(spline(times_check).as_rotvec(), rv_ref)
    assert_allclose(spline(times_check, 1), np.resize(rate, (3, 3)))
    assert_allclose(spline(times_check, 2), 0, atol=1e-16)

def test_constant_attitude():
    if False:
        while True:
            i = 10
    times = np.arange(10)
    rotations = Rotation.from_rotvec(np.ones((10, 3)))
    spline = RotationSpline(times, rotations)
    times_check = np.linspace(-1, 11)
    assert_allclose(spline(times_check).as_rotvec(), 1, rtol=1e-15)
    assert_allclose(spline(times_check, 1), 0, atol=1e-17)
    assert_allclose(spline(times_check, 2), 0, atol=1e-17)
    assert_allclose(spline(5.5).as_rotvec(), 1, rtol=1e-15)
    assert_allclose(spline(5.5, 1), 0, atol=1e-17)
    assert_allclose(spline(5.5, 2), 0, atol=1e-17)

def test_spline_properties():
    if False:
        print('Hello World!')
    times = np.array([0, 5, 15, 27])
    angles = [[-5, 10, 27], [3, 5, 38], [-12, 10, 25], [-15, 20, 11]]
    rotations = Rotation.from_euler('xyz', angles, degrees=True)
    spline = RotationSpline(times, rotations)
    assert_allclose(spline(times).as_euler('xyz', degrees=True), angles)
    assert_allclose(spline(0).as_euler('xyz', degrees=True), angles[0])
    h = 1e-08
    rv0 = spline(times).as_rotvec()
    rvm = spline(times - h).as_rotvec()
    rvp = spline(times + h).as_rotvec()
    assert_allclose(rv0, 0.5 * (rvp + rvm), rtol=1.5e-15)
    r0 = spline(times, 1)
    rm = spline(times - h, 1)
    rp = spline(times + h, 1)
    assert_allclose(r0, 0.5 * (rm + rp), rtol=1e-14)
    a0 = spline(times, 2)
    am = spline(times - h, 2)
    ap = spline(times + h, 2)
    assert_allclose(a0, am, rtol=1e-07)
    assert_allclose(a0, ap, rtol=1e-07)

def test_error_handling():
    if False:
        return 10
    raises(ValueError, RotationSpline, [1.0], Rotation.random())
    r = Rotation.random(10)
    t = np.arange(10).reshape(5, 2)
    raises(ValueError, RotationSpline, t, r)
    t = np.arange(9)
    raises(ValueError, RotationSpline, t, r)
    t = np.arange(10)
    t[5] = 0
    raises(ValueError, RotationSpline, t, r)
    t = np.arange(10)
    s = RotationSpline(t, r)
    raises(ValueError, s, 10, -1)
    raises(ValueError, s, np.arange(10).reshape(5, 2))