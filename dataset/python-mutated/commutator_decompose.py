"""Functions to compute the decomposition of an SO(3) matrix as balanced commutator."""
from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence

def _compute_trace_so3(matrix: np.ndarray) -> float:
    if False:
        print('Hello World!')
    'Computes trace of an SO(3)-matrix.\n\n    Args:\n        matrix: an SO(3)-matrix\n\n    Returns:\n        Trace of ``matrix``.\n\n    Raises:\n        ValueError: if ``matrix`` is not an SO(3)-matrix.\n    '
    _check_is_so3(matrix)
    trace = np.matrix.trace(matrix)
    trace_rounded = min(trace, 3)
    return trace_rounded

def _compute_rotation_axis(matrix: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Computes rotation axis of SO(3)-matrix.\n\n    Args:\n        matrix: The SO(3)-matrix for which rotation angle needs to be computed.\n\n    Returns:\n        The rotation axis of the SO(3)-matrix ``matrix``.\n\n    Raises:\n        ValueError: if ``matrix`` is not an SO(3)-matrix.\n    '
    _check_is_so3(matrix)
    trace = _compute_trace_so3(matrix)
    theta = math.acos(0.5 * (trace - 1))
    if math.sin(theta) > 1e-10:
        x = 1 / (2 * math.sin(theta)) * (matrix[2][1] - matrix[1][2])
        y = 1 / (2 * math.sin(theta)) * (matrix[0][2] - matrix[2][0])
        z = 1 / (2 * math.sin(theta)) * (matrix[1][0] - matrix[0][1])
    else:
        x = 1.0
        y = 0.0
        z = 0.0
    return np.array([x, y, z])

def _solve_decomposition_angle(matrix: np.ndarray) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Computes angle for balanced commutator of SO(3)-matrix ``matrix``.\n\n    Computes angle a so that the SO(3)-matrix ``matrix`` can be decomposed\n    as commutator [v,w] where v and w are both rotations of a about some axis.\n    The computation is done by solving a trigonometric equation using scipy.optimize.fsolve.\n\n    Args:\n        matrix: The SO(3)-matrix for which the decomposition angle needs to be computed.\n\n    Returns:\n        Angle a so that matrix = [v,w] with v and w rotations of a about some axis.\n\n    Raises:\n        ValueError: if ``matrix`` is not an SO(3)-matrix.\n    '
    from scipy.optimize import fsolve
    _check_is_so3(matrix)
    trace = _compute_trace_so3(matrix)
    angle = math.acos(1 / 2 * (trace - 1))
    lhs = math.sin(angle / 2)

    def objective(phi):
        if False:
            while True:
                i = 10
        sin_sq = np.sin(phi / 2) ** 2
        return 2 * sin_sq * np.sqrt(1 - sin_sq ** 2) - lhs
    decomposition_angle = fsolve(objective, angle)[0]
    return decomposition_angle

def _compute_rotation_from_angle_and_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Computes the SO(3)-matrix corresponding to the rotation of ``angle`` about ``axis``.\n\n    Args:\n        angle: The angle of the rotation.\n        axis: The axis of the rotation.\n\n    Returns:\n        SO(3)-matrix that represents a rotation of ``angle`` about ``axis``.\n\n    Raises:\n        ValueError: if ``axis`` is not a 3-dim unit vector.\n    '
    if axis.shape != (3,):
        raise ValueError(f'Axis must be a 1d array of length 3, but has shape {axis.shape}.')
    if abs(np.linalg.norm(axis) - 1.0) > 0.0001:
        raise ValueError(f'Axis must have a norm of 1, but has {np.linalg.norm(axis)}.')
    res = math.cos(angle) * np.identity(3) + math.sin(angle) * _cross_product_matrix(axis)
    res += (1 - math.cos(angle)) * np.outer(axis, axis)
    return res

def _compute_rotation_between(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Computes the SO(3)-matrix for rotating ``from_vector`` to ``to_vector``.\n\n    Args:\n        from_vector: unit vector of size 3\n        to_vector: unit vector of size 3\n\n    Returns:\n        SO(3)-matrix that brings ``from_vector`` to ``to_vector``.\n\n    Raises:\n        ValueError: if at least one of ``from_vector`` of ``to_vector`` is not a 3-dim unit vector.\n    '
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)
    dot = np.dot(from_vector, to_vector)
    cross = _cross_product_matrix(np.cross(from_vector, to_vector))
    rotation_matrix = np.identity(3) + cross + np.dot(cross, cross) / (1 + dot)
    return rotation_matrix

def _cross_product_matrix(v: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Computes cross product matrix from vector.\n\n    Args:\n        v: Vector for which cross product matrix needs to be computed.\n\n    Returns:\n        The cross product matrix corresponding to vector ``v``.\n    '
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def _compute_commutator_so3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Computes the commutator of the SO(3)-matrices ``a`` and ``b``.\n\n    The computation uses the fact that the inverse of an SO(3)-matrix is equal to its transpose.\n\n    Args:\n        a: SO(3)-matrix\n        b: SO(3)-matrix\n\n    Returns:\n        The commutator [a,b] of ``a`` and ``b`` w\n\n    Raises:\n        ValueError: if at least one of ``a`` or ``b`` is not an SO(3)-matrix.\n    '
    _check_is_so3(a)
    _check_is_so3(b)
    a_dagger = np.conj(a).T
    b_dagger = np.conj(b).T
    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)

def commutator_decompose(u_so3: np.ndarray, check_input: bool=True) -> tuple[GateSequence, GateSequence]:
    if False:
        return 10
    'Decompose an :math:`SO(3)`-matrix, :math:`U` as a balanced commutator.\n\n    This function finds two :math:`SO(3)` matrices :math:`V, W` such that the input matrix\n    equals\n\n    .. math::\n\n        U = V^\\dagger W^\\dagger V W.\n\n    For this decomposition, the following statement holds\n\n\n    .. math::\n\n        ||V - I||_F, ||W - I||_F \\leq \\frac{\\sqrt{||U - I||_F}}{2},\n\n    where :math:`I` is the identity and :math:`||\\cdot ||_F` is the Frobenius norm.\n\n    Args:\n        u_so3: SO(3)-matrix that needs to be decomposed as balanced commutator.\n        check_input: If True, checks whether the input matrix is actually SO(3).\n\n    Returns:\n        Tuple of GateSequences from SO(3)-matrices :math:`V, W`.\n\n    Raises:\n        ValueError: if ``u_so3`` is not an SO(3)-matrix.\n    '
    if check_input:
        _check_is_so3(u_so3)
        if not is_identity_matrix(u_so3.dot(u_so3.T)):
            raise ValueError('Input matrix is not orthogonal.')
    angle = _solve_decomposition_angle(u_so3)
    vx = _compute_rotation_from_angle_and_axis(angle, np.array([1, 0, 0]))
    wy = _compute_rotation_from_angle_and_axis(angle, np.array([0, 1, 0]))
    commutator = _compute_commutator_so3(vx, wy)
    u_so3_axis = _compute_rotation_axis(u_so3)
    commutator_axis = _compute_rotation_axis(commutator)
    sim_matrix = _compute_rotation_between(commutator_axis, u_so3_axis)
    sim_matrix_dagger = np.conj(sim_matrix).T
    v = np.dot(np.dot(sim_matrix, vx), sim_matrix_dagger)
    w = np.dot(np.dot(sim_matrix, wy), sim_matrix_dagger)
    return (GateSequence.from_matrix(v), GateSequence.from_matrix(w))