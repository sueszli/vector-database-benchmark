import math
import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from opensfm import pygeometry, pymap, pyrobust, transformations as tf

def nullspace(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Compute the null space of A.\n\n    Return the smallest singular value and the corresponding vector.\n    '
    (u, s, vh) = np.linalg.svd(A)
    return (s[-1], vh[-1])

def homogeneous(x: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Add a column of ones to x.'
    s = x.shape[:-1] + (1,)
    return np.hstack((x, np.ones(s)))

def homogeneous_vec(x: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Add a column of zeros to x.'
    s = x.shape[:-1] + (1,)
    return np.hstack((x, np.zeros(s)))

def euclidean(x: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Divide by last column and drop it.'
    return x[..., :-1] / x[..., -1:]

def cross_product_matrix(x: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    "Return the matrix representation of x's cross product"
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def P_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'P = K[R|t].'
    P = np.empty((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, t)
    return P

def KRt_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Factorize the camera matrix into K,R,t as P = K[R|t].\n\n    >>> K = np.array([[1, 2, 3],\n    ...               [0, 4, 5],\n    ...               [0, 0, 1]])\n    >>> R = np.array([[ 0.57313786, -0.60900664,  0.54829181],\n    ...               [ 0.74034884,  0.6716445 , -0.02787928],\n    ...               [-0.35127851,  0.42190588,  0.83582225]])\n    >>> t = np.array([1, 2, 3])\n    >>> P = P_from_KRt(K, R, t)\n    >>> KK, RR, tt = KRt_from_P(P)\n    >>> np.allclose(K, KK)\n    True\n    >>> np.allclose(R, RR)\n    True\n    >>> np.allclose(t, tt)\n    True\n    '
    (K, R) = rq(P[:, :3])
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)
    R = np.dot(T, R)
    t = np.linalg.solve(K, P[:, 3])
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    K /= K[2, 2]
    return (K, R, t)

def rq(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        return 10
    'Decompose a matrix into a triangular times rotation.\n    (from PCV)\n\n    >>> Q = np.array([[ 0.57313786, -0.60900664,  0.54829181],\n    ...               [ 0.74034884,  0.6716445 , -0.02787928],\n    ...               [-0.35127851,  0.42190588,  0.83582225]])\n    >>> R = np.array([[1, 2, 3],\n    ...               [0, 4, 5],\n    ...               [0, 0, 1]])\n    >>> r, q = rq(R.dot(Q))\n    >>> np.allclose(r.dot(q), R.dot(Q))\n    True\n    >>> np.allclose(abs(np.linalg.det(q)), 1.0)\n    True\n    >>> np.allclose(r[1,0], 0) and np.allclose(r[2,0], 0) and np.allclose(r[2,1], 0)\n    True\n    '
    (Q, R) = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    return (R[:, ::-1], Q[::-1, :])

def vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    if False:
        i = 10
        return i + 15
    'Angle between two vectors.\n\n    >>> u = [ 0.99500417, -0.33333333, -0.09983342]\n    >>> v = [ -0.99500417, +0.33333333, +0.09983342]\n    >>> vector_angle(u, u)\n    0.0\n    >>> np.isclose(vector_angle(u, v), np.pi)\n    True\n    '
    cos = np.dot(u, v) / math.sqrt(np.dot(u, u) * np.dot(v, v))
    cos = np.clip(cos, -1, 1)
    return math.acos(cos)

def decompose_similarity_transform(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    'Decompose the similarity transform to scale, rotation and translation'
    (m, n) = T.shape[0:2]
    assert m == n
    (A, b) = (T[:m - 1, :m - 1], T[:m - 1, m - 1])
    s = np.linalg.det(A) ** (1.0 / (m - 1))
    return (s, A / s, b)

def ransac_max_iterations(kernel: Any, inliers: np.ndarray, failure_probability: float) -> float:
    if False:
        return 10
    if len(inliers) >= kernel.num_samples():
        return 0
    inlier_ratio = float(len(inliers)) / kernel.num_samples()
    n = kernel.required_samples
    return math.log(failure_probability) / math.log(1.0 - inlier_ratio ** n)
TRansacSolution = Tuple[np.ndarray, np.ndarray, float]

def ransac(kernel: Any, threshold: float) -> TRansacSolution:
    if False:
        return 10
    'Robustly fit a model to data.\n\n    >>> x = np.array([1., 2., 3.])\n    >>> y = np.array([2., 4., 7.])\n    >>> kernel = TestLinearKernel(x, y)\n    >>> model, inliers, error = ransac(kernel, 0.1)\n    >>> np.allclose(model, 2.0)\n    True\n    >>> inliers\n    array([0, 1])\n    >>> np.allclose(error, 0.1)\n    True\n    '
    max_iterations = 1000
    best_error = float('inf')
    best_model = None
    best_inliers = []
    i = 0
    while i < max_iterations:
        try:
            samples = kernel.sampling()
        except AttributeError:
            samples = random.sample(range(kernel.num_samples()), kernel.required_samples)
        models = kernel.fit(samples)
        for model in models:
            errors = kernel.evaluate(model)
            inliers = np.flatnonzero(np.fabs(errors) < threshold)
            error = np.fabs(errors).clip(0, threshold).sum()
            if len(inliers) and error < best_error:
                best_error = error
                best_model = model
                best_inliers = inliers
                max_iterations = min(max_iterations, ransac_max_iterations(kernel, best_inliers, 0.01))
        i += 1
    return (best_model, best_inliers, best_error)

class TestLinearKernel:
    """A kernel for the model y = a * x.

    >>> x = np.array([1., 2., 3.])
    >>> y = np.array([2., 4., 7.])
    >>> kernel = TestLinearKernel(x, y)
    >>> models = kernel.fit([0])
    >>> models
    [2.0]
    >>> errors = kernel.evaluate(models[0])
    >>> np.allclose(errors, [0., 0., 1.])
    True
    """
    required_samples = 1

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if False:
            return 10
        self.x: np.ndarray = x
        self.y: np.ndarray = y

    def num_samples(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.x)

    def fit(self, samples: np.ndarray) -> List[float]:
        if False:
            return 10
        x = self.x[samples[0]]
        y = self.y[samples[0]]
        return [y / x]

    def evaluate(self, model: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        return self.y - model * self.x

class PlaneKernel:
    """
    A kernel for estimating plane from on-plane points and vectors
    """

    def __init__(self, points, vectors, verticals, point_threshold=1.0, vector_threshold=5.0) -> None:
        if False:
            print('Hello World!')
        self.points = points
        self.vectors = vectors
        self.verticals = verticals
        self.required_samples = 3
        self.point_threshold = point_threshold
        self.vector_threshold = vector_threshold

    def num_samples(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.points)

    def sampling(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        samples = {}
        if len(self.vectors) > 0:
            samples['points'] = self.points[random.sample(range(len(self.points)), 2), :]
            samples['vectors'] = [self.vectors[i] for i in random.sample(range(len(self.vectors)), 1)]
        else:
            samples['points'] = self.points[:, random.sample(range(len(self.points)), 3)]
            samples['vectors'] = None
        return samples

    def fit(self, samples: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if False:
            print('Hello World!')
        model = fit_plane(samples['points'], samples['vectors'], self.verticals)
        return [model]

    def evaluate(self, model) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        normal = model[0:3]
        normal_norm = np.linalg.norm(normal) + 1e-10
        point_error = np.abs(model.T.dot(homogeneous(self.points).T)) / normal_norm
        vectors = np.array(self.vectors)
        vector_norm = np.sum(vectors * vectors, axis=1)
        vectors = (vectors.T / vector_norm).T
        vector_error = abs(np.rad2deg(abs(np.arccos(vectors.dot(normal) / normal_norm))) - 90)
        vector_error[vector_error < self.vector_threshold] = 0.0
        vector_error[vector_error >= self.vector_threshold] = self.point_threshold + 0.1
        point_error[point_error < self.point_threshold] = 0.0
        point_error[point_error >= self.point_threshold] = self.point_threshold + 0.1
        errors = np.hstack((point_error, vector_error))
        return errors

def fit_plane_ransac(points: np.ndarray, vectors: np.ndarray, verticals: np.ndarray, point_threshold: float=1.2, vector_threshold: float=5.0) -> TRansacSolution:
    if False:
        while True:
            i = 10
    vectors = np.array([v / math.pi * 180.0 for v in vectors])
    kernel = PlaneKernel(points - points.mean(axis=0), vectors, verticals, point_threshold, vector_threshold)
    (p, inliers, error) = ransac(kernel, point_threshold)
    num_point = points.shape[0]
    points_inliers = points[inliers[inliers < num_point], :]
    vectors_inliers = np.array([vectors[i - num_point] for i in inliers[inliers >= num_point]])
    p = fit_plane(points_inliers - points_inliers.mean(axis=0), vectors_inliers, verticals)
    return (p, inliers, error)

def fit_plane(points: np.ndarray, vectors: Optional[np.ndarray], verticals: Optional[np.ndarray]) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Estimate a plane from on-plane points and vectors.\n\n    >>> x = [[0,0,0], [1,0,0], [0,1,0]]\n    >>> p = fit_plane(x, None, None)\n    >>> np.allclose(p, [0,0,1,0]) or np.allclose(p, [0,0,-1,0])\n    True\n    >>> x = [[0,0,0], [0,1,0]]\n    >>> v = [[1,0,0]]\n    >>> p = fit_plane(x, v, None)\n    >>> np.allclose(p, [0,0,1,0]) or np.allclose(p, [0,0,-1,0])\n    True\n    >>> vert = [[0,0,1]]\n    >>> p = fit_plane(x, v, vert)\n    >>> np.allclose(p, [0,0,1,0])\n    True\n    '
    points = np.array(points)
    s = 1.0 / max(1e-08, points.std())
    x = homogeneous(s * points)
    if vectors is not None and len(vectors) > 0:
        v = homogeneous_vec(s * np.array(vectors))
        A = np.vstack((x, v))
    else:
        A = x
    (evalues, evectors) = np.linalg.eig(A.T.dot(A))
    smallest_evalue_idx = min(enumerate(evalues), key=lambda x: x[1])[0]
    p = evectors[:, smallest_evalue_idx]
    if np.allclose(p[:3], [0, 0, 0]):
        return np.array([0.0, 0.0, 1.0, 0])
    if verticals is not None and len(verticals) > 0:
        d = 0
        for vertical in verticals:
            d += p[:3].dot(vertical)
        p *= np.sign(d)
    return p

def plane_horizontalling_rotation(p: np.ndarray) -> Optional[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Compute a rotation that brings p to z=0\n\n    >>> p = [1.0, 2.0, 3.0]\n    >>> R = plane_horizontalling_rotation(p)\n    >>> np.allclose(R.dot(p), [0, 0, np.linalg.norm(p)])\n    True\n\n    >>> p = [0, 0, 1.0]\n    >>> R = plane_horizontalling_rotation(p)\n    >>> np.allclose(R.dot(p), [0, 0, np.linalg.norm(p)])\n    True\n\n    >>> p = [0, 0, -1.0]\n    >>> R = plane_horizontalling_rotation(p)\n    >>> np.allclose(R.dot(p), [0, 0, np.linalg.norm(p)])\n    True\n\n    >>> p = [1e-14, 1e-14, -1.0]\n    >>> R = plane_horizontalling_rotation(p)\n    >>> np.allclose(R.dot(p), [0, 0, np.linalg.norm(p)])\n    True\n    '
    v0 = p[:3]
    v1 = np.array([0.0, 0.0, 1.0])
    angle = tf.angle_between_vectors(v0, v1)
    axis = tf.vector_product(v0, v1)
    if np.linalg.norm(axis) > 0:
        return tf.rotation_matrix(angle, axis)[:3, :3]
    elif angle < 1.0:
        return np.eye(3)
    elif angle > 3.0:
        return np.diag([1, -1, -1])
    return None

def fit_similarity_transform(p1: np.ndarray, p2: np.ndarray, max_iterations: int=1000, threshold: float=1) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        return 10
    'Fit a similarity transform T such as p2 = T . p1 between two points sets p1 and p2'
    (num_points, dim) = p1.shape[0:2]
    assert p1.shape[0] == p2.shape[0]
    best_inliers = []
    best_T = np.array((3, 4))
    for _ in range(max_iterations):
        rnd = np.random.permutation(num_points)
        rnd = rnd[0:dim]
        T = tf.affine_matrix_from_points(p1[rnd, :].T, p2[rnd, :].T, shear=False)
        p1h = homogeneous(p1)
        p2h = homogeneous(p2)
        errors = np.sqrt(np.sum((p2h.T - np.dot(T, p1h.T)) ** 2, axis=0))
        inliers = np.argwhere(errors < threshold)[:, 0]
        if len(inliers) >= len(best_inliers):
            best_T = T.copy()
            best_inliers = np.argwhere(errors < threshold)[:, 0]
    if len(best_inliers) > dim + 3:
        best_T = tf.affine_matrix_from_points(p1[best_inliers, :].T, p2[best_inliers, :].T, shear=False)
        errors = np.sqrt(np.sum((p2h.T - np.dot(best_T, p1h.T)) ** 2, axis=0))
        best_inliers = np.argwhere(errors < threshold)[:, 0]
    return (best_T, best_inliers)

def K_from_camera(camera: Dict[str, Any]) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    f = float(camera['focal'])
    return np.array([[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]])

def focal_from_homography(H: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Solve for w = H w H^t, with w = diag(a, a, b)\n\n    >>> K = np.diag([0.8, 0.8, 1])\n    >>> R = cv2.Rodrigues(np.array([0.3, 0, 0]))[0]\n    >>> H = K.dot(R).dot(np.linalg.inv(K))\n    >>> f = focal_from_homography(3 * H)\n    >>> np.allclose(f, 0.8)\n    True\n    '
    H = H / np.linalg.det(H) ** (1.0 / 3.0)
    A = np.array([[H[0, 0] * H[0, 0] + H[0, 1] * H[0, 1] - 1, H[0, 2] * H[0, 2]], [H[0, 0] * H[1, 0] + H[0, 1] * H[1, 1], H[0, 2] * H[1, 2]], [H[0, 0] * H[2, 0] + H[0, 1] * H[2, 1], H[0, 2] * H[2, 2]], [H[1, 0] * H[1, 0] + H[1, 1] * H[1, 1] - 1, H[1, 2] * H[1, 2]], [H[1, 0] * H[2, 0] + H[1, 1] * H[2, 1], H[1, 2] * H[2, 2]], [H[2, 0] * H[2, 0] + H[2, 1] * H[2, 1], H[2, 2] * H[2, 2] - 1]])
    (_, (a, b)) = nullspace(A)
    focal = np.sqrt(a / b)
    return focal

def R_from_homography(H: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> Optional[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    K1 = np.diag([f1, f1, 1])
    K2 = np.diag([f2, f2, 1])
    K2inv = np.linalg.inv(K2)
    R = K2inv.dot(H).dot(K1)
    R = project_to_rotation_matrix(R)
    return R

def project_to_rotation_matrix(A: np.ndarray) -> Optional[np.ndarray]:
    if False:
        return 10
    try:
        (u, d, vt) = np.linalg.svd(A)
    except np.linalg.linalg.LinAlgError:
        return None
    return u.dot(vt)

def camera_up_vector(rotation_matrix: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Unit vector pointing to zenit in camera coords.\n\n    :param rotation: camera pose rotation\n    '
    return rotation_matrix[:, 2]

def camera_compass_angle(rotation_matrix: np.ndarray) -> float:
    if False:
        i = 10
        return i + 15
    "Compass angle of a camera\n\n    Angle between world's Y axis and camera's Z axis projected\n    onto the XY world plane.\n\n    :param rotation: camera pose rotation\n    "
    z = rotation_matrix[2, :]
    angle = np.arctan2(z[0], z[1])
    return np.degrees(angle)

def rotation_matrix_from_up_vector_and_compass(up_vector: List[float], compass_angle: float) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Camera rotation given up_vector and compass.\n\n    >>> d = [1, 2, 3]\n    >>> angle = -123\n    >>> R = rotation_matrix_from_up_vector_and_compass(d, angle)\n    >>> np.allclose(np.linalg.det(R), 1.0)\n    True\n    >>> up = camera_up_vector(R)\n    >>> np.allclose(d / np.linalg.norm(d), up)\n    True\n    >>> np.allclose(camera_compass_angle(R), angle)\n    True\n\n    >>> d = [0, 0, 1]\n    >>> angle = 123\n    >>> R = rotation_matrix_from_up_vector_and_compass(d, angle)\n    >>> np.allclose(np.linalg.det(R), 1.0)\n    True\n    >>> up = camera_up_vector(R)\n    >>> np.allclose(d / np.linalg.norm(d), up)\n    True\n    '
    r3 = np.array(up_vector) / np.linalg.norm(up_vector)
    ez = np.array([0.0, 0.0, 1.0])
    r2 = ez - np.dot(ez, r3) * r3
    r2n = np.linalg.norm(r2)
    if r2n > 1e-08:
        r2 /= r2n
        r1 = np.cross(r2, r3)
    else:
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.cross(r3, r1)
    compass_rotation = cv2.Rodrigues(np.radians([0.0, 0.0, compass_angle]))[0]
    return np.column_stack([r1, r2, r3]).dot(compass_rotation)

def motion_from_plane_homography(H: np.ndarray) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    if False:
        print('Hello World!')
    'Compute candidate camera motions from a plane-induced homography.\n\n    Returns up to 8 motions.\n    The homography is assumed to be in normalized camera coordinates.\n\n    Uses the method of [Faugueras and Lustman 1988]\n\n    [Faugueras and Lustman 1988] Faugeras, Olivier, and F. Lustman.\n    “Motion and Structure from Motion in a Piecewise Planar Environment.”\n    Report. INRIA, June 1988. https://hal.inria.fr/inria-00075698/document\n    '
    try:
        (u, l, vh) = np.linalg.svd(H)
    except ValueError:
        return None
    (d1, d2, d3) = l
    s = np.linalg.det(u) * np.linalg.det(vh)
    if d1 / d2 < 1.0001 or d2 / d3 < 1.0001:
        return None
    abs_x1 = np.sqrt((d1 ** 2 - d2 ** 2) / (d1 ** 2 - d3 ** 2))
    abs_x3 = np.sqrt((d2 ** 2 - d3 ** 2) / (d1 ** 2 - d3 ** 2))
    possible_x1_x3 = [(abs_x1, abs_x3), (abs_x1, -abs_x3), (-abs_x1, abs_x3), (-abs_x1, -abs_x3)]
    solutions = []
    for (x1, x3) in possible_x1_x3:
        sin_term = x1 * x3 / d2
        sin_theta = (d1 - d3) * sin_term
        sin_phi = (d1 + d3) * sin_term
        d1_x3_2 = d1 * x3 ** 2
        d3_x1_2 = d3 * x1 ** 2
        cos_theta = (d3_x1_2 + d1_x3_2) / d2
        cos_phi = (d3_x1_2 - d1_x3_2) / d2
        Rp_p = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
        Rp_n = np.array([[cos_phi, 0, sin_phi], [0, -1, 0], [sin_phi, 0, -cos_phi]])
        np_ = np.array([x1, 0, x3])
        tp_p = (d1 - d3) * np.array([x1, 0, -x3])
        tp_n = (d1 + d3) * np_
        R_p = s * np.dot(np.dot(u, Rp_p), vh)
        R_n = s * np.dot(np.dot(u, Rp_n), vh)
        t_p = np.dot(u, tp_p)
        t_n = np.dot(u, tp_n)
        n = -np.dot(vh.T, np_)
        d = s * d2
        solutions.append((R_p, t_p, n, d))
        solutions.append((R_n, t_n, n, -d))
    return solutions

def absolute_pose_known_rotation_ransac(bs: np.ndarray, Xs: np.ndarray, threshold: float, iterations: int, probability: float) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    params = pyrobust.RobustEstimatorParams()
    params.iterations = iterations
    result = pyrobust.ransac_absolute_pose_known_rotation(bs, Xs, threshold, params, pyrobust.RansacType.RANSAC)
    t = -result.lo_model.copy()
    R = np.identity(3)
    return np.concatenate((R, [[t[0]], [t[1]], [t[2]]]), axis=1)

def absolute_pose_ransac(bs: np.ndarray, Xs: np.ndarray, threshold: float, iterations: int, probability: float) -> np.ndarray:
    if False:
        print('Hello World!')
    params = pyrobust.RobustEstimatorParams()
    params.iterations = iterations
    result = pyrobust.ransac_absolute_pose(bs, Xs, threshold, params, pyrobust.RansacType.RANSAC)
    Rt = result.lo_model.copy()
    (R, t) = (Rt[:3, :3].copy(), Rt[:, 3].copy())
    Rt[:3, :3] = R.T
    Rt[:, 3] = -R.T.dot(t)
    return Rt

def relative_pose_ransac(b1: np.ndarray, b2: np.ndarray, threshold: float, iterations: int, probability: float) -> np.ndarray:
    if False:
        return 10
    params = pyrobust.RobustEstimatorParams()
    params.iterations = iterations
    result = pyrobust.ransac_relative_pose(b1, b2, threshold, params, pyrobust.RansacType.RANSAC)
    Rt = result.lo_model.copy()
    (R, t) = (Rt[:3, :3].copy(), Rt[:, 3].copy())
    Rt[:3, :3] = R.T
    Rt[:, 3] = -R.T.dot(t)
    return Rt

def relative_pose_ransac_rotation_only(b1: np.ndarray, b2: np.ndarray, threshold: float, iterations: int, probability: float) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    params = pyrobust.RobustEstimatorParams()
    params.iterations = iterations
    result = pyrobust.ransac_relative_rotation(b1, b2, threshold, params, pyrobust.RansacType.RANSAC)
    return result.lo_model.T

def relative_pose_optimize_nonlinear(b1: np.ndarray, b2: np.ndarray, t: np.ndarray, R: np.ndarray, iterations: int) -> np.ndarray:
    if False:
        print('Hello World!')
    Rt = np.zeros((3, 4))
    Rt[:3, :3] = R.T
    Rt[:, 3] = -R.T.dot(t)
    Rt_refined = pygeometry.relative_pose_refinement(Rt, b1, b2, iterations)
    (R, t) = (Rt_refined[:3, :3].copy(), Rt_refined[:, 3].copy())
    Rt[:3, :3] = R.T
    Rt[:, 3] = -R.T.dot(t)
    return Rt

def triangulate_gcp(point: pymap.GroundControlPoint, shots: Dict[str, pymap.Shot], reproj_threshold: float=0.02, min_ray_angle_degrees: float=1.0) -> Optional[np.ndarray]:
    if False:
        print('Hello World!')
    'Compute the reconstructed position of a GCP from observations.'
    (os, bs, ids) = ([], [], [])
    for observation in point.observations:
        shot_id = observation.shot_id
        if shot_id in shots:
            shot = shots[shot_id]
            os.append(shot.pose.get_origin())
            x = observation.projection
            b = shot.camera.pixel_bearing(np.array(x))
            r = shot.pose.get_rotation_matrix().T
            bs.append(r.dot(b))
            ids.append(shot_id)
    if len(os) >= 2:
        thresholds = len(os) * [reproj_threshold]
        (valid_triangulation, X) = pygeometry.triangulate_bearings_midpoint(np.asarray(os), np.asarray(bs), thresholds, np.radians(min_ray_angle_degrees), np.radians(180.0 - min_ray_angle_degrees))
        if valid_triangulation:
            return X
    return None