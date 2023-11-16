from __future__ import annotations
from functools import reduce
import math
import operator as op
import platform
from mapbox_earcut import triangulate_float32 as earcut
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm as ProgressDisplay
from manimlib.constants import DOWN, OUT, RIGHT, UP
from manimlib.constants import PI, TAU
from manimlib.utils.iterables import adjacent_pairs
from manimlib.utils.simple_functions import clip
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence, List, Tuple
    from manimlib.typing import Vect2, Vect3, Vect4, VectN, Matrix3x3, Vect3Array, Vect2Array

def cross(v1: Vect3 | List[float], v2: Vect3 | List[float], out: np.ndarray | None=None) -> Vect3 | Vect3Array:
    if False:
        return 10
    is2d = isinstance(v1, np.ndarray) and len(v1.shape) == 2
    if is2d:
        (x1, y1, z1) = (v1[:, 0], v1[:, 1], v1[:, 2])
        (x2, y2, z2) = (v2[:, 0], v2[:, 1], v2[:, 2])
    else:
        (x1, y1, z1) = v1
        (x2, y2, z2) = v2
    if out is None:
        out = np.empty(np.shape(v1))
    out.T[:] = [y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2]
    return out

def get_norm(vect: VectN | List[float]) -> float:
    if False:
        print('Hello World!')
    return sum((x ** 2 for x in vect)) ** 0.5

def normalize(vect: VectN | List[float], fall_back: VectN | List[float] | None=None) -> VectN:
    if False:
        i = 10
        return i + 15
    norm = get_norm(vect)
    if norm > 0:
        return np.array(vect) / norm
    elif fall_back is not None:
        return np.array(fall_back)
    else:
        return np.zeros(len(vect))

def quaternion_mult(*quats: Vect4) -> Vect4:
    if False:
        print('Hello World!')
    '\n    Inputs are treated as quaternions, where the real part is the\n    last entry, so as to follow the scipy Rotation conventions.\n    '
    if len(quats) == 0:
        return np.array([0, 0, 0, 1])
    result = np.array(quats[0])
    for next_quat in quats[1:]:
        (x1, y1, z1, w1) = result
        (x2, y2, z2, w2) = next_quat
        result[:] = [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2, w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]
    return result

def quaternion_from_angle_axis(angle: float, axis: Vect3) -> Vect4:
    if False:
        for i in range(10):
            print('nop')
    return Rotation.from_rotvec(angle * normalize(axis)).as_quat()

def angle_axis_from_quaternion(quat: Vect4) -> Tuple[float, Vect3]:
    if False:
        while True:
            i = 10
    rot_vec = Rotation.from_quat(quat).as_rotvec()
    norm = get_norm(rot_vec)
    return (norm, rot_vec / norm)

def quaternion_conjugate(quaternion: Vect4) -> Vect4:
    if False:
        return 10
    result = np.array(quaternion)
    result[:3] *= -1
    return result

def rotate_vector(vector: Vect3, angle: float, axis: Vect3=OUT) -> Vect3:
    if False:
        while True:
            i = 10
    rot = Rotation.from_rotvec(angle * normalize(axis))
    return np.dot(vector, rot.as_matrix().T)

def rotate_vector_2d(vector: Vect2, angle: float) -> Vect2:
    if False:
        print('Hello World!')
    z = complex(*vector) * np.exp(complex(0, angle))
    return np.array([z.real, z.imag])

def rotation_matrix_transpose_from_quaternion(quat: Vect4) -> Matrix3x3:
    if False:
        print('Hello World!')
    return Rotation.from_quat(quat).as_matrix()

def rotation_matrix_from_quaternion(quat: Vect4) -> Matrix3x3:
    if False:
        i = 10
        return i + 15
    return np.transpose(rotation_matrix_transpose_from_quaternion(quat))

def rotation_matrix(angle: float, axis: Vect3) -> Matrix3x3:
    if False:
        i = 10
        return i + 15
    '\n    Rotation in R^3 about a specified axis of rotation.\n    '
    return Rotation.from_rotvec(angle * normalize(axis)).as_matrix()

def rotation_matrix_transpose(angle: float, axis: Vect3) -> Matrix3x3:
    if False:
        print('Hello World!')
    return rotation_matrix(angle, axis).T

def rotation_about_z(angle: float) -> Matrix3x3:
    if False:
        for i in range(10):
            print('nop')
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

def rotation_between_vectors(v1: Vect3, v2: Vect3) -> Matrix3x3:
    if False:
        for i in range(10):
            print('nop')
    atol = 1e-08
    if get_norm(v1 - v2) < atol:
        return np.identity(3)
    axis = cross(v1, v2)
    if get_norm(axis) < atol:
        axis = cross(v1, RIGHT)
    if get_norm(axis) < atol:
        axis = cross(v1, UP)
    return rotation_matrix(angle=angle_between_vectors(v1, v2), axis=axis)

def z_to_vector(vector: Vect3) -> Matrix3x3:
    if False:
        while True:
            i = 10
    return rotation_between_vectors(OUT, vector)

def angle_of_vector(vector: Vect2 | Vect3) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns polar coordinate theta when vector is project on xy plane\n    '
    return math.atan2(vector[1], vector[0])

def angle_between_vectors(v1: VectN, v2: VectN) -> float:
    if False:
        while True:
            i = 10
    '\n    Returns the angle between two 3D vectors.\n    This angle will always be btw 0 and pi\n    '
    n1 = get_norm(v1)
    n2 = get_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    cos_angle = np.dot(v1, v2) / np.float64(n1 * n2)
    return math.acos(clip(cos_angle, -1, 1))

def project_along_vector(point: Vect3, vector: Vect3) -> Vect3:
    if False:
        print('Hello World!')
    matrix = np.identity(3) - np.outer(vector, vector)
    return np.dot(point, matrix.T)

def normalize_along_axis(array: np.ndarray, axis: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    norms = np.sqrt((array * array).sum(axis))
    norms[norms == 0] = 1
    return (array.T / norms).T

def get_unit_normal(v1: Vect3, v2: Vect3, tol: float=1e-06) -> Vect3:
    if False:
        for i in range(10):
            print('nop')
    v1 = normalize(v1)
    v2 = normalize(v2)
    cp = cross(v1, v2)
    cp_norm = get_norm(cp)
    if cp_norm < tol:
        new_cp = cross(cross(v1, OUT), v1)
        new_cp_norm = get_norm(new_cp)
        if new_cp_norm < tol:
            return DOWN
        return new_cp / new_cp_norm
    return cp / cp_norm

def thick_diagonal(dim: int, thickness: int=2) -> np.ndarray:
    if False:
        print('Hello World!')
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    col_indices = np.transpose(row_indices)
    return (np.abs(row_indices - col_indices) < thickness).astype('uint8')

def compass_directions(n: int=4, start_vect: Vect3=RIGHT) -> Vect3:
    if False:
        for i in range(10):
            print('nop')
    angle = TAU / n
    return np.array([rotate_vector(start_vect, k * angle) for k in range(n)])

def complex_to_R3(complex_num: complex) -> Vect3:
    if False:
        for i in range(10):
            print('nop')
    return np.array((complex_num.real, complex_num.imag, 0))

def R3_to_complex(point: Vect3) -> complex:
    if False:
        print('Hello World!')
    return complex(*point[:2])

def complex_func_to_R3_func(complex_func: Callable[[complex], complex]) -> Callable[[Vect3], Vect3]:
    if False:
        print('Hello World!')

    def result(p: Vect3):
        if False:
            i = 10
            return i + 15
        return complex_to_R3(complex_func(R3_to_complex(p)))
    return result

def center_of_mass(points: Sequence[Vect3]) -> Vect3:
    if False:
        while True:
            i = 10
    return np.array(points).sum(0) / len(points)

def midpoint(point1: VectN, point2: VectN) -> VectN:
    if False:
        return 10
    return center_of_mass([point1, point2])

def line_intersection(line1: Tuple[Vect3, Vect3], line2: Tuple[Vect3, Vect3]) -> Vect3:
    if False:
        while True:
            i = 10
    '\n    return intersection point of two lines,\n    each defined with a pair of vectors determining\n    the end points\n    '
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        if False:
            i = 10
            return i + 15
        return a[0] * b[1] - a[1] * b[0]
    div = det(x_diff, y_diff)
    if div == 0:
        raise Exception('Lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return np.array([x, y, 0])

def find_intersection(p0: Vect3 | Vect3Array, v0: Vect3 | Vect3Array, p1: Vect3 | Vect3Array, v1: Vect3 | Vect3Array, threshold: float=1e-05) -> Vect3:
    if False:
        i = 10
        return i + 15
    '\n    Return the intersection of a line passing through p0 in direction v0\n    with one passing through p1 in direction v1.  (Or array of intersections\n    from arrays of such points/directions).\n\n    For 3d values, it returns the point on the ray p0 + v0 * t closest to the\n    ray p1 + v1 * t\n    '
    d = len(p0.shape)
    if d == 1:
        is_3d = any((arr[2] for arr in (p0, v0, p1, v1)))
    else:
        is_3d = any((z for arr in (p0, v0, p1, v1) for z in arr.T[2]))
    if not is_3d:
        numer = np.array(cross2d(v1, p1 - p0))
        denom = np.array(cross2d(v1, v0))
    else:
        cp1 = cross(v1, p1 - p0)
        cp2 = cross(v1, v0)
        numer = np.array((cp1 * cp1).sum(d - 1))
        denom = np.array((cp1 * cp2).sum(d - 1))
    denom[abs(denom) < threshold] = np.inf
    ratio = numer / denom
    return p0 + (ratio * v0.T).T

def line_intersects_path(start: Vect2 | Vect3, end: Vect2 | Vect3, path: Vect2Array | Vect3Array) -> bool:
    if False:
        return 10
    '\n    Tests whether the line (start, end) intersects\n    a polygonal path defined by its vertices\n    '
    n = len(path) - 1
    p1 = np.empty((n, 2))
    q1 = np.empty((n, 2))
    p1[:] = start[:2]
    q1[:] = end[:2]
    p2 = path[:-1, :2]
    q2 = path[1:, :2]
    v1 = q1 - p1
    v2 = q2 - p2
    mis1 = cross2d(v1, p2 - p1) * cross2d(v1, q2 - p1) < 0
    mis2 = cross2d(v2, p1 - p2) * cross2d(v2, q1 - p2) < 0
    return bool((mis1 * mis2).any())

def get_closest_point_on_line(a: VectN, b: VectN, p: VectN) -> VectN:
    if False:
        i = 10
        return i + 15
    '\n        It returns point x such that\n        x is on line ab and xp is perpendicular to ab.\n        If x lies beyond ab line, then it returns nearest edge(a or b).\n    '
    t = np.dot(p - b, a - b) / np.dot(a - b, a - b)
    if t < 0:
        t = 0
    if t > 1:
        t = 1
    return t * a + (1 - t) * b

def get_winding_number(points: Sequence[Vect2 | Vect3]) -> float:
    if False:
        print('Hello World!')
    total_angle = 0
    for (p1, p2) in adjacent_pairs(points):
        d_angle = angle_of_vector(p2) - angle_of_vector(p1)
        d_angle = (d_angle + PI) % TAU - PI
        total_angle += d_angle
    return total_angle / TAU

def cross2d(a: Vect2 | Vect2Array, b: Vect2 | Vect2Array) -> Vect2 | Vect2Array:
    if False:
        print('Hello World!')
    if len(a.shape) == 2:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    else:
        return a[0] * b[1] - b[0] * a[1]

def tri_area(a: Vect2, b: Vect2, c: Vect2) -> float:
    if False:
        print('Hello World!')
    return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))

def is_inside_triangle(p: Vect2, a: Vect2, b: Vect2, c: Vect2) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if point p is inside triangle abc\n    '
    crosses = np.array([cross2d(p - a, b - p), cross2d(p - b, c - p), cross2d(p - c, a - p)])
    return bool(np.all(crosses > 0) or np.all(crosses < 0))

def norm_squared(v: VectN | List[float]) -> float:
    if False:
        for i in range(10):
            print('nop')
    return sum((x * x for x in v))

def earclip_triangulation(verts: Vect3Array | Vect2Array, ring_ends: list[int]) -> list[int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of indices giving a triangulation\n    of a polygon, potentially with holes\n\n    - verts is a numpy array of points\n\n    - ring_ends is a list of indices indicating where\n    the ends of new paths are\n    '
    rings = [list(range(e0, e1)) for (e0, e1) in zip([0, *ring_ends], ring_ends)]
    epsilon = 1e-06

    def is_in(point, ring_id):
        if False:
            i = 10
            return i + 15
        return abs(abs(get_winding_number([i - point for i in verts[rings[ring_id]]])) - 1) < epsilon

    def ring_area(ring_id):
        if False:
            return 10
        ring = rings[ring_id]
        s = 0
        for (i, j) in zip(ring[1:], ring):
            s += cross2d(verts[i], verts[j])
        return abs(s) / 2
    for i in rings:
        if len(i) < 2:
            continue
        verts[i[0]] += (verts[i[1]] - verts[i[0]]) * epsilon
        verts[i[-1]] += (verts[i[-2]] - verts[i[-1]]) * epsilon
    right = [max(verts[rings[i], 0]) for i in range(len(rings))]
    left = [min(verts[rings[i], 0]) for i in range(len(rings))]
    top = [max(verts[rings[i], 1]) for i in range(len(rings))]
    bottom = [min(verts[rings[i], 1]) for i in range(len(rings))]
    area = [ring_area(i) for i in range(len(rings))]
    rings_sorted = list(range(len(rings)))
    rings_sorted.sort(key=lambda x: area[x], reverse=True)

    def is_in_fast(ring_a, ring_b):
        if False:
            while True:
                i = 10
        return reduce(op.and_, (left[ring_b] <= left[ring_a] <= right[ring_a] <= right[ring_b], bottom[ring_b] <= bottom[ring_a] <= top[ring_a] <= top[ring_b], is_in(verts[rings[ring_a][0]], ring_b)))
    chilren = [[] for i in rings]
    ringenum = ProgressDisplay(enumerate(rings_sorted), total=len(rings), leave=False, ascii=True if platform.system() == 'Windows' else None, dynamic_ncols=True, desc='SVG Triangulation', delay=3)
    for (idx, i) in ringenum:
        for j in rings_sorted[:idx][::-1]:
            if is_in_fast(i, j):
                chilren[j].append(i)
                break
    res = []
    used = [False] * len(rings)
    for i in rings_sorted:
        if used[i]:
            continue
        v = rings[i]
        ring_ends = [len(v)]
        for j in chilren[i]:
            used[j] = True
            v += rings[j]
            ring_ends.append(len(v))
        res += [v[i] for i in earcut(verts[v, :2], ring_ends)]
    return res