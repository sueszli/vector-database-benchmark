"""Vector class, including rotation-related functions."""
import numpy as np
from typing import Tuple, Optional

def m2rotaxis(m):
    if False:
        for i in range(10):
            print('nop')
    'Return angles, axis pair that corresponds to rotation matrix m.\n\n    The case where ``m`` is the identity matrix corresponds to a singularity\n    where any rotation axis is valid. In that case, ``Vector([1, 0, 0])``,\n    is returned.\n    '
    eps = 1e-05
    if abs(m[0, 1] - m[1, 0]) < eps and abs(m[0, 2] - m[2, 0]) < eps and (abs(m[1, 2] - m[2, 1]) < eps):
        if abs(m[0, 1] + m[1, 0]) < eps and abs(m[0, 2] + m[2, 0]) < eps and (abs(m[1, 2] + m[2, 1]) < eps) and (abs(m[0, 0] + m[1, 1] + m[2, 2] - 3) < eps):
            angle = 0
        else:
            angle = np.pi
    else:
        t = 0.5 * (np.trace(m) - 1)
        t = max(-1, t)
        t = min(1, t)
        angle = np.arccos(t)
    if angle < 1e-15:
        return (0.0, Vector(1, 0, 0))
    elif angle < np.pi:
        x = m[2, 1] - m[1, 2]
        y = m[0, 2] - m[2, 0]
        z = m[1, 0] - m[0, 1]
        axis = Vector(x, y, z)
        axis.normalize()
        return (angle, axis)
    else:
        m00 = m[0, 0]
        m11 = m[1, 1]
        m22 = m[2, 2]
        if m00 > m11 and m00 > m22:
            x = np.sqrt(m00 - m11 - m22 + 0.5)
            y = m[0, 1] / (2 * x)
            z = m[0, 2] / (2 * x)
        elif m11 > m00 and m11 > m22:
            y = np.sqrt(m11 - m00 - m22 + 0.5)
            x = m[0, 1] / (2 * y)
            z = m[1, 2] / (2 * y)
        else:
            z = np.sqrt(m22 - m00 - m11 + 0.5)
            x = m[0, 2] / (2 * z)
            y = m[1, 2] / (2 * z)
        axis = Vector(x, y, z)
        axis.normalize()
        return (np.pi, axis)

def vector_to_axis(line, point):
    if False:
        return 10
    'Vector to axis method.\n\n    Return the vector between a point and\n    the closest point on a line (ie. the perpendicular\n    projection of the point on the line).\n\n    :type line: L{Vector}\n    :param line: vector defining a line\n\n    :type point: L{Vector}\n    :param point: vector defining the point\n    '
    line = line.normalized()
    np = point.norm()
    angle = line.angle(point)
    return point - line ** (np * np.cos(angle))

def rotaxis2m(theta, vector):
    if False:
        for i in range(10):
            print('nop')
    'Calculate left multiplying rotation matrix.\n\n    Calculate a left multiplying rotation matrix that rotates\n    theta rad around vector.\n\n    :type theta: float\n    :param theta: the rotation angle\n\n    :type vector: L{Vector}\n    :param vector: the rotation axis\n\n    :return: The rotation matrix, a 3x3 NumPy array.\n\n    Examples\n    --------\n    >>> from numpy import pi\n    >>> from Bio.PDB.vectors import rotaxis2m\n    >>> from Bio.PDB.vectors import Vector\n    >>> m = rotaxis2m(pi, Vector(1, 0, 0))\n    >>> Vector(1, 2, 3).left_multiply(m)\n    <Vector 1.00, -2.00, -3.00>\n\n    '
    vector = vector.normalized()
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    (x, y, z) = vector.get_array()
    rot = np.zeros((3, 3))
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y - s * z
    rot[0, 2] = t * x * z + s * y
    rot[1, 0] = t * x * y + s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z - s * x
    rot[2, 0] = t * x * z - s * y
    rot[2, 1] = t * y * z + s * x
    rot[2, 2] = t * z * z + c
    return rot
rotaxis = rotaxis2m

def refmat(p, q):
    if False:
        return 10
    'Return a (left multiplying) matrix that mirrors p onto q.\n\n    :type p,q: L{Vector}\n    :return: The mirror operation, a 3x3 NumPy array.\n\n    Examples\n    --------\n    >>> from Bio.PDB.vectors import refmat\n    >>> p, q = Vector(1, 2, 3), Vector(2, 3, 5)\n    >>> mirror = refmat(p, q)\n    >>> qq = p.left_multiply(mirror)\n    >>> print(q)\n    <Vector 2.00, 3.00, 5.00>\n    >>> print(qq)\n    <Vector 1.21, 1.82, 3.03>\n\n    '
    p = p.normalized()
    q = q.normalized()
    if (p - q).norm() < 1e-05:
        return np.identity(3)
    pq = p - q
    pq.normalize()
    b = pq.get_array()
    b.shape = (3, 1)
    i = np.identity(3)
    ref = i - 2 * np.dot(b, np.transpose(b))
    return ref

def rotmat(p, q):
    if False:
        for i in range(10):
            print('nop')
    'Return a (left multiplying) matrix that rotates p onto q.\n\n    :param p: moving vector\n    :type p: L{Vector}\n\n    :param q: fixed vector\n    :type q: L{Vector}\n\n    :return: rotation matrix that rotates p onto q\n    :rtype: 3x3 NumPy array\n\n    Examples\n    --------\n    >>> from Bio.PDB.vectors import rotmat\n    >>> p, q = Vector(1, 2, 3), Vector(2, 3, 5)\n    >>> r = rotmat(p, q)\n    >>> print(q)\n    <Vector 2.00, 3.00, 5.00>\n    >>> print(p)\n    <Vector 1.00, 2.00, 3.00>\n    >>> p.left_multiply(r)\n    <Vector 1.21, 1.82, 3.03>\n\n    '
    rot = np.dot(refmat(q, -p), refmat(p, -p))
    return rot

def calc_angle(v1, v2, v3):
    if False:
        i = 10
        return i + 15
    'Calculate angle method.\n\n    Calculate the angle between 3 vectors\n    representing 3 connected points.\n\n    :param v1, v2, v3: the tree points that define the angle\n    :type v1, v2, v3: L{Vector}\n\n    :return: angle\n    :rtype: float\n    '
    v1 = v1 - v2
    v3 = v3 - v2
    return v1.angle(v3)

def calc_dihedral(v1, v2, v3, v4):
    if False:
        print('Hello World!')
    'Calculate dihedral angle method.\n\n    Calculate the dihedral angle between 4 vectors\n    representing 4 connected points. The angle is in\n    ]-pi, pi].\n\n    :param v1, v2, v3, v4: the four points that define the dihedral angle\n    :type v1, v2, v3, v4: L{Vector}\n    '
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = ab ** cb
    v = db ** cb
    w = u ** v
    angle = u.angle(v)
    try:
        if cb.angle(w) > 0.001:
            angle = -angle
    except ZeroDivisionError:
        pass
    return angle

class Vector:
    """3D vector."""

    def __init__(self, x, y=None, z=None):
        if False:
            return 10
        'Initialize the class.'
        if y is None and z is None:
            if len(x) != 3:
                raise ValueError('Vector: x is not a list/tuple/array of 3 numbers')
            self._ar = np.array(x, 'd')
        else:
            self._ar = np.array((x, y, z), 'd')

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return vector 3D coordinates.'
        (x, y, z) = self._ar
        return f'<Vector {x:.2f}, {y:.2f}, {z:.2f}>'

    def __neg__(self):
        if False:
            while True:
                i = 10
        'Return Vector(-x, -y, -z).'
        a = -self._ar
        return Vector(a)

    def __add__(self, other):
        if False:
            return 10
        'Return Vector+other Vector or scalar.'
        if isinstance(other, Vector):
            a = self._ar + other._ar
        else:
            a = self._ar + np.array(other)
        return Vector(a)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return Vector-other Vector or scalar.'
        if isinstance(other, Vector):
            a = self._ar - other._ar
        else:
            a = self._ar - np.array(other)
        return Vector(a)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        'Return Vector.Vector (dot product).'
        return sum(self._ar * other._ar)

    def __truediv__(self, x):
        if False:
            print('Hello World!')
        'Return Vector(coords/a).'
        a = self._ar / np.array(x)
        return Vector(a)

    def __pow__(self, other):
        if False:
            i = 10
            return i + 15
        'Return VectorxVector (cross product) or Vectorxscalar.'
        if isinstance(other, Vector):
            (a, b, c) = self._ar
            (d, e, f) = other._ar
            c1 = np.linalg.det(np.array(((b, c), (e, f))))
            c2 = -np.linalg.det(np.array(((a, c), (d, f))))
            c3 = np.linalg.det(np.array(((a, b), (d, e))))
            return Vector(c1, c2, c3)
        else:
            a = self._ar * np.array(other)
            return Vector(a)

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Return value of array index i.'
        return self._ar[i]

    def __setitem__(self, i, value):
        if False:
            while True:
                i = 10
        'Assign values to array index i.'
        self._ar[i] = value

    def __contains__(self, i):
        if False:
            return 10
        'Validate if i is in array.'
        return i in self._ar

    def norm(self):
        if False:
            return 10
        'Return vector norm.'
        return np.sqrt(sum(self._ar * self._ar))

    def normsq(self):
        if False:
            i = 10
            return i + 15
        'Return square of vector norm.'
        return abs(sum(self._ar * self._ar))

    def normalize(self):
        if False:
            print('Hello World!')
        "Normalize the Vector object.\n\n        Changes the state of ``self`` and doesn't return a value.\n        If you need to chain function calls or create a new object\n        use the ``normalized`` method.\n        "
        if self.norm():
            self._ar = self._ar / self.norm()

    def normalized(self):
        if False:
            i = 10
            return i + 15
        'Return a normalized copy of the Vector.\n\n        To avoid allocating new objects use the ``normalize`` method.\n        '
        v = self.copy()
        v.normalize()
        return v

    def angle(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return angle between two vectors.'
        n1 = self.norm()
        n2 = other.norm()
        c = self * other / (n1 * n2)
        c = min(c, 1)
        c = max(-1, c)
        return np.arccos(c)

    def get_array(self):
        if False:
            for i in range(10):
                print('nop')
        'Return (a copy of) the array of coordinates.'
        return np.array(self._ar)

    def left_multiply(self, matrix):
        if False:
            i = 10
            return i + 15
        'Return Vector=Matrix x Vector.'
        a = np.dot(matrix, self._ar)
        return Vector(a)

    def right_multiply(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        'Return Vector=Vector x Matrix.'
        a = np.dot(self._ar, matrix)
        return Vector(a)

    def copy(self):
        if False:
            while True:
                i = 10
        'Return a deep copy of the Vector.'
        return Vector(self._ar)
'Homogeneous matrix geometry routines.\n\nRotation, translation, scale, and coordinate transformations.\n\nRobert T. Miller 2019\n'

def homog_rot_mtx(angle_rads: float, axis: str) -> np.array:
    if False:
        i = 10
        return i + 15
    'Generate a 4x4 single-axis NumPy rotation matrix.\n\n    :param float angle_rads: the desired rotation angle in radians\n    :param char axis: character specifying the rotation axis\n    '
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    if 'z' == axis:
        return np.array(((cosang, -sinang, 0, 0), (sinang, cosang, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64)
    elif 'y' == axis:
        return np.array(((cosang, 0, sinang, 0), (0, 1, 0, 0), (-sinang, 0, cosang, 0), (0, 0, 0, 1)), dtype=np.float64)
    else:
        return np.array(((1, 0, 0, 0), (0, cosang, -sinang, 0), (0, sinang, cosang, 0), (0, 0, 0, 1)), dtype=np.float64)

def set_Z_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    if False:
        i = 10
        return i + 15
    'Update existing Z rotation matrix to new angle.'
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[0][0] = mtx[1][1] = cosang
    mtx[1][0] = sinang
    mtx[0][1] = -sinang

def set_Y_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    if False:
        for i in range(10):
            print('nop')
    'Update existing Y rotation matrix to new angle.'
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[0][0] = mtx[2][2] = cosang
    mtx[0][2] = sinang
    mtx[2][0] = -sinang

def set_X_homog_rot_mtx(angle_rads: float, mtx: np.ndarray):
    if False:
        print('Hello World!')
    'Update existing X rotation matrix to new angle.'
    cosang = np.cos(angle_rads)
    sinang = np.sin(angle_rads)
    mtx[1][1] = mtx[2][2] = cosang
    mtx[2][1] = sinang
    mtx[1][2] = -sinang

def homog_trans_mtx(x: float, y: float, z: float) -> np.array:
    if False:
        for i in range(10):
            print('nop')
    'Generate a 4x4 NumPy translation matrix.\n\n    :param x, y, z: translation in each axis\n    '
    return np.array(((1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1)), dtype=np.float64)

def set_homog_trans_mtx(x: float, y: float, z: float, mtx: np.ndarray):
    if False:
        return 10
    'Update existing translation matrix to new values.'
    mtx[0][3] = x
    mtx[1][3] = y
    mtx[2][3] = z

def homog_scale_mtx(scale: float) -> np.array:
    if False:
        i = 10
        return i + 15
    'Generate a 4x4 NumPy scaling matrix.\n\n    :param float scale: scale multiplier\n    '
    return np.array([[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]], dtype=np.float64)

def _get_azimuth(x: float, y: float) -> float:
    if False:
        i = 10
        return i + 15
    sign_y = -1.0 if y < 0.0 else 1.0
    sign_x = -1.0 if x < 0.0 else 1.0
    return np.arctan2(y, x) if 0 != x and 0 != y else np.pi / 2.0 * sign_y if 0 != y else np.pi if sign_x < 0.0 else 0.0

def get_spherical_coordinates(xyz: np.array) -> Tuple[float, float, float]:
    if False:
        i = 10
        return i + 15
    'Compute spherical coordinates (r, azimuth, polar_angle) for X,Y,Z point.\n\n    :param array xyz: column vector (3 row x 1 column NumPy array)\n    :return: tuple of r, azimuth, polar_angle for input coordinate\n    '
    r = np.linalg.norm(xyz)
    if 0 == r:
        return (0, 0, 0)
    azimuth = _get_azimuth(xyz[0], xyz[1])
    polar_angle = np.arccos(xyz[2] / r)
    return (r, azimuth, polar_angle)
gtm = np.identity(4, dtype=np.float64)
gmrz = np.identity(4, dtype=np.float64)
gmry = np.identity(4, dtype=np.float64)
gmrz2 = np.identity(4, dtype=np.float64)

def coord_space(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, rev: bool=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if False:
        print('Hello World!')
    'Generate transformation matrix to coordinate space defined by 3 points.\n\n    New coordinate space will have:\n        acs[0] on XZ plane\n        acs[1] origin\n        acs[2] on +Z axis\n\n    :param NumPy column array x3 acs: X,Y,Z column input coordinates x3\n    :param bool rev: if True, also return reverse transformation matrix\n        (to return from coord_space)\n    :returns: 4x4 NumPy array, x2 if rev=True\n    '
    global gtm
    global gmry
    global gmrz, gmrz2
    tm = gtm
    mry = gmry
    mrz = gmrz
    mrz2 = gmrz2
    set_homog_trans_mtx(-a1[0], -a1[1], -a1[2], tm)
    p = a2 - a1
    sc = get_spherical_coordinates(p)
    set_Z_homog_rot_mtx(-sc[1], mrz)
    set_Y_homog_rot_mtx(-sc[2], mry)
    mt = gmry.dot(gmrz.dot(gtm))
    p = mt.dot(a0)
    azimuth2 = _get_azimuth(p[0], p[1])
    set_Z_homog_rot_mtx(-azimuth2, mrz2)
    mt = gmrz2.dot(mt)
    if not rev:
        return (mt, None)
    set_Z_homog_rot_mtx(azimuth2, mrz2)
    set_Y_homog_rot_mtx(sc[2], mry)
    set_Z_homog_rot_mtx(sc[1], mrz)
    set_homog_trans_mtx(a1[0], a1[1], a1[2], tm)
    mr = gtm.dot(gmrz.dot(gmry.dot(gmrz2)))
    return (mt, mr)

def multi_rot_Z(angle_rads: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Create [entries] NumPy Z rotation matrices for [entries] angles.\n\n    :param entries: int number of matrices generated.\n    :param angle_rads: NumPy array of angles\n    :returns: entries x 4 x 4 homogeneous rotation matrices\n    '
    rz = np.empty((angle_rads.shape[0], 4, 4))
    rz[...] = np.identity(4)
    rz[:, 0, 0] = rz[:, 1, 1] = np.cos(angle_rads)
    rz[:, 1, 0] = np.sin(angle_rads)
    rz[:, 0, 1] = -rz[:, 1, 0]
    return rz

def multi_rot_Y(angle_rads: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Create [entries] NumPy Y rotation matrices for [entries] angles.\n\n    :param entries: int number of matrices generated.\n    :param angle_rads: NumPy array of angles\n    :returns: entries x 4 x 4 homogeneous rotation matrices\n    '
    ry = np.empty((angle_rads.shape[0], 4, 4))
    ry[...] = np.identity(4)
    ry[:, 0, 0] = ry[:, 2, 2] = np.cos(angle_rads)
    ry[:, 0, 2] = np.sin(angle_rads)
    ry[:, 2, 0] = -ry[:, 0, 2]
    return ry

def multi_coord_space(a3: np.ndarray, dLen: int, rev: bool=False) -> np.ndarray:
    if False:
        print('Hello World!')
    'Generate [dLen] transform matrices to coord space defined by 3 points.\n\n    New coordinate space will have:\n        acs[0] on XZ plane\n        acs[1] origin\n        acs[2] on +Z axis\n\n    :param NumPy array [entries]x3x3 [entries] XYZ coords for 3 atoms\n    :param bool rev: if True, also return reverse transformation matrix\n    (to return from coord_space)\n    :returns: [entries] 4x4 NumPy arrays, x2 if rev=True\n\n    '
    tm = np.empty((dLen, 4, 4))
    tm[...] = np.identity(4)
    tm[:, 0:3, 3] = -a3[:, 1, 0:3]
    p = a3[:, 2] - a3[:, 1]
    r = np.linalg.norm(p, axis=1)
    azimuth = np.arctan2(p[:, 1], p[:, 0])
    polar_angle = np.arccos(np.divide(p[:, 2], r, where=r != 0))
    rz = multi_rot_Z(-azimuth)
    ry = multi_rot_Y(-polar_angle)
    mt = np.matmul(ry, np.matmul(rz, tm))
    p = np.matmul(mt, a3[:, 0].reshape(-1, 4, 1)).reshape(-1, 4)
    azimuth2 = np.arctan2(p[:, 1], p[:, 0])
    rz2 = multi_rot_Z(-azimuth2)
    if not rev:
        return np.matmul(rz2, mt[:])
    mt = np.matmul(rz2, mt[:])
    mrz2 = multi_rot_Z(azimuth2)
    mry = multi_rot_Y(polar_angle)
    mrz = multi_rot_Z(azimuth)
    tm[:, 0:3, 3] = a3[:, 1, 0:3]
    mr = tm @ mrz @ mry @ mrz2
    return np.array([mt, mr])