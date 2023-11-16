from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import eye, rot_axis1, rot_axis2, rot_axis3
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector

class Orienter(Basic):
    """
    Super-class for all orienter classes.
    """

    def rotation_matrix(self):
        if False:
            while True:
                i = 10
        '\n        The rotation matrix corresponding to this orienter\n        instance.\n        '
        return self._parent_orient

class AxisOrienter(Orienter):
    """
    Class to denote an axis orienter.
    """

    def __new__(cls, angle, axis):
        if False:
            return 10
        if not isinstance(axis, sympy.vector.Vector):
            raise TypeError('axis should be a Vector')
        angle = sympify(angle)
        obj = super().__new__(cls, angle, axis)
        obj._angle = angle
        obj._axis = axis
        return obj

    def __init__(self, angle, axis):
        if False:
            print('Hello World!')
        "\n        Axis rotation is a rotation about an arbitrary axis by\n        some angle. The angle is supplied as a SymPy expr scalar, and\n        the axis is supplied as a Vector.\n\n        Parameters\n        ==========\n\n        angle : Expr\n            The angle by which the new system is to be rotated\n\n        axis : Vector\n            The axis around which the rotation has to be performed\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> from sympy import symbols\n        >>> q1 = symbols('q1')\n        >>> N = CoordSys3D('N')\n        >>> from sympy.vector import AxisOrienter\n        >>> orienter = AxisOrienter(q1, N.i + 2 * N.j)\n        >>> B = N.orient_new('B', (orienter, ))\n\n        "
        pass

    @cacheit
    def rotation_matrix(self, system):
        if False:
            return 10
        '\n        The rotation matrix corresponding to this orienter\n        instance.\n\n        Parameters\n        ==========\n\n        system : CoordSys3D\n            The coordinate system wrt which the rotation matrix\n            is to be computed\n        '
        axis = sympy.vector.express(self.axis, system).normalize()
        axis = axis.to_matrix(system)
        theta = self.angle
        parent_orient = (eye(3) - axis * axis.T) * cos(theta) + Matrix([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) * sin(theta) + axis * axis.T
        parent_orient = parent_orient.T
        return parent_orient

    @property
    def angle(self):
        if False:
            while True:
                i = 10
        return self._angle

    @property
    def axis(self):
        if False:
            for i in range(10):
                print('nop')
        return self._axis

class ThreeAngleOrienter(Orienter):
    """
    Super-class for Body and Space orienters.
    """

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if False:
            print('Hello World!')
        if isinstance(rot_order, Str):
            rot_order = rot_order.name
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
        original_rot_order = rot_order
        rot_order = str(rot_order).upper()
        if not len(rot_order) == 3:
            raise TypeError('rot_order should be a str of length 3')
        rot_order = [i.replace('X', '1') for i in rot_order]
        rot_order = [i.replace('Y', '2') for i in rot_order]
        rot_order = [i.replace('Z', '3') for i in rot_order]
        rot_order = ''.join(rot_order)
        if rot_order not in approved_orders:
            raise TypeError('Invalid rot_type parameter')
        a1 = int(rot_order[0])
        a2 = int(rot_order[1])
        a3 = int(rot_order[2])
        angle1 = sympify(angle1)
        angle2 = sympify(angle2)
        angle3 = sympify(angle3)
        if cls._in_order:
            parent_orient = _rot(a1, angle1) * _rot(a2, angle2) * _rot(a3, angle3)
        else:
            parent_orient = _rot(a3, angle3) * _rot(a2, angle2) * _rot(a1, angle1)
        parent_orient = parent_orient.T
        obj = super().__new__(cls, angle1, angle2, angle3, Str(rot_order))
        obj._angle1 = angle1
        obj._angle2 = angle2
        obj._angle3 = angle3
        obj._rot_order = original_rot_order
        obj._parent_orient = parent_orient
        return obj

    @property
    def angle1(self):
        if False:
            i = 10
            return i + 15
        return self._angle1

    @property
    def angle2(self):
        if False:
            for i in range(10):
                print('nop')
        return self._angle2

    @property
    def angle3(self):
        if False:
            while True:
                i = 10
        return self._angle3

    @property
    def rot_order(self):
        if False:
            print('Hello World!')
        return self._rot_order

class BodyOrienter(ThreeAngleOrienter):
    """
    Class to denote a body-orienter.
    """
    _in_order = True

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if False:
            while True:
                i = 10
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3, rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        if False:
            return 10
        "\n        Body orientation takes this coordinate system through three\n        successive simple rotations.\n\n        Body fixed rotations include both Euler Angles and\n        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.\n\n        Parameters\n        ==========\n\n        angle1, angle2, angle3 : Expr\n            Three successive angles to rotate the coordinate system by\n\n        rotation_order : string\n            String defining the order of axes for rotation\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, BodyOrienter\n        >>> from sympy import symbols\n        >>> q1, q2, q3 = symbols('q1 q2 q3')\n        >>> N = CoordSys3D('N')\n\n        A 'Body' fixed rotation is described by three angles and\n        three body-fixed rotation axes. To orient a coordinate system D\n        with respect to N, each sequential rotation is always about\n        the orthogonal unit vectors fixed to D. For example, a '123'\n        rotation will specify rotations about N.i, then D.j, then\n        D.k. (Initially, D.i is same as N.i)\n        Therefore,\n\n        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')\n        >>> D = N.orient_new('D', (body_orienter, ))\n\n        is same as\n\n        >>> from sympy.vector import AxisOrienter\n        >>> axis_orienter1 = AxisOrienter(q1, N.i)\n        >>> D = N.orient_new('D', (axis_orienter1, ))\n        >>> axis_orienter2 = AxisOrienter(q2, D.j)\n        >>> D = D.orient_new('D', (axis_orienter2, ))\n        >>> axis_orienter3 = AxisOrienter(q3, D.k)\n        >>> D = D.orient_new('D', (axis_orienter3, ))\n\n        Acceptable rotation orders are of length 3, expressed in XYZ or\n        123, and cannot have a rotation about about an axis twice in a row.\n\n        >>> body_orienter1 = BodyOrienter(q1, q2, q3, '123')\n        >>> body_orienter2 = BodyOrienter(q1, q2, 0, 'ZXZ')\n        >>> body_orienter3 = BodyOrienter(0, 0, 0, 'XYX')\n\n        "
        pass

class SpaceOrienter(ThreeAngleOrienter):
    """
    Class to denote a space-orienter.
    """
    _in_order = False

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if False:
            print('Hello World!')
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3, rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        if False:
            return 10
        "\n        Space rotation is similar to Body rotation, but the rotations\n        are applied in the opposite order.\n\n        Parameters\n        ==========\n\n        angle1, angle2, angle3 : Expr\n            Three successive angles to rotate the coordinate system by\n\n        rotation_order : string\n            String defining the order of axes for rotation\n\n        See Also\n        ========\n\n        BodyOrienter : Orienter to orient systems wrt Euler angles.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, SpaceOrienter\n        >>> from sympy import symbols\n        >>> q1, q2, q3 = symbols('q1 q2 q3')\n        >>> N = CoordSys3D('N')\n\n        To orient a coordinate system D with respect to N, each\n        sequential rotation is always about N's orthogonal unit vectors.\n        For example, a '123' rotation will specify rotations about\n        N.i, then N.j, then N.k.\n        Therefore,\n\n        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')\n        >>> D = N.orient_new('D', (space_orienter, ))\n\n        is same as\n\n        >>> from sympy.vector import AxisOrienter\n        >>> axis_orienter1 = AxisOrienter(q1, N.i)\n        >>> B = N.orient_new('B', (axis_orienter1, ))\n        >>> axis_orienter2 = AxisOrienter(q2, N.j)\n        >>> C = B.orient_new('C', (axis_orienter2, ))\n        >>> axis_orienter3 = AxisOrienter(q3, N.k)\n        >>> D = C.orient_new('C', (axis_orienter3, ))\n\n        "
        pass

class QuaternionOrienter(Orienter):
    """
    Class to denote a quaternion-orienter.
    """

    def __new__(cls, q0, q1, q2, q3):
        if False:
            print('Hello World!')
        q0 = sympify(q0)
        q1 = sympify(q1)
        q2 = sympify(q2)
        q3 = sympify(q3)
        parent_orient = Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)], [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)], [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])
        parent_orient = parent_orient.T
        obj = super().__new__(cls, q0, q1, q2, q3)
        obj._q0 = q0
        obj._q1 = q1
        obj._q2 = q2
        obj._q3 = q3
        obj._parent_orient = parent_orient
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        if False:
            i = 10
            return i + 15
        "\n        Quaternion orientation orients the new CoordSys3D with\n        Quaternions, defined as a finite rotation about lambda, a unit\n        vector, by some amount theta.\n\n        This orientation is described by four parameters:\n\n        q0 = cos(theta/2)\n\n        q1 = lambda_x sin(theta/2)\n\n        q2 = lambda_y sin(theta/2)\n\n        q3 = lambda_z sin(theta/2)\n\n        Quaternion does not take in a rotation order.\n\n        Parameters\n        ==========\n\n        q0, q1, q2, q3 : Expr\n            The quaternions to rotate the coordinate system by\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> from sympy import symbols\n        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')\n        >>> N = CoordSys3D('N')\n        >>> from sympy.vector import QuaternionOrienter\n        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)\n        >>> B = N.orient_new('B', (q_orienter, ))\n\n        "
        pass

    @property
    def q0(self):
        if False:
            print('Hello World!')
        return self._q0

    @property
    def q1(self):
        if False:
            print('Hello World!')
        return self._q1

    @property
    def q2(self):
        if False:
            return 10
        return self._q2

    @property
    def q3(self):
        if False:
            while True:
                i = 10
        return self._q3

def _rot(axis, angle):
    if False:
        while True:
            i = 10
    'DCM for simple axis 1, 2 or 3 rotations. '
    if axis == 1:
        return Matrix(rot_axis1(angle).T)
    elif axis == 2:
        return Matrix(rot_axis2(angle).T)
    elif axis == 3:
        return Matrix(rot_axis3(angle).T)