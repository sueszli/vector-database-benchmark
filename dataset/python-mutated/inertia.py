from sympy import sympify
from sympy.physics.vector import Point, Dyadic, ReferenceFrame
from collections import namedtuple
__all__ = ['inertia', 'inertia_of_point_mass', 'Inertia']

def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    if False:
        while True:
            i = 10
    "Simple way to create inertia Dyadic object.\n\n    Explanation\n    ===========\n\n    Creates an inertia Dyadic based on the given tensor values and a body-fixed\n    reference frame.\n\n    Parameters\n    ==========\n\n    frame : ReferenceFrame\n        The frame the inertia is defined in.\n    ixx : Sympifyable\n        The xx element in the inertia dyadic.\n    iyy : Sympifyable\n        The yy element in the inertia dyadic.\n    izz : Sympifyable\n        The zz element in the inertia dyadic.\n    ixy : Sympifyable\n        The xy element in the inertia dyadic.\n    iyz : Sympifyable\n        The yz element in the inertia dyadic.\n    izx : Sympifyable\n        The zx element in the inertia dyadic.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.mechanics import ReferenceFrame, inertia\n    >>> N = ReferenceFrame('N')\n    >>> inertia(N, 1, 2, 3)\n    (N.x|N.x) + 2*(N.y|N.y) + 3*(N.z|N.z)\n\n    "
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Need to define the inertia in a frame')
    (ixx, iyy, izz) = (sympify(ixx), sympify(iyy), sympify(izz))
    (ixy, iyz, izx) = (sympify(ixy), sympify(iyz), sympify(izx))
    return ixx * (frame.x | frame.x) + ixy * (frame.x | frame.y) + izx * (frame.x | frame.z) + ixy * (frame.y | frame.x) + iyy * (frame.y | frame.y) + iyz * (frame.y | frame.z) + izx * (frame.z | frame.x) + iyz * (frame.z | frame.y) + izz * (frame.z | frame.z)

def inertia_of_point_mass(mass, pos_vec, frame):
    if False:
        for i in range(10):
            print('nop')
    "Inertia dyadic of a point mass relative to point O.\n\n    Parameters\n    ==========\n\n    mass : Sympifyable\n        Mass of the point mass\n    pos_vec : Vector\n        Position from point O to point mass\n    frame : ReferenceFrame\n        Reference frame to express the dyadic in\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass\n    >>> N = ReferenceFrame('N')\n    >>> r, m = symbols('r m')\n    >>> px = r * N.x\n    >>> inertia_of_point_mass(m, px, N)\n    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)\n\n    "
    return mass * (((frame.x | frame.x) + (frame.y | frame.y) + (frame.z | frame.z)) * (pos_vec & pos_vec) - (pos_vec | pos_vec))

class Inertia(namedtuple('Inertia', ['dyadic', 'point'])):
    """Inertia object consisting of a Dyadic and a Point of reference.

    Explanation
    ===========

    This is a simple class to store the Point and Dyadic, belonging to an
    inertia.

    Attributes
    ==========

    dyadic : Dyadic
        The dyadic of the inertia.
    point : Point
        The reference point of the inertia.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Inertia(N.x.outer(N.x) + N.y.outer(N.y) + N.z.outer(N.z), Po)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    In the example above the Dyadic was created manually, one can however also
    use the ``inertia`` function for this or the class method ``from_tensor`` as
    shown below.

    >>> Inertia.from_inertia_scalars(Po, N, 1, 1, 1)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    """

    def __new__(cls, dyadic, point):
        if False:
            while True:
                i = 10
        if isinstance(dyadic, Point) and isinstance(point, Dyadic):
            (point, dyadic) = (dyadic, point)
        if not isinstance(point, Point):
            raise TypeError('Reference point should be of type Point')
        if not isinstance(dyadic, Dyadic):
            raise TypeError('Inertia value should be expressed as a Dyadic')
        return super().__new__(cls, dyadic, point)

    @classmethod
    def from_inertia_scalars(cls, point, frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
        if False:
            while True:
                i = 10
        "Simple way to create an Inertia object based on the tensor values.\n\n        Explanation\n        ===========\n\n        This class method uses the :func`~.inertia` to create the Dyadic based\n        on the tensor values.\n\n        Parameters\n        ==========\n\n        point : Point\n            The reference point of the inertia.\n        frame : ReferenceFrame\n            The frame the inertia is defined in.\n        ixx : Sympifyable\n            The xx element in the inertia dyadic.\n        iyy : Sympifyable\n            The yy element in the inertia dyadic.\n        izz : Sympifyable\n            The zz element in the inertia dyadic.\n        ixy : Sympifyable\n            The xy element in the inertia dyadic.\n        iyz : Sympifyable\n            The yz element in the inertia dyadic.\n        izx : Sympifyable\n            The zx element in the inertia dyadic.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia\n        >>> ixx, iyy, izz, ixy, iyz, izx = symbols('ixx iyy izz ixy iyz izx')\n        >>> N = ReferenceFrame('N')\n        >>> P = Point('P')\n        >>> I = Inertia.from_inertia_scalars(P, N, ixx, iyy, izz, ixy, iyz, izx)\n\n        The tensor values can easily be seen when converting the dyadic to a\n        matrix.\n\n        >>> I.dyadic.to_matrix(N)\n        Matrix([\n        [ixx, ixy, izx],\n        [ixy, iyy, iyz],\n        [izx, iyz, izz]])\n\n        "
        return cls(inertia(frame, ixx, iyy, izz, ixy, iyz, izx), point)

    def __add__(self, other):
        if False:
            return 10
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError(f"unsupported operand type(s) for *: '{self.__class__.__name__}' and '{other.__class__.__name__}'")
    __radd__ = __add__
    __rmul__ = __mul__