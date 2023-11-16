from abc import ABC
from collections import namedtuple
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.vector import Vector, ReferenceFrame, Point
__all__ = ['LoadBase', 'Force', 'Torque']

class LoadBase(ABC, namedtuple('LoadBase', ['location', 'vector'])):
    """Abstract base class for the various loading types."""

    def __add__(self, other):
        if False:
            while True:
                i = 10
        raise TypeError(f"unsupported operand type(s) for +: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

    def __mul__(self, other):
        if False:
            print('Hello World!')
        raise TypeError(f"unsupported operand type(s) for *: '{self.__class__.__name__}' and '{other.__class__.__name__}'")
    __radd__ = __add__
    __rmul__ = __mul__

class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """

    def __new__(cls, point, force):
        if False:
            i = 10
            return i + 15
        if isinstance(point, BodyBase):
            point = point.masscenter
        if not isinstance(point, Point):
            raise TypeError('Force location should be a Point.')
        if not isinstance(force, Vector):
            raise TypeError('Force vector should be a Vector.')
        return super().__new__(cls, point, force)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(point={self.point}, force={self.force})'

    @property
    def point(self):
        if False:
            for i in range(10):
                print('nop')
        return self.location

    @property
    def force(self):
        if False:
            i = 10
            return i + 15
        return self.vector

class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    >>> from sympy.physics.mechanics import RigidBody
    >>> rb = RigidBody('rb', frame=N)
    >>> Torque(rb, 2 * N.x)
    (N, 2*N.x)

    """

    def __new__(cls, frame, torque):
        if False:
            return 10
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('Torque location should be a ReferenceFrame.')
        if not isinstance(torque, Vector):
            raise TypeError('Torque vector should be a Vector.')
        return super().__new__(cls, frame, torque)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}(frame={self.frame}, torque={self.torque})'

    @property
    def frame(self):
        if False:
            return 10
        return self.location

    @property
    def torque(self):
        if False:
            return 10
        return self.vector

def gravity(acceleration, *bodies):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of gravity forces given the acceleration\n    due to gravity and any number of particles or rigidbodies.\n\n    Example\n    =======\n\n    >>> from sympy.physics.mechanics import ReferenceFrame, Particle, RigidBody\n    >>> from sympy.physics.mechanics.loads import gravity\n    >>> from sympy import symbols\n    >>> N = ReferenceFrame('N')\n    >>> g = symbols('g')\n    >>> P = Particle('P')\n    >>> B = RigidBody('B')\n    >>> gravity(g*N.y, P, B)\n    [(P_masscenter, P_mass*g*N.y),\n     (B_masscenter, B_mass*g*N.y)]\n\n    "
    gravity_force = []
    for body in bodies:
        if not isinstance(body, BodyBase):
            raise TypeError(f'{type(body)} is not a body type')
        gravity_force.append(Force(body.masscenter, body.mass * acceleration))
    return gravity_force

def _parse_load(load):
    if False:
        while True:
            i = 10
    'Helper function to parse loads and convert tuples to load objects.'
    if isinstance(load, LoadBase):
        return load
    elif isinstance(load, tuple):
        if len(load) != 2:
            raise ValueError(f'Load {load} should have a length of 2.')
        if isinstance(load[0], Point):
            return Force(load[0], load[1])
        elif isinstance(load[0], ReferenceFrame):
            return Torque(load[0], load[1])
        else:
            raise ValueError(f'Load not recognized. The load location {load[0]} should either be a Point or a ReferenceFrame.')
    raise TypeError(f'Load type {type(load)} not recognized as a load. It should be a Force, Torque or tuple.')