from sympy import Symbol, S
from sympy.physics.vector import ReferenceFrame, Dyadic, Point, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass, Inertia
from sympy.utilities.exceptions import sympy_deprecation_warning
__all__ = ['RigidBody']

class RigidBody(BodyBase):
    """An idealized rigid body.

    Explanation
    ===========

    This is essentially a container which holds the various components which
    describe a rigid body: a name, mass, center of mass, reference frame, and
    inertia.

    All of these need to be supplied on creation, but can be changed
    afterwards.

    Attributes
    ==========

    name : string
        The body's name.
    masscenter : Point
        The point which represents the center of mass of the rigid body.
    frame : ReferenceFrame
        The ReferenceFrame which the rigid body is fixed in.
    mass : Sympifyable
        The body's mass.
    inertia : (Dyadic, Point)
        The body's inertia about a point; stored in a tuple as shown above.
    potential_energy : Sympifyable
        The potential energy of the RigidBody.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
    >>> from sympy.physics.mechanics import outer
    >>> m = Symbol('m')
    >>> A = ReferenceFrame('A')
    >>> P = Point('P')
    >>> I = outer (A.x, A.x)
    >>> inertia_tuple = (I, P)
    >>> B = RigidBody('B', P, A, m, inertia_tuple)
    >>> # Or you could change them afterwards
    >>> m2 = Symbol('m2')
    >>> B.mass = m2

    """

    def __init__(self, name, masscenter=None, frame=None, mass=None, inertia=None):
        if False:
            return 10
        super().__init__(name, masscenter, mass)
        if frame is None:
            frame = ReferenceFrame(f'{name}_frame')
        self.frame = frame
        if inertia is None:
            ixx = Symbol(f'{name}_ixx')
            iyy = Symbol(f'{name}_iyy')
            izz = Symbol(f'{name}_izz')
            izx = Symbol(f'{name}_izx')
            ixy = Symbol(f'{name}_ixy')
            iyz = Symbol(f'{name}_iyz')
            inertia = Inertia.from_inertia_scalars(self.masscenter, self.frame, ixx, iyy, izz, ixy, iyz, izx)
        self.inertia = inertia

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({repr(self.name)}, masscenter={repr(self.masscenter)}, frame={repr(self.frame)}, mass={repr(self.mass)}), inertia={repr(self.inertia)}))'

    @property
    def frame(self):
        if False:
            return 10
        'The ReferenceFrame fixed to the body.'
        return self._frame

    @frame.setter
    def frame(self, F):
        if False:
            i = 10
            return i + 15
        if not isinstance(F, ReferenceFrame):
            raise TypeError('RigidBody frame must be a ReferenceFrame object.')
        self._frame = F

    @property
    def x(self):
        if False:
            return 10
        'The basis Vector for the body, in the x direction. '
        return self.frame.x

    @property
    def y(self):
        if False:
            print('Hello World!')
        'The basis Vector for the body, in the y direction. '
        return self.frame.y

    @property
    def z(self):
        if False:
            i = 10
            return i + 15
        'The basis Vector for the body, in the z direction. '
        return self.frame.z

    @property
    def inertia(self):
        if False:
            for i in range(10):
                print('nop')
        "The body's inertia about a point; stored as (Dyadic, Point)."
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        if False:
            print('Hello World!')
        if len(I) != 2 or not isinstance(I[0], Dyadic) or (not isinstance(I[1], Point)):
            raise TypeError('RigidBody inertia must be a tuple of the form (Dyadic, Point).')
        self._inertia = Inertia(I[0], I[1])
        I_Ss_O = inertia_of_point_mass(self.mass, self.masscenter.pos_from(I[1]), self.frame)
        self._central_inertia = I[0] - I_Ss_O

    @property
    def central_inertia(self):
        if False:
            print('Hello World!')
        "The body's central inertia dyadic."
        return self._central_inertia

    @central_inertia.setter
    def central_inertia(self, I):
        if False:
            i = 10
            return i + 15
        if not isinstance(I, Dyadic):
            raise TypeError('RigidBody inertia must be a Dyadic object.')
        self.inertia = Inertia(I, self.masscenter)

    def linear_momentum(self, frame):
        if False:
            for i in range(10):
                print('nop')
        " Linear momentum of the rigid body.\n\n        Explanation\n        ===========\n\n        The linear momentum L, of a rigid body B, with respect to frame N is\n        given by:\n\n        ``L = m * v``\n\n        where m is the mass of the rigid body, and v is the velocity of the mass\n        center of B in the frame N.\n\n        Parameters\n        ==========\n\n        frame : ReferenceFrame\n            The frame in which linear momentum is desired.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer\n        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols\n        >>> from sympy.physics.vector import init_vprinting\n        >>> init_vprinting(pretty_print=False)\n        >>> m, v = dynamicsymbols('m v')\n        >>> N = ReferenceFrame('N')\n        >>> P = Point('P')\n        >>> P.set_vel(N, v * N.x)\n        >>> I = outer (N.x, N.x)\n        >>> Inertia_tuple = (I, P)\n        >>> B = RigidBody('B', P, N, m, Inertia_tuple)\n        >>> B.linear_momentum(N)\n        m*v*N.x\n\n        "
        return self.mass * self.masscenter.vel(frame)

    def angular_momentum(self, point, frame):
        if False:
            i = 10
            return i + 15
        "Returns the angular momentum of the rigid body about a point in the\n        given frame.\n\n        Explanation\n        ===========\n\n        The angular momentum H of a rigid body B about some point O in a frame N\n        is given by:\n\n        ``H = dot(I, w) + cross(r, m * v)``\n\n        where I and m are the central inertia dyadic and mass of rigid body B, w\n        is the angular velocity of body B in the frame N, r is the position\n        vector from point O to the mass center of B, and v is the velocity of\n        the mass center in the frame N.\n\n        Parameters\n        ==========\n\n        point : Point\n            The point about which angular momentum is desired.\n        frame : ReferenceFrame\n            The frame in which angular momentum is desired.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer\n        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols\n        >>> from sympy.physics.vector import init_vprinting\n        >>> init_vprinting(pretty_print=False)\n        >>> m, v, r, omega = dynamicsymbols('m v r omega')\n        >>> N = ReferenceFrame('N')\n        >>> b = ReferenceFrame('b')\n        >>> b.set_ang_vel(N, omega * b.x)\n        >>> P = Point('P')\n        >>> P.set_vel(N, 1 * N.x)\n        >>> I = outer(b.x, b.x)\n        >>> B = RigidBody('B', P, b, m, (I, P))\n        >>> B.angular_momentum(P, N)\n        omega*b.x\n\n        "
        I = self.central_inertia
        w = self.frame.ang_vel_in(frame)
        m = self.mass
        r = self.masscenter.pos_from(point)
        v = self.masscenter.vel(frame)
        return I.dot(w) + r.cross(m * v)

    def kinetic_energy(self, frame):
        if False:
            return 10
        "Kinetic energy of the rigid body.\n\n        Explanation\n        ===========\n\n        The kinetic energy, T, of a rigid body, B, is given by:\n\n        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``\n\n        where I and m are the central inertia dyadic and mass of rigid body B\n        respectively, w is the body's angular velocity, and v is the velocity of\n        the body's mass center in the supplied ReferenceFrame.\n\n        Parameters\n        ==========\n\n        frame : ReferenceFrame\n            The RigidBody's angular velocity and the velocity of it's mass\n            center are typically defined with respect to an inertial frame but\n            any relevant frame in which the velocities are known can be\n            supplied.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer\n        >>> from sympy.physics.mechanics import RigidBody\n        >>> from sympy import symbols\n        >>> m, v, r, omega = symbols('m v r omega')\n        >>> N = ReferenceFrame('N')\n        >>> b = ReferenceFrame('b')\n        >>> b.set_ang_vel(N, omega * b.x)\n        >>> P = Point('P')\n        >>> P.set_vel(N, v * N.x)\n        >>> I = outer (b.x, b.x)\n        >>> inertia_tuple = (I, P)\n        >>> B = RigidBody('B', P, b, m, inertia_tuple)\n        >>> B.kinetic_energy(N)\n        m*v**2/2 + omega**2/2\n\n        "
        rotational_KE = S.Half * dot(self.frame.ang_vel_in(frame), dot(self.central_inertia, self.frame.ang_vel_in(frame)))
        translational_KE = S.Half * self.mass * dot(self.masscenter.vel(frame), self.masscenter.vel(frame))
        return rotational_KE + translational_KE

    def set_potential_energy(self, scalar):
        if False:
            print('Hello World!')
        sympy_deprecation_warning('\nThe sympy.physics.mechanics.RigidBody.set_potential_energy()\nmethod is deprecated. Instead use\n\n    B.potential_energy = scalar\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-set-potential-energy')
        self.potential_energy = scalar

    def parallel_axis(self, point, frame=None):
        if False:
            return 10
        'Returns the inertia dyadic of the body with respect to another point.\n\n        Parameters\n        ==========\n\n        point : sympy.physics.vector.Point\n            The point to express the inertia dyadic about.\n        frame : sympy.physics.vector.ReferenceFrame\n            The reference frame used to construct the dyadic.\n\n        Returns\n        =======\n\n        inertia : sympy.physics.vector.Dyadic\n            The inertia dyadic of the rigid body expressed about the provided\n            point.\n\n        '
        if frame is None:
            frame = self.frame
        return self.central_inertia + inertia_of_point_mass(self.mass, self.masscenter.pos_from(point), frame)