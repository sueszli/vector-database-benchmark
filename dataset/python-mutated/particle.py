from sympy import S
from sympy.physics.vector import cross, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass
from sympy.utilities.exceptions import sympy_deprecation_warning
__all__ = ['Particle']

class Particle(BodyBase):
    """A particle.

    Explanation
    ===========

    Particles have a non-zero mass and lack spatial extension; they take up no
    space.

    Values need to be supplied on initialization, but can be changed later.

    Parameters
    ==========

    name : str
        Name of particle
    point : Point
        A physics/mechanics Point which represents the position, velocity, and
        acceleration of this Particle
    mass : Sympifyable
        A SymPy expression representing the Particle's mass
    potential_energy : Sympifyable
        The potential energy of the Particle.

    Examples
    ========

    >>> from sympy.physics.mechanics import Particle, Point
    >>> from sympy import Symbol
    >>> po = Point('po')
    >>> m = Symbol('m')
    >>> pa = Particle('pa', po, m)
    >>> # Or you could change these later
    >>> pa.mass = m
    >>> pa.point = po

    """
    point = BodyBase.masscenter

    def __init__(self, name, point=None, mass=None):
        if False:
            while True:
                i = 10
        super().__init__(name, point, mass)

    def linear_momentum(self, frame):
        if False:
            i = 10
            return i + 15
        "Linear momentum of the particle.\n\n        Explanation\n        ===========\n\n        The linear momentum L, of a particle P, with respect to frame N is\n        given by:\n\n        L = m * v\n\n        where m is the mass of the particle, and v is the velocity of the\n        particle in the frame N.\n\n        Parameters\n        ==========\n\n        frame : ReferenceFrame\n            The frame in which linear momentum is desired.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame\n        >>> from sympy.physics.mechanics import dynamicsymbols\n        >>> from sympy.physics.vector import init_vprinting\n        >>> init_vprinting(pretty_print=False)\n        >>> m, v = dynamicsymbols('m v')\n        >>> N = ReferenceFrame('N')\n        >>> P = Point('P')\n        >>> A = Particle('A', P, m)\n        >>> P.set_vel(N, v * N.x)\n        >>> A.linear_momentum(N)\n        m*v*N.x\n\n        "
        return self.mass * self.point.vel(frame)

    def angular_momentum(self, point, frame):
        if False:
            print('Hello World!')
        "Angular momentum of the particle about the point.\n\n        Explanation\n        ===========\n\n        The angular momentum H, about some point O of a particle, P, is given\n        by:\n\n        ``H = cross(r, m * v)``\n\n        where r is the position vector from point O to the particle P, m is\n        the mass of the particle, and v is the velocity of the particle in\n        the inertial frame, N.\n\n        Parameters\n        ==========\n\n        point : Point\n            The point about which angular momentum of the particle is desired.\n\n        frame : ReferenceFrame\n            The frame in which angular momentum is desired.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame\n        >>> from sympy.physics.mechanics import dynamicsymbols\n        >>> from sympy.physics.vector import init_vprinting\n        >>> init_vprinting(pretty_print=False)\n        >>> m, v, r = dynamicsymbols('m v r')\n        >>> N = ReferenceFrame('N')\n        >>> O = Point('O')\n        >>> A = O.locatenew('A', r * N.x)\n        >>> P = Particle('P', A, m)\n        >>> P.point.set_vel(N, v * N.y)\n        >>> P.angular_momentum(O, N)\n        m*r*v*N.z\n\n        "
        return cross(self.point.pos_from(point), self.mass * self.point.vel(frame))

    def kinetic_energy(self, frame):
        if False:
            i = 10
            return i + 15
        "Kinetic energy of the particle.\n\n        Explanation\n        ===========\n\n        The kinetic energy, T, of a particle, P, is given by:\n\n        ``T = 1/2 (dot(m * v, v))``\n\n        where m is the mass of particle P, and v is the velocity of the\n        particle in the supplied ReferenceFrame.\n\n        Parameters\n        ==========\n\n        frame : ReferenceFrame\n            The Particle's velocity is typically defined with respect to\n            an inertial frame but any relevant frame in which the velocity is\n            known can be supplied.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame\n        >>> from sympy import symbols\n        >>> m, v, r = symbols('m v r')\n        >>> N = ReferenceFrame('N')\n        >>> O = Point('O')\n        >>> P = Particle('P', O, m)\n        >>> P.point.set_vel(N, v * N.y)\n        >>> P.kinetic_energy(N)\n        m*v**2/2\n\n        "
        return S.Half * self.mass * dot(self.point.vel(frame), self.point.vel(frame))

    def set_potential_energy(self, scalar):
        if False:
            while True:
                i = 10
        sympy_deprecation_warning('\nThe sympy.physics.mechanics.Particle.set_potential_energy()\nmethod is deprecated. Instead use\n\n    P.potential_energy = scalar\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-set-potential-energy')
        self.potential_energy = scalar

    def parallel_axis(self, point, frame):
        if False:
            return 10
        'Returns an inertia dyadic of the particle with respect to another\n        point and frame.\n\n        Parameters\n        ==========\n\n        point : sympy.physics.vector.Point\n            The point to express the inertia dyadic about.\n        frame : sympy.physics.vector.ReferenceFrame\n            The reference frame used to construct the dyadic.\n\n        Returns\n        =======\n\n        inertia : sympy.physics.vector.Dyadic\n            The inertia dyadic of the particle expressed about the provided\n            point and frame.\n\n        '
        return inertia_of_point_mass(self.mass, self.point.pos_from(point), frame)