from sympy.physics.mechanics import Body, Lagrangian, KanesMethod, LagrangesMethod, RigidBody, Particle
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.method import _Methods
from sympy import Matrix
__all__ = ['JointsMethod']

class JointsMethod(_Methods):
    """Method for formulating the equations of motion using a set of interconnected bodies with joints.

    Parameters
    ==========

    newtonion : Body or ReferenceFrame
        The newtonion(inertial) frame.
    *joints : Joint
        The joints in the system

    Attributes
    ==========

    q, u : iterable
        Iterable of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Body objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    mass_matrix : Matrix, shape(n, n)
        The system's mass matrix
    forcing : Matrix, shape(n, 1)
        The system's forcing vector
    mass_matrix_full : Matrix, shape(2*n, 2*n)
        The "mass matrix" for the u's and q's
    forcing_full : Matrix, shape(2*n, 1)
        The "forcing vector" for the u's and q's
    method : KanesMethod or Lagrange's method
        Method's object.
    kdes : iterable
        Iterable of kde in they system.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, JointsMethod, PrismaticJoint
    >>> from sympy.physics.vector import dynamicsymbols
    >>> c, k = symbols('c k')
    >>> x, v = dynamicsymbols('x v')
    >>> wall = Body('W')
    >>> body = Body('B')
    >>> J = PrismaticJoint('J', wall, body, coordinates=x, speeds=v)
    >>> wall.apply_force(c*v*wall.x, reaction_body=body)
    >>> wall.apply_force(k*x*wall.x, reaction_body=body)
    >>> method = JointsMethod(wall, J)
    >>> method.form_eoms()
    Matrix([[-B_mass*Derivative(v(t), t) - c*v(t) - k*x(t)]])
    >>> M = method.mass_matrix_full
    >>> F = method.forcing_full
    >>> rhs = M.LUsolve(F)
    >>> rhs
    Matrix([
    [                     v(t)],
    [(-c*v(t) - k*x(t))/B_mass]])

    Notes
    =====

    ``JointsMethod`` currently only works with systems that do not have any
    configuration or motion constraints.

    """

    def __init__(self, newtonion, *joints):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(newtonion, BodyBase):
            self.frame = newtonion.frame
        else:
            self.frame = newtonion
        self._joints = joints
        self._bodies = self._generate_bodylist()
        self._loads = self._generate_loadlist()
        self._q = self._generate_q()
        self._u = self._generate_u()
        self._kdes = self._generate_kdes()
        self._method = None

    @property
    def bodies(self):
        if False:
            return 10
        'List of bodies in they system.'
        return self._bodies

    @property
    def loads(self):
        if False:
            i = 10
            return i + 15
        'List of loads on the system.'
        return self._loads

    @property
    def q(self):
        if False:
            while True:
                i = 10
        'List of the generalized coordinates.'
        return self._q

    @property
    def u(self):
        if False:
            return 10
        'List of the generalized speeds.'
        return self._u

    @property
    def kdes(self):
        if False:
            for i in range(10):
                print('nop')
        'List of the generalized coordinates.'
        return self._kdes

    @property
    def forcing_full(self):
        if False:
            for i in range(10):
                print('nop')
        'The "forcing vector" for the u\'s and q\'s.'
        return self.method.forcing_full

    @property
    def mass_matrix_full(self):
        if False:
            i = 10
            return i + 15
        'The "mass matrix" for the u\'s and q\'s.'
        return self.method.mass_matrix_full

    @property
    def mass_matrix(self):
        if False:
            print('Hello World!')
        "The system's mass matrix."
        return self.method.mass_matrix

    @property
    def forcing(self):
        if False:
            i = 10
            return i + 15
        "The system's forcing vector."
        return self.method.forcing

    @property
    def method(self):
        if False:
            print('Hello World!')
        'Object of method used to form equations of systems.'
        return self._method

    def _generate_bodylist(self):
        if False:
            i = 10
            return i + 15
        bodies = []
        for joint in self._joints:
            if joint.child not in bodies:
                bodies.append(joint.child)
            if joint.parent not in bodies:
                bodies.append(joint.parent)
        return bodies

    def _generate_loadlist(self):
        if False:
            return 10
        load_list = []
        for body in self.bodies:
            if isinstance(body, Body):
                load_list.extend(body.loads)
        return load_list

    def _generate_q(self):
        if False:
            while True:
                i = 10
        q_ind = []
        for joint in self._joints:
            for coordinate in joint.coordinates:
                if coordinate in q_ind:
                    raise ValueError('Coordinates of joints should be unique.')
                q_ind.append(coordinate)
        return Matrix(q_ind)

    def _generate_u(self):
        if False:
            while True:
                i = 10
        u_ind = []
        for joint in self._joints:
            for speed in joint.speeds:
                if speed in u_ind:
                    raise ValueError('Speeds of joints should be unique.')
                u_ind.append(speed)
        return Matrix(u_ind)

    def _generate_kdes(self):
        if False:
            for i in range(10):
                print('nop')
        kd_ind = Matrix(1, 0, []).T
        for joint in self._joints:
            kd_ind = kd_ind.col_join(joint.kdes)
        return kd_ind

    def _convert_bodies(self):
        if False:
            print('Hello World!')
        bodylist = []
        for body in self.bodies:
            if not isinstance(body, Body):
                bodylist.append(body)
                continue
            if body.is_rigidbody:
                rb = RigidBody(body.name, body.masscenter, body.frame, body.mass, (body.central_inertia, body.masscenter))
                rb.potential_energy = body.potential_energy
                bodylist.append(rb)
            else:
                part = Particle(body.name, body.masscenter, body.mass)
                part.potential_energy = body.potential_energy
                bodylist.append(part)
        return bodylist

    def form_eoms(self, method=KanesMethod):
        if False:
            for i in range(10):
                print('nop')
        "Method to form system's equation of motions.\n\n        Parameters\n        ==========\n\n        method : Class\n            Class name of method.\n\n        Returns\n        ========\n\n        Matrix\n            Vector of equations of motions.\n\n        Examples\n        ========\n\n        This is a simple example for a one degree of freedom translational\n        spring-mass-damper.\n\n        >>> from sympy import S, symbols\n        >>> from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, Body\n        >>> from sympy.physics.mechanics import PrismaticJoint, JointsMethod\n        >>> q = dynamicsymbols('q')\n        >>> qd = dynamicsymbols('q', 1)\n        >>> m, k, b = symbols('m k b')\n        >>> wall = Body('W')\n        >>> part = Body('P', mass=m)\n        >>> part.potential_energy = k * q**2 / S(2)\n        >>> J = PrismaticJoint('J', wall, part, coordinates=q, speeds=qd)\n        >>> wall.apply_force(b * qd * wall.x, reaction_body=part)\n        >>> method = JointsMethod(wall, J)\n        >>> method.form_eoms(LagrangesMethod)\n        Matrix([[b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])\n\n        We can also solve for the states using the 'rhs' method.\n\n        >>> method.rhs()\n        Matrix([\n        [                Derivative(q(t), t)],\n        [(-b*Derivative(q(t), t) - k*q(t))/m]])\n\n        "
        bodylist = self._convert_bodies()
        if issubclass(method, LagrangesMethod):
            L = Lagrangian(self.frame, *bodylist)
            self._method = method(L, self.q, self.loads, bodylist, self.frame)
        else:
            self._method = method(self.frame, q_ind=self.q, u_ind=self.u, kd_eqs=self.kdes, forcelist=self.loads, bodies=bodylist)
        soln = self.method._form_eoms()
        return soln

    def rhs(self, inv_method=None):
        if False:
            print('Hello World!')
        "Returns equations that can be solved numerically.\n\n        Parameters\n        ==========\n\n        inv_method : str\n            The specific sympy inverse matrix calculation method to use. For a\n            list of valid methods, see\n            :meth:`~sympy.matrices.matrices.MatrixBase.inv`\n\n        Returns\n        ========\n\n        Matrix\n            Numerically solvable equations.\n\n        See Also\n        ========\n\n        sympy.physics.mechanics.kane.KanesMethod.rhs:\n            KanesMethod's rhs function.\n        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:\n            LagrangesMethod's rhs function.\n\n        "
        return self.method.rhs(inv_method=inv_method)