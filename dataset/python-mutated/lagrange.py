from sympy import diff, zeros, Matrix, eye, sympify
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.functions import find_dynamicsymbols, msubs, _f_list_parser, _validate_coordinates
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
__all__ = ['LagrangesMethod']

class LagrangesMethod(_Methods):
    """Lagrange's method object.

    Explanation
    ===========

    This object generates the equations of motion in a two step procedure. The
    first step involves the initialization of LagrangesMethod by supplying the
    Lagrangian and the generalized coordinates, at the bare minimum. If there
    are any constraint equations, they can be supplied as keyword arguments.
    The Lagrange multipliers are automatically generated and are equal in
    number to the constraint equations. Similarly any non-conservative forces
    can be supplied in an iterable (as described below and also shown in the
    example) along with a ReferenceFrame. This is also discussed further in the
    __init__ method.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    bodies : iterable
        Iterable containing the rigid bodies and particles of the system.
    mass_matrix : Matrix
        The system's mass matrix
    forcing : Matrix
        The system's forcing vector
    mass_matrix_full : Matrix
        The "mass matrix" for the qdot's, qdoubledot's, and the
        lagrange multipliers (lam)
    forcing_full : Matrix
        The forcing vector for the qdot's, qdoubledot's and
        lagrange multipliers (lam)

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized coordinates and their derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy.physics.mechanics import LagrangesMethod, Lagrangian
        >>> from sympy.physics.mechanics import ReferenceFrame, Particle, Point
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import symbols
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, qd * N.x)

    We need to then prepare the information as required by LagrangesMethod to
    generate equations of motion.
    First we create the Particle, which has a point attached to it.
    Following this the lagrangian is created from the kinetic and potential
    energies.
    Then, an iterable of nonconservative forces/torques must be constructed,
    where each item is a (Point, Vector) or (ReferenceFrame, Vector) tuple,
    with the Vectors representing the nonconservative forces or torques.

        >>> Pa = Particle('Pa', P, m)
        >>> Pa.potential_energy = k * q**2 / 2.0
        >>> L = Lagrangian(N, Pa)
        >>> fl = [(P, -b * qd * N.x)]

    Finally we can generate the equations of motion.
    First we create the LagrangesMethod object. To do this one must supply
    the Lagrangian, and the generalized coordinates. The constraint equations,
    the forcelist, and the inertial frame may also be provided, if relevant.
    Next we generate Lagrange's equations of motion, such that:
    Lagrange's equations of motion = 0.
    We have the equations of motion at this point.

        >>> l = LagrangesMethod(L, [q], forcelist = fl, frame = N)
        >>> print(l.form_lagranges_equations())
        Matrix([[b*Derivative(q(t), t) + 1.0*k*q(t) + m*Derivative(q(t), (t, 2))]])

    We can also solve for the states using the 'rhs' method.

        >>> print(l.rhs())
        Matrix([[Derivative(q(t), t)], [(-b*Derivative(q(t), t) - 1.0*k*q(t))/m]])

    Please refer to the docstrings on each method for more details.
    """

    def __init__(self, Lagrangian, qs, forcelist=None, bodies=None, frame=None, hol_coneqs=None, nonhol_coneqs=None):
        if False:
            for i in range(10):
                print('nop')
        'Supply the following for the initialization of LagrangesMethod.\n\n        Lagrangian : Sympifyable\n\n        qs : array_like\n            The generalized coordinates\n\n        hol_coneqs : array_like, optional\n            The holonomic constraint equations\n\n        nonhol_coneqs : array_like, optional\n            The nonholonomic constraint equations\n\n        forcelist : iterable, optional\n            Takes an iterable of (Point, Vector) or (ReferenceFrame, Vector)\n            tuples which represent the force at a point or torque on a frame.\n            This feature is primarily to account for the nonconservative forces\n            and/or moments.\n\n        bodies : iterable, optional\n            Takes an iterable containing the rigid bodies and particles of the\n            system.\n\n        frame : ReferenceFrame, optional\n            Supply the inertial frame. This is used to determine the\n            generalized forces due to non-conservative forces.\n        '
        self._L = Matrix([sympify(Lagrangian)])
        self.eom = None
        self._m_cd = Matrix()
        self._m_d = Matrix()
        self._f_cd = Matrix()
        self._f_d = Matrix()
        self.lam_coeffs = Matrix()
        forcelist = forcelist if forcelist else []
        if not iterable(forcelist):
            raise TypeError('Force pairs must be supplied in an iterable.')
        self._forcelist = forcelist
        if frame and (not isinstance(frame, ReferenceFrame)):
            raise TypeError('frame must be a valid ReferenceFrame')
        self._bodies = bodies
        self.inertial = frame
        self.lam_vec = Matrix()
        self._term1 = Matrix()
        self._term2 = Matrix()
        self._term3 = Matrix()
        self._term4 = Matrix()
        if not iterable(qs):
            raise TypeError('Generalized coordinates must be an iterable')
        self._q = Matrix(qs)
        self._qdots = self.q.diff(dynamicsymbols._t)
        self._qdoubledots = self._qdots.diff(dynamicsymbols._t)
        _validate_coordinates(self.q)
        mat_build = lambda x: Matrix(x) if x else Matrix()
        hol_coneqs = mat_build(hol_coneqs)
        nonhol_coneqs = mat_build(nonhol_coneqs)
        self.coneqs = Matrix([hol_coneqs.diff(dynamicsymbols._t), nonhol_coneqs])
        self._hol_coneqs = hol_coneqs

    def form_lagranges_equations(self):
        if False:
            i = 10
            return i + 15
        "Method to form Lagrange's equations of motion.\n\n        Returns a vector of equations of motion using Lagrange's equations of\n        the second kind.\n        "
        qds = self._qdots
        qdd_zero = {i: 0 for i in self._qdoubledots}
        n = len(self.q)
        self._term1 = self._L.jacobian(qds)
        self._term1 = self._term1.diff(dynamicsymbols._t).T
        self._term2 = self._L.jacobian(self.q).T
        if self.coneqs:
            coneqs = self.coneqs
            m = len(coneqs)
            self.lam_vec = Matrix(dynamicsymbols('lam1:' + str(m + 1)))
            self.lam_coeffs = -coneqs.jacobian(qds)
            self._term3 = self.lam_coeffs.T * self.lam_vec
            diffconeqs = coneqs.diff(dynamicsymbols._t)
            self._m_cd = diffconeqs.jacobian(self._qdoubledots)
            self._f_cd = -diffconeqs.subs(qdd_zero)
        else:
            self._term3 = zeros(n, 1)
        if self.forcelist:
            N = self.inertial
            self._term4 = zeros(n, 1)
            for (i, qd) in enumerate(qds):
                flist = zip(*_f_list_parser(self.forcelist, N))
                self._term4[i] = sum((v.diff(qd, N) & f for (v, f) in flist))
        else:
            self._term4 = zeros(n, 1)
        without_lam = self._term1 - self._term2 - self._term4
        self._m_d = without_lam.jacobian(self._qdoubledots)
        self._f_d = -without_lam.subs(qdd_zero)
        self.eom = without_lam - self._term3
        return self.eom

    def _form_eoms(self):
        if False:
            i = 10
            return i + 15
        return self.form_lagranges_equations()

    @property
    def mass_matrix(self):
        if False:
            return 10
        "Returns the mass matrix, which is augmented by the Lagrange\n        multipliers, if necessary.\n\n        Explanation\n        ===========\n\n        If the system is described by 'n' generalized coordinates and there are\n        no constraint equations then an n X n matrix is returned.\n\n        If there are 'n' generalized coordinates and 'm' constraint equations\n        have been supplied during initialization then an n X (n+m) matrix is\n        returned. The (n + m - 1)th and (n + m)th columns contain the\n        coefficients of the Lagrange multipliers.\n        "
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return self._m_d.row_join(self.lam_coeffs.T)
        else:
            return self._m_d

    @property
    def mass_matrix_full(self):
        if False:
            i = 10
            return i + 15
        'Augments the coefficients of qdots to the mass_matrix.'
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        n = len(self.q)
        m = len(self.coneqs)
        row1 = eye(n).row_join(zeros(n, n + m))
        row2 = zeros(n, n).row_join(self.mass_matrix)
        if self.coneqs:
            row3 = zeros(m, n).row_join(self._m_cd).row_join(zeros(m, m))
            return row1.col_join(row2).col_join(row3)
        else:
            return row1.col_join(row2)

    @property
    def forcing(self):
        if False:
            return 10
        "Returns the forcing vector from 'lagranges_equations' method."
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        return self._f_d

    @property
    def forcing_full(self):
        if False:
            while True:
                i = 10
        'Augments qdots to the forcing vector above.'
        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return self._qdots.col_join(self.forcing).col_join(self._f_cd)
        else:
            return self._qdots.col_join(self.forcing)

    def to_linearizer(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None, linear_solver='LU'):
        if False:
            print('Hello World!')
        "Returns an instance of the Linearizer class, initiated from the data\n        in the LagrangesMethod class. This may be more desirable than using the\n        linearize class method, as the Linearizer object will allow more\n        efficient recalculation (i.e. about varying operating points).\n\n        Parameters\n        ==========\n\n        q_ind, qd_ind : array_like, optional\n            The independent generalized coordinates and speeds.\n        q_dep, qd_dep : array_like, optional\n            The dependent generalized coordinates and speeds.\n        linear_solver : str, callable\n            Method used to solve the several symbolic linear systems of the\n            form ``A*x=b`` in the linearization process. If a string is\n            supplied, it should be a valid method that can be used with the\n            :meth:`sympy.matrices.matrices.MatrixBase.solve`. If a callable is\n            supplied, it should have the format ``x = f(A, b)``, where it\n            solves the equations and returns the solution. The default is\n            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.\n            ``LUsolve()`` is fast to compute but will often result in\n            divide-by-zero and thus ``nan`` results.\n\n        Returns\n        =======\n        Linearizer\n            An instantiated\n            :class:`sympy.physics.mechanics.linearize.Linearizer`.\n\n        "
        t = dynamicsymbols._t
        q = self.q
        u = self._qdots
        ud = u.diff(t)
        lams = self.lam_vec
        mat_build = lambda x: Matrix(x) if x else Matrix()
        q_i = mat_build(q_ind)
        q_d = mat_build(q_dep)
        u_i = mat_build(qd_ind)
        u_d = mat_build(qd_dep)
        f_c = self._hol_coneqs
        f_v = self.coneqs
        f_a = f_v.diff(t)
        f_0 = u
        f_1 = -u
        f_2 = self._term1
        f_3 = -(self._term2 + self._term4)
        f_4 = -self._term3
        if len(q_d) != len(f_c) or len(u_d) != len(f_v):
            raise ValueError(('Must supply {:} dependent coordinates, and ' + '{:} dependent speeds').format(len(f_c), len(f_v)))
        if set(Matrix([q_i, q_d])) != set(q):
            raise ValueError('Must partition q into q_ind and q_dep, with ' + 'no extra or missing symbols.')
        if set(Matrix([u_i, u_d])) != set(u):
            raise ValueError('Must partition qd into qd_ind and qd_dep, ' + 'with no extra or missing symbols.')
        insyms = set(Matrix([q, u, ud, lams]))
        r = list(find_dynamicsymbols(f_3, insyms))
        r.sort(key=default_sort_key)
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified                                  quantities when linearizing forcing terms.')
        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i, q_d, u_i, u_d, r, lams, linear_solver=linear_solver)

    def linearize(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None, linear_solver='LU', **kwargs):
        if False:
            while True:
                i = 10
        "Linearize the equations of motion about a symbolic operating point.\n\n        Parameters\n        ==========\n        linear_solver : str, callable\n            Method used to solve the several symbolic linear systems of the\n            form ``A*x=b`` in the linearization process. If a string is\n            supplied, it should be a valid method that can be used with the\n            :meth:`sympy.matrices.matrices.MatrixBase.solve`. If a callable is\n            supplied, it should have the format ``x = f(A, b)``, where it\n            solves the equations and returns the solution. The default is\n            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.\n            ``LUsolve()`` is fast to compute but will often result in\n            divide-by-zero and thus ``nan`` results.\n        **kwargs\n            Extra keyword arguments are passed to\n            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.\n\n        Explanation\n        ===========\n\n        If kwarg A_and_B is False (default), returns M, A, B, r for the\n        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.\n\n        If kwarg A_and_B is True, returns A, B, r for the linearized form\n        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is\n        computationally intensive if there are many symbolic parameters. For\n        this reason, it may be more desirable to use the default A_and_B=False,\n        returning M, A, and B. Values may then be substituted in to these\n        matrices, and the state space form found as\n        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.\n\n        In both cases, r is found as all dynamicsymbols in the equations of\n        motion that are not part of q, u, q', or u'. They are sorted in\n        canonical form.\n\n        The operating points may be also entered using the ``op_point`` kwarg.\n        This takes a dictionary of {symbol: value}, or a an iterable of such\n        dictionaries. The values may be numeric or symbolic. The more values\n        you can specify beforehand, the faster this computation will run.\n\n        For more documentation, please see the ``Linearizer`` class."
        linearizer = self.to_linearizer(q_ind, qd_ind, q_dep, qd_dep, linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def solve_multipliers(self, op_point=None, sol_type='dict'):
        if False:
            print('Hello World!')
        "Solves for the values of the lagrange multipliers symbolically at\n        the specified operating point.\n\n        Parameters\n        ==========\n\n        op_point : dict or iterable of dicts, optional\n            Point at which to solve at. The operating point is specified as\n            a dictionary or iterable of dictionaries of {symbol: value}. The\n            value may be numeric or symbolic itself.\n\n        sol_type : str, optional\n            Solution return type. Valid options are:\n            - 'dict': A dict of {symbol : value} (default)\n            - 'Matrix': An ordered column matrix of the solution\n        "
        k = len(self.lam_vec)
        if k == 0:
            raise ValueError('System has no lagrange multipliers to solve for.')
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif iterable(op_point):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        elif op_point is None:
            op_point_dict = {}
        else:
            raise TypeError('op_point must be either a dictionary or an iterable of dictionaries.')
        mass_matrix = self.mass_matrix.col_join(-self.lam_coeffs.row_join(zeros(k, k)))
        force_matrix = self.forcing.col_join(self._f_cd)
        mass_matrix = msubs(mass_matrix, op_point_dict)
        force_matrix = msubs(force_matrix, op_point_dict)
        sol_list = mass_matrix.LUsolve(-force_matrix)[-k:]
        if sol_type == 'dict':
            return dict(zip(self.lam_vec, sol_list))
        elif sol_type == 'Matrix':
            return Matrix(sol_list)
        else:
            raise ValueError('Unknown sol_type {:}.'.format(sol_type))

    def rhs(self, inv_method=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns equations that can be solved numerically.\n\n        Parameters\n        ==========\n\n        inv_method : str\n            The specific sympy inverse matrix calculation method to use. For a\n            list of valid methods, see\n            :meth:`~sympy.matrices.matrices.MatrixBase.inv`\n        '
        if inv_method is None:
            self._rhs = self.mass_matrix_full.LUsolve(self.forcing_full)
        else:
            self._rhs = self.mass_matrix_full.inv(inv_method, try_block_diag=True) * self.forcing_full
        return self._rhs

    @property
    def q(self):
        if False:
            print('Hello World!')
        return self._q

    @property
    def u(self):
        if False:
            i = 10
            return i + 15
        return self._qdots

    @property
    def bodies(self):
        if False:
            i = 10
            return i + 15
        return self._bodies

    @property
    def forcelist(self):
        if False:
            return 10
        return self._forcelist

    @property
    def loads(self):
        if False:
            print('Hello World!')
        return self._forcelist