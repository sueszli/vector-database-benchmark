from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, RigidBody, LagrangesMethod, Particle, inertia, Lagrangian
from sympy.core.function import Derivative, Function
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises

def test_invalid_coordinates():
    if False:
        print('Hello World!')
    (l, m, g) = symbols('l m g')
    q = symbols('q')
    (N, O) = (ReferenceFrame('N'), Point('O'))
    O.set_vel(N, 0)
    P = Particle('P', Point('P'), m)
    P.point.set_pos(O, l * (sin(q) * N.x - cos(q) * N.y))
    P.potential_energy = m * g * P.point.pos_from(O).dot(N.y)
    L = Lagrangian(N, P)
    raises(ValueError, lambda : LagrangesMethod(L, [q], bodies=P))

def test_disc_on_an_incline_plane():
    if False:
        print('Hello World!')
    (y, theta) = dynamicsymbols('y theta')
    (yd, thetad) = dynamicsymbols('y theta', 1)
    (m, g, R, l, alpha) = symbols('m g R l alpha')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [pi / 2 - alpha, N.z])
    B = A.orientnew('B', 'Axis', [-theta, A.z])
    Do = Point('Do')
    Do.set_vel(N, yd * A.x)
    I = m * R ** 2 / 2 * B.z | B.z
    D = RigidBody('D', Do, B, m, (I, Do))
    D.potential_energy = m * g * (l - y) * sin(alpha)
    L = Lagrangian(N, D)
    q = [y, theta]
    hol_coneqs = [y - R * theta]
    m = LagrangesMethod(L, q, hol_coneqs=hol_coneqs)
    m.form_lagranges_equations()
    rhs = m.rhs()
    rhs.simplify()
    assert rhs[2] == 2 * g * sin(alpha) / 3

def test_simp_pen():
    if False:
        for i in range(10):
            print('nop')
    (q, u) = dynamicsymbols('q u')
    (qd, ud) = dynamicsymbols('q u ', 1)
    (l, m, g) = symbols('l m g')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q, N.z])
    A.set_ang_vel(N, qd * N.z)
    O = Point('O')
    O.set_vel(N, 0)
    P = O.locatenew('P', l * A.x)
    P.v2pt_theory(O, N, A)
    Pa = Particle('Pa', P, m)
    Pa.potential_energy = -m * g * l * cos(q)
    L = Lagrangian(N, Pa)
    lm = LagrangesMethod(L, [q])
    lm.form_lagranges_equations()
    RHS = lm.rhs()
    assert RHS[1] == -g * sin(q) / l

def test_nonminimal_pendulum():
    if False:
        i = 10
        return i + 15
    (q1, q2) = dynamicsymbols('q1:3')
    (q1d, q2d) = dynamicsymbols('q1:3', level=1)
    (L, m, t) = symbols('L, m, t')
    g = 9.8
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    P = pN.locatenew('P1', q1 * N.x + q2 * N.y)
    P.set_vel(N, P.pos_from(pN).dt(N))
    pP = Particle('pP', P, m)
    f_c = Matrix([q1 ** 2 + q2 ** 2 - L ** 2])
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1, q2], hol_coneqs=f_c, forcelist=[(P, m * g * N.x)], frame=N)
    LM.form_lagranges_equations()
    lam1 = LM.lam_vec[0, 0]
    eom_sol = Matrix([[m * Derivative(q1, t, t) - 9.8 * m + 2 * lam1 * q1], [m * Derivative(q2, t, t) + 2 * lam1 * q2]])
    assert LM.eom == eom_sol
    lam_sol = Matrix([(19.6 * q1 + 2 * q1d ** 2 + 2 * q2d ** 2) / (4 * q1 ** 2 / m + 4 * q2 ** 2 / m)])
    assert simplify(LM.solve_multipliers(sol_type='Matrix')) == simplify(lam_sol)

def test_dub_pen():
    if False:
        for i in range(10):
            print('nop')
    (q1, q2) = dynamicsymbols('q1 q2')
    (q1d, q2d) = dynamicsymbols('q1 q2', 1)
    (q1dd, q2dd) = dynamicsymbols('q1 q2', 2)
    (u1, u2) = dynamicsymbols('u1 u2')
    (u1d, u2d) = dynamicsymbols('u1 u2', 1)
    (l, m, g) = symbols('l m g')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])
    A.set_ang_vel(N, q1d * A.z)
    B.set_ang_vel(N, q2d * A.z)
    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)
    O.set_vel(N, 0)
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)
    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)
    ParP.potential_energy = -m * g * l * cos(q1)
    ParR.potential_energy = -m * g * l * cos(q1) - m * g * l * cos(q2)
    L = Lagrangian(N, ParP, ParR)
    lm = LagrangesMethod(L, [q1, q2], bodies=[ParP, ParR])
    lm.form_lagranges_equations()
    assert simplify(l * m * (2 * g * sin(q1) + l * sin(q1) * sin(q2) * q2dd + l * sin(q1) * cos(q2) * q2d ** 2 - l * sin(q2) * cos(q1) * q2d ** 2 + l * cos(q1) * cos(q2) * q2dd + 2 * l * q1dd) - lm.eom[0]) == 0
    assert simplify(l * m * (g * sin(q2) + l * sin(q1) * sin(q2) * q1dd - l * sin(q1) * cos(q2) * q1d ** 2 + l * sin(q2) * cos(q1) * q1d ** 2 + l * cos(q1) * cos(q2) * q1dd + l * q2dd) - lm.eom[1]) == 0
    assert lm.bodies == [ParP, ParR]

def test_rolling_disc():
    if False:
        while True:
            i = 10
    (q1, q2, q3) = dynamicsymbols('q1 q2 q3')
    (q1d, q2d, q3d) = dynamicsymbols('q1 q2 q3', 1)
    (r, m, g) = symbols('r m g')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    I = inertia(L, m / 4 * r ** 2, m / 2 * r ** 2, m / 4 * r ** 2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyD.potential_energy = -m * g * r * cos(q2)
    Lag = Lagrangian(N, BodyD)
    q = [q1, q2, q3]
    q1 = Function('q1')
    q2 = Function('q2')
    q3 = Function('q3')
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()
    RHS = l.rhs()
    RHS.simplify()
    t = symbols('t')
    assert l.mass_matrix[3:6] == [0, 5 * m * r ** 2 / 4, 0]
    assert RHS[4].simplify() == (-8 * g * sin(q2(t)) + r * (5 * sin(2 * q2(t)) * Derivative(q1(t), t) + 12 * cos(q2(t)) * Derivative(q3(t), t)) * Derivative(q1(t), t)) / (10 * r)
    assert RHS[5] == (-5 * cos(q2(t)) * Derivative(q1(t), t) + 6 * tan(q2(t)) * Derivative(q3(t), t) + 4 * Derivative(q1(t), t) / cos(q2(t))) * Derivative(q2(t), t)