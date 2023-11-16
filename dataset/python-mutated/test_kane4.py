from sympy import cos, sin, Matrix, symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, KanesMethod, Particle

def test_replace_qdots_in_force():
    if False:
        while True:
            i = 10
    (q1, q2) = dynamicsymbols('q1, q2')
    (qd1, qd2) = dynamicsymbols('q1, q2', level=1)
    (u1, u2) = dynamicsymbols('u1, u2')
    (l, m) = symbols('l, m')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', (q1, N.z))
    B = A.orientnew('B', 'Axis', (q2, N.z))
    O = Point('O')
    O.set_vel(N, 0)
    P = O.locatenew('P', l * A.x)
    P.v2pt_theory(O, N, A)
    Q = P.locatenew('Q', l * B.x)
    Q.v2pt_theory(P, N, B)
    Ap = Particle('Ap', P, m)
    Bp = Particle('Bp', Q, m)
    (sig, delta) = symbols('sigma, delta')
    Ta = (sig * q2 + delta * qd2) * N.z
    forces = [(A, Ta), (B, -Ta)]
    kde1 = [u1 - qd1, u2 - qd2]
    kde2 = [u1 - qd1, u2 - (qd1 + qd2)]
    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    (fr1, fstar1) = KM1.kanes_equations([Ap, Bp], forces)
    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    (fr2, fstar2) = KM2.kanes_equations([Ap, Bp], forces)
    forcing_matrix_expected = Matrix([[m * l ** 2 * sin(q2) * u2 ** 2 + sig * q2 + delta * (u2 - u1)], [m * l ** 2 * sin(q2) * -u1 ** 2 - sig * q2 - delta * (u2 - u1)]])
    mass_matrix_expected = Matrix([[2 * m * l ** 2, m * l ** 2 * cos(q2)], [m * l ** 2 * cos(q2), m * l ** 2]])
    assert KM2.mass_matrix.expand() == mass_matrix_expected.expand()
    assert KM2.forcing.expand() == forcing_matrix_expected.expand()
    fr1_expected = Matrix([0, -(sig * q2 + delta * u2)])
    assert fr1.expand() == fr1_expected.expand()
    fr2_expected = Matrix([sig * q2 + delta * (u2 - u1), -sig * q2 - delta * (u2 - u1)])
    assert fr2.expand() == fr2_expected.expand()
    Ta = (sig * q2 + delta * u2) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    (fr1, fstar1) = KM1.kanes_equations([Ap, Bp], forces)
    assert fr1.expand() == fr1_expected.expand()
    Ta = (sig * q2 + delta * (u2 - u1)) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    (fr2, fstar2) = KM2.kanes_equations([Ap, Bp], forces)
    assert fr2.expand() == fr2_expected.expand()
    Ta = (sig * q2 + delta * qd2 ** 3) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    (fr1, fstar1) = KM1.kanes_equations([Ap, Bp], forces)
    fr1_cubic_expected = Matrix([0, -(sig * q2 + delta * u2 ** 3)])
    assert fr1.expand() == fr1_cubic_expected.expand()
    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    (fr2, fstar2) = KM2.kanes_equations([Ap, Bp], forces)
    fr2_cubic_expected = Matrix([sig * q2 + delta * (u2 - u1) ** 3, -sig * q2 - delta * (u2 - u1) ** 3])
    assert fr2.expand() == fr2_cubic_expected.expand()