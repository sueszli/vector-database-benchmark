from sympy import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import cross, dot, dynamicsymbols, find_dynamicsymbols, KanesMethod, inertia, inertia_of_point_mass, Point, ReferenceFrame, RigidBody

def test_aux_dep():
    if False:
        while True:
            i = 10
    (t, r, m, g, I, J) = symbols('t r m g I J')
    (Fx, Fy, Fz) = symbols('Fx Fy Fz')
    q = dynamicsymbols('q:4')
    qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    ud = [ui.diff(t) for ui in u]
    ud_zero = dict(zip(ud, [0.0] * len(ud)))
    ua = dynamicsymbols('ua:3')
    ua_zero = dict(zip(ua, [0.0] * len(ua)))
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q[0], N.z])
    B = A.orientnew('B', 'Axis', [q[1], A.x])
    C = B.orientnew('C', 'Axis', [q[2], B.y])
    C.set_ang_vel(N, u[0] * B.x + u[1] * B.y + u[2] * B.z)
    C.set_ang_acc(N, C.ang_vel_in(N).diff(t, B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))
    P = Point('P')
    P.set_vel(N, ua[0] * A.x + ua[1] * A.y + ua[2] * A.z)
    O = P.locatenew('O', q[3] * A.z + r * sin(q[1]) * A.y)
    O.set_vel(N, u[3] * A.x + u[4] * A.y + u[5] * A.z)
    O.set_acc(N, O.vel(N).diff(t, A) + cross(A.ang_vel_in(N), O.vel(N)))
    w_c_n_qd = qd[0] * A.z + qd[1] * B.x + qd[2] * B.y
    v_o_n_qd = O.pos_from(P).diff(t, A) + cross(A.ang_vel_in(N), O.pos_from(P))
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] + [dot(v_o_n_qd - O.vel(N), A.z)])
    qd_kd = solve(kindiffs, qd)
    steady_conditions = solve(kindiffs.subs({qd[1]: 0, qd[3]: 0}), u)
    steady_conditions.update({qd[1]: 0, qd[3]: 0})
    partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u + ua]
    partial_v_O = [O.vel(N).diff(ui, N) for ui in u + ua]
    partial_v_P = [P.vel(N).diff(ui, N) for ui in u + ua]
    f_c = Matrix([dot(-r * B.z, A.z) - q[3]])
    f_v = Matrix([dot(O.vel(N) - (P.vel(N) + cross(C.ang_vel_in(N), O.pos_from(P))), ai).expand() for ai in A])
    v_o_n = cross(C.ang_vel_in(N), O.pos_from(P))
    a_o_n = v_o_n.diff(t, A) + cross(A.ang_vel_in(N), v_o_n)
    f_a = Matrix([dot(O.acc(N) - a_o_n, ai) for ai in A])
    M_v = zeros(3, 9)
    for i in range(3):
        for (j, ui) in enumerate(u + ua):
            M_v[i, j] = f_v[i].diff(ui)
    M_v_i = M_v[:, :3]
    M_v_d = M_v[:, 3:6]
    M_v_aux = M_v[:, 6:]
    M_v_i_aux = M_v_i.row_join(M_v_aux)
    A_rs = -M_v_d.inv() * M_v_i_aux
    u_dep = A_rs[:, :3] * Matrix(u[:3])
    u_dep_dict = dict(zip(u[3:], u_dep))
    F_O = m * g * A.z
    F_P = Fx * A.x + Fy * A.y + Fz * A.z
    Fr_u = Matrix([dot(F_O, pv_o) + dot(F_P, pv_p) for (pv_o, pv_p) in zip(partial_v_O, partial_v_P)])
    R_star_O = -m * O.acc(N)
    I_C_O = inertia(B, I, J, I)
    T_star_C = -(dot(I_C_O, C.ang_acc_in(N)) + cross(C.ang_vel_in(N), dot(I_C_O, C.ang_vel_in(N))))
    Fr_star_u = Matrix([dot(R_star_O, pv) + dot(T_star_C, pav) for (pv, pav) in zip(partial_v_O, partial_w_C)])
    Fr_c = Fr_u[:3, :].col_join(Fr_u[6:, :]) + A_rs.T * Fr_u[3:6, :]
    Fr_star_c = Fr_star_u[:3, :].col_join(Fr_star_u[6:, :]) + A_rs.T * Fr_star_u[3:6, :]
    Fr_star_steady = Fr_star_c.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions).subs({q[3]: -r * cos(q[1])}).expand()
    iner_tuple = (I_C_O, O)
    disc = RigidBody('disc', O, C, m, iner_tuple)
    bodyList = [disc]
    F_o = (O, F_O)
    F_p = (P, F_P)
    forceList = [F_o, F_p]
    kane = KanesMethod(N, q_ind=q[:3], u_ind=u[:3], kd_eqs=kindiffs, q_dependent=q[3:], configuration_constraints=f_c, u_dependent=u[3:], velocity_constraints=f_v, u_auxiliary=ua)
    (fr, frstar) = kane.kanes_equations(bodyList, forceList)
    frstar_steady = frstar.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions).subs({q[3]: -r * cos(q[1])}).expand()
    kdd = kane.kindiffdict()
    assert Matrix(Fr_c).expand() == fr.expand()
    assert Matrix(Fr_star_c.subs(kdd)).expand() == frstar.expand()
    assert simplify(Matrix(Fr_star_steady).expand()).xreplace({0: 0.0}) == simplify(frstar_steady.expand()).xreplace({0: 0.0})
    syms_in_forcing = find_dynamicsymbols(kane.forcing)
    for qdi in qd:
        assert qdi not in syms_in_forcing

def test_non_central_inertia():
    if False:
        while True:
            i = 10
    (q1, q2, q3) = dynamicsymbols('q1:4')
    (q1d, q2d, q3d) = dynamicsymbols('q1:4', level=1)
    (u1, u2, u3, u4, u5) = dynamicsymbols('u1:6')
    (u_prime, R, M, g, e, f, theta) = symbols("u' R, M, g, e, f, theta")
    (a, b, mA, mB, IA, J, K, t) = symbols('a b mA mB IA J K t')
    (Q1, Q2, Q3) = symbols('Q1, Q2 Q3')
    (IA22, IA23, IA33) = symbols('IA22 IA23 IA33')
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1 * A.x + u3 * A.z)
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    B.set_ang_vel(A, u4 * A.z)
    C.set_ang_vel(A, u5 * A.z)
    pD = Point('D')
    pD.set_vel(A, 0)
    pD.set_vel(F, u2 * A.y)
    pS_star = pD.locatenew('S*', e * A.y)
    pQ = pD.locatenew('Q', f * A.y - R * A.x)
    for p in [pS_star, pQ]:
        p.v2pt_theory(pD, F, A)
    pA_star = pD.locatenew('A*', a * A.y)
    pB_star = pD.locatenew('B*', b * A.z)
    pC_star = pD.locatenew('C*', -b * A.z)
    for p in [pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)
    pB_hat = pB_star.locatenew('B^', -R * A.x)
    pC_hat = pC_star.locatenew('C^', -R * A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)
    kde = [q1d - u1, q2d - u4, q3d - u5]
    vc = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))
    km = KanesMethod(F, q_ind=[q1, q2, q3], u_ind=[u1, u2], kd_eqs=kde, u_dependent=[u4, u5], velocity_constraints=vc, u_auxiliary=[u3])
    forces = [(pS_star, -M * g * F.x), (pQ, Q1 * A.x + Q2 * A.y + Q3 * A.z)]
    bodies = [rbA, rbB, rbC]
    (fr, fr_star) = km.kanes_equations(bodies, forces)
    vc_map = solve(vc, [u4, u5])
    fr_star_expected = Matrix([-(IA + 2 * J * b ** 2 / R ** 2 + 2 * K + mA * a ** 2 + 2 * mB * b ** 2) * u1.diff(t) - mA * a * u1 * u2, -(mA + 2 * mB + 2 * J / R ** 2) * u2.diff(t) + mA * a * u1 ** 2, 0])
    t = trigsimp(fr_star.subs(vc_map).subs({u3: 0})).doit().expand()
    assert (fr_star_expected - t).expand() == zeros(3, 1)
    bodies2 = []
    for (rb, I_star) in zip([rbA, rbB, rbC], [inertia_A, inertia_B, inertia_C]):
        I = I_star + inertia_of_point_mass(rb.mass, rb.masscenter.pos_from(pD), rb.frame)
        bodies2.append(RigidBody('', rb.masscenter, rb.frame, rb.mass, (I, pD)))
    (fr2, fr_star2) = km.kanes_equations(bodies2, forces)
    t = trigsimp(fr_star2.subs(vc_map).subs({u3: 0})).doit()
    assert (fr_star_expected - t).expand() == zeros(3, 1)

def test_sub_qdot():
    if False:
        return 10
    (q1, q2, q3) = dynamicsymbols('q1:4')
    (q1d, q2d, q3d) = dynamicsymbols('q1:4', level=1)
    (u1, u2, u3) = dynamicsymbols('u1:4')
    (u_prime, R, M, g, e, f, theta) = symbols("u' R, M, g, e, f, theta")
    (a, b, mA, mB, IA, J, K, t) = symbols('a b mA mB IA J K t')
    (IA22, IA23, IA33) = symbols('IA22 IA23 IA33')
    (Q1, Q2, Q3) = symbols('Q1 Q2 Q3')
    F = ReferenceFrame('F')
    P = F.orientnew('P', 'axis', [-theta, F.y])
    A = P.orientnew('A', 'axis', [q1, P.x])
    A.set_ang_vel(F, u1 * A.x + u3 * A.z)
    B = A.orientnew('B', 'axis', [q2, A.z])
    C = A.orientnew('C', 'axis', [q3, A.z])
    pD = Point('D')
    pD.set_vel(A, 0)
    pD.set_vel(F, u2 * A.y)
    pS_star = pD.locatenew('S*', e * A.y)
    pQ = pD.locatenew('Q', f * A.y - R * A.x)
    pA_star = pD.locatenew('A*', a * A.y)
    pB_star = pD.locatenew('B*', b * A.z)
    pC_star = pD.locatenew('C*', -b * A.z)
    for p in [pS_star, pQ, pA_star, pB_star, pC_star]:
        p.v2pt_theory(pD, F, A)
    pB_hat = pB_star.locatenew('B^', -R * A.x)
    pC_hat = pC_star.locatenew('C^', -R * A.x)
    pB_hat.v2pt_theory(pB_star, F, B)
    pC_hat.v2pt_theory(pC_star, F, C)
    kde = [dot(p.vel(F), A.y) for p in [pB_hat, pC_hat]]
    kde += [u1 - q1d]
    kde_map = solve(kde, [q1d, q2d, q3d])
    for (k, v) in list(kde_map.items()):
        kde_map[k.diff(t)] = v.diff(t)
    inertia_A = inertia(A, IA, IA22, IA33, 0, IA23, 0)
    inertia_B = inertia(B, K, K, J)
    inertia_C = inertia(C, K, K, J)
    rbA = RigidBody('rbA', pA_star, A, mA, (inertia_A, pA_star))
    rbB = RigidBody('rbB', pB_star, B, mB, (inertia_B, pB_star))
    rbC = RigidBody('rbC', pC_star, C, mB, (inertia_C, pC_star))
    km = KanesMethod(F, [q1, q2, q3], [u1, u2], kd_eqs=kde, u_auxiliary=[u3])
    forces = [(pS_star, -M * g * F.x), (pQ, Q1 * A.x + Q2 * A.y + Q3 * A.z)]
    bodies = [rbA, rbB, rbC]
    fr_expected = Matrix([f * Q3 + M * g * e * sin(theta) * cos(q1), Q2 + M * g * sin(theta) * sin(q1), e * M * g * cos(theta) - Q1 * f - Q2 * R])
    fr_star_expected = Matrix([-(IA + 2 * J * b ** 2 / R ** 2 + 2 * K + mA * a ** 2 + 2 * mB * b ** 2) * u1.diff(t) - mA * a * u1 * u2, -(mA + 2 * mB + 2 * J / R ** 2) * u2.diff(t) + mA * a * u1 ** 2, 0])
    (fr, fr_star) = km.kanes_equations(bodies, forces)
    assert fr.expand() == fr_expected.expand()
    assert (fr_star_expected - trigsimp(fr_star)).expand() == zeros(3, 1)

def test_sub_qdot2():
    if False:
        return 10
    (g, m, Px, Py, Pz, R, t) = symbols('g m Px Py Pz R t')
    q = dynamicsymbols('q:5')
    qd = dynamicsymbols('q:5', level=1)
    u = dynamicsymbols('u:5')
    A = ReferenceFrame('A')
    B_prime = A.orientnew('B_prime', 'Axis', [q[0], A.z])
    B = B_prime.orientnew('B', 'Axis', [pi / 2 - q[1], B_prime.x])
    C = B.orientnew('C', 'Axis', [q[2], B.z])
    pO = Point('O')
    pO.set_vel(A, 0)
    pR = pO.locatenew('R', q[3] * A.x + q[4] * A.y)
    pR.set_vel(A, pR.pos_from(pO).diff(t, A))
    pR.set_vel(B, 0)
    pC_hat = pR.locatenew('C^', 0)
    pC_hat.set_vel(C, 0)
    pCs = pC_hat.locatenew('C*', R * B.y)
    pCs.set_vel(C, 0)
    pCs.set_vel(B, 0)
    pCs.v2pt_theory(pR, A, B)
    pC_hat.v2pt_theory(pCs, A, C)
    R_C_hat = Px * A.x + Py * A.y + Pz * A.z
    R_Cs = -m * g * A.z
    forces = [(pC_hat, R_C_hat), (pCs, R_Cs)]
    u_expr = [C.ang_vel_in(A) & uv for uv in B]
    u_expr += qd[3:]
    kde = [ui - e for (ui, e) in zip(u, u_expr)]
    km1 = KanesMethod(A, q, u, kde)
    (fr1, _) = km1.kanes_equations([], forces)
    u_indep = u[:3]
    u_dep = list(set(u) - set(u_indep))
    vc = [pC_hat.vel(A) & uv for uv in [A.x, A.y]]
    km2 = KanesMethod(A, q, u_indep, kde, u_dependent=u_dep, velocity_constraints=vc)
    (fr2, _) = km2.kanes_equations([], forces)
    fr1_expected = Matrix([-R * g * m * sin(q[1]), -R * (Px * cos(q[0]) + Py * sin(q[0])) * tan(q[1]), R * (Px * cos(q[0]) + Py * sin(q[0])), Px, Py])
    fr2_expected = Matrix([-R * g * m * sin(q[1]), 0, 0])
    assert trigsimp(fr1.expand()) == trigsimp(fr1_expected.expand())
    assert trigsimp(fr2.expand()) == trigsimp(fr2_expected.expand())