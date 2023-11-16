from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, sin, cos
from sympy.matrices.dense import Matrix
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols, KanesMethod, inertia, Point, RigidBody, dot
from sympy.testing.pytest import slow

@slow
def test_bicycle():
    if False:
        while True:
            i = 10
    (q1, q2, q4, q5) = dynamicsymbols('q1 q2 q4 q5')
    (q1d, q2d, q4d, q5d) = dynamicsymbols('q1 q2 q4 q5', 1)
    (u1, u2, u3, u4, u5, u6) = dynamicsymbols('u1 u2 u3 u4 u5 u6')
    (u1d, u2d, u3d, u4d, u5d, u6d) = dynamicsymbols('u1 u2 u3 u4 u5 u6', 1)
    (WFrad, WRrad, htangle, forkoffset) = symbols('WFrad WRrad htangle forkoffset')
    (forklength, framelength, forkcg1) = symbols('forklength framelength forkcg1')
    (forkcg3, framecg1, framecg3, Iwr11) = symbols('forkcg3 framecg1 framecg3 Iwr11')
    (Iwr22, Iwf11, Iwf22, Iframe11) = symbols('Iwr22 Iwf11 Iwf22 Iframe11')
    (Iframe22, Iframe33, Iframe31, Ifork11) = symbols('Iframe22 Iframe33 Iframe31 Ifork11')
    (Ifork22, Ifork33, Ifork31, g) = symbols('Ifork22 Ifork33 Ifork31 g')
    (mframe, mfork, mwf, mwr) = symbols('mframe mfork mwf mwr')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    R = Y.orientnew('R', 'Axis', [q2, Y.x])
    Frame = R.orientnew('Frame', 'Axis', [q4 + htangle, R.y])
    WR = ReferenceFrame('WR')
    TempFrame = Frame.orientnew('TempFrame', 'Axis', [-htangle, Frame.y])
    Fork = Frame.orientnew('Fork', 'Axis', [q5, Frame.x])
    TempFork = Fork.orientnew('TempFork', 'Axis', [-htangle, Fork.y])
    WF = ReferenceFrame('WF')
    WR_cont = Point('WR_cont')
    WR_mc = WR_cont.locatenew('WR_mc', WRrad * R.z)
    Steer = WR_mc.locatenew('Steer', framelength * Frame.z)
    Frame_mc = WR_mc.locatenew('Frame_mc', -framecg1 * Frame.x + framecg3 * Frame.z)
    Fork_mc = Steer.locatenew('Fork_mc', -forkcg1 * Fork.x + forkcg3 * Fork.z)
    WF_mc = Steer.locatenew('WF_mc', forklength * Fork.x + forkoffset * Fork.z)
    WF_cont = WF_mc.locatenew('WF_cont', WFrad * (dot(Fork.y, Y.z) * Fork.y - Y.z).normalize())
    Y.set_ang_vel(N, u1 * Y.z)
    R.set_ang_vel(Y, u2 * R.x)
    WR.set_ang_vel(Frame, u3 * Frame.y)
    Frame.set_ang_vel(R, u4 * Frame.y)
    Fork.set_ang_vel(Frame, u5 * Fork.x)
    WF.set_ang_vel(Fork, u6 * Fork.y)
    WR_cont.set_vel(N, 0)
    WR_mc.v2pt_theory(WR_cont, N, WR)
    Steer.v2pt_theory(WR_mc, N, Frame)
    Frame_mc.v2pt_theory(WR_mc, N, Frame)
    Fork_mc.v2pt_theory(Steer, N, Fork)
    WF_mc.v2pt_theory(Steer, N, Fork)
    WF_cont.v2pt_theory(WF_mc, N, WF)
    Frame_I = (inertia(TempFrame, Iframe11, Iframe22, Iframe33, 0, 0, Iframe31), Frame_mc)
    Fork_I = (inertia(TempFork, Ifork11, Ifork22, Ifork33, 0, 0, Ifork31), Fork_mc)
    WR_I = (inertia(Frame, Iwr11, Iwr22, Iwr11), WR_mc)
    WF_I = (inertia(Fork, Iwf11, Iwf22, Iwf11), WF_mc)
    BodyFrame = RigidBody('BodyFrame', Frame_mc, Frame, mframe, Frame_I)
    BodyFork = RigidBody('BodyFork', Fork_mc, Fork, mfork, Fork_I)
    BodyWR = RigidBody('BodyWR', WR_mc, WR, mwr, WR_I)
    BodyWF = RigidBody('BodyWF', WF_mc, WF, mwf, WF_I)
    kd = [q1d - u1, q2d - u2, q4d - u4, q5d - u5]
    conlist_speed = [WF_cont.vel(N) & Y.x, WF_cont.vel(N) & Y.y, WF_cont.vel(N) & Y.z]
    conlist_coord = [WF_cont.pos_from(WR_cont) & Y.z]
    FL = [(Frame_mc, -mframe * g * Y.z), (Fork_mc, -mfork * g * Y.z), (WF_mc, -mwf * g * Y.z), (WR_mc, -mwr * g * Y.z)]
    BL = [BodyFrame, BodyFork, BodyWR, BodyWF]
    KM = KanesMethod(N, q_ind=[q1, q2, q5], q_dependent=[q4], configuration_constraints=conlist_coord, u_ind=[u2, u3, u5], u_dependent=[u1, u4, u6], velocity_constraints=conlist_speed, kd_eqs=kd, constraint_solver='CRAMER')
    (fr, frstar) = KM.kanes_equations(BL, FL)
    PaperRadRear = 0.3
    PaperRadFront = 0.35
    HTA = (pi / 2 - pi / 10).evalf()
    TrailPaper = 0.08
    rake = (-(TrailPaper * sin(HTA) - PaperRadFront * cos(HTA))).evalf()
    PaperWb = 1.02
    PaperFrameCgX = 0.3
    PaperFrameCgZ = 0.9
    PaperForkCgX = 0.9
    PaperForkCgZ = 0.7
    FrameLength = (PaperWb * sin(HTA) - (rake - (PaperRadFront - PaperRadRear) * cos(HTA))).evalf()
    FrameCGNorm = ((PaperFrameCgZ - PaperRadRear - PaperFrameCgX / sin(HTA) * cos(HTA)) * sin(HTA)).evalf()
    FrameCGPar = (PaperFrameCgX / sin(HTA) + (PaperFrameCgZ - PaperRadRear - PaperFrameCgX / sin(HTA) * cos(HTA)) * cos(HTA)).evalf()
    tempa = PaperForkCgZ - PaperRadFront
    tempb = PaperWb - PaperForkCgX
    tempc = sqrt(tempa ** 2 + tempb ** 2).evalf()
    PaperForkL = (PaperWb * cos(HTA) - (PaperRadFront - PaperRadRear) * sin(HTA)).evalf()
    ForkCGNorm = (rake + tempc * sin(pi / 2 - HTA - acos(tempa / tempc))).evalf()
    ForkCGPar = (tempc * cos(pi / 2 - HTA - acos(tempa / tempc)) - PaperForkL).evalf()
    v = symbols('v')
    val_dict = {WFrad: PaperRadFront, WRrad: PaperRadRear, htangle: HTA, forkoffset: rake, forklength: PaperForkL, framelength: FrameLength, forkcg1: ForkCGPar, forkcg3: ForkCGNorm, framecg1: FrameCGNorm, framecg3: FrameCGPar, Iwr11: 0.0603, Iwr22: 0.12, Iwf11: 0.1405, Iwf22: 0.28, Ifork11: 0.05892, Ifork22: 0.06, Ifork33: 0.00708, Ifork31: 0.00756, Iframe11: 9.2, Iframe22: 11, Iframe33: 2.8, Iframe31: -2.4, mfork: 4, mframe: 85, mwf: 3, mwr: 2, g: 9.81, q1: 0, q2: 0, q4: 0, q5: 0, u1: 0, u2: 0, u3: v / PaperRadRear, u4: 0, u5: 0, u6: v / PaperRadFront}
    (A, B, _) = KM.linearize(A_and_B=True, op_point={u1.diff(): 0, u2.diff(): 0, u3.diff(): 0, u4.diff(): 0, u5.diff(): 0, u6.diff(): 0, u1: 0, u2: 0, u3: v / PaperRadRear, u4: 0, u5: 0, u6: v / PaperRadFront, q1: 0, q2: 0, q4: 0, q5: 0}, linear_solver='CRAMER')
    A_s = A.xreplace(val_dict)
    B_s = B.xreplace(val_dict)
    A_s = A_s.evalf()
    B_s = B_s.evalf()
    A = A_s.extract([1, 2, 3, 5], [1, 2, 3, 5])
    Res = Matrix([[0, 0, 1.0, 0], [0, 0, 0, 1.0], [9.48977444677355, -0.891197738059089 * v ** 2 - 0.571523173729245, -0.105522449805691 * v, -0.330515398992311 * v], [11.7194768719633, -1.97171508499972 * v ** 2 + 30.9087533932407, 3.67680523332152 * v, -3.08486552743311 * v]])
    eps = 1e-12
    for i in range(6):
        error = Res.subs(v, i) - A.subs(v, i)
        assert all((abs(x) < eps for x in error))