"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
SIM_LOOP = 500
MAX_SPEED = 50.0 / 3.6
MAX_ACCEL = 2.0
MAX_CURVATURE = 1.0
MAX_ROAD_WIDTH = 7.0
D_ROAD_W = 1.0
DT = 0.2
MAX_T = 5.0
MIN_T = 4.0
TARGET_SPEED = 30.0 / 3.6
D_T_S = 5.0 / 3.6
N_S_SAMPLE = 1
ROBOT_RADIUS = 2.0
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0
show_animation = True

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        if False:
            while True:
                i = 10
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        A = np.array([[3 * time ** 2, 4 * time ** 3], [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        if False:
            return 10
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4
        return xt

    def calc_first_derivative(self, t):
        if False:
            i = 10
            return i + 15
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
        return xt

    def calc_second_derivative(self, t):
        if False:
            return 10
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
        return xt

    def calc_third_derivative(self, t):
        if False:
            for i in range(10):
                print('nop')
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt

class FrenetPath:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    if False:
        while True:
            i = 10
    frenet_paths = []
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                Jp = sum(np.power(tfp.d_ddd, 2))
                Js = sum(np.power(tfp.s_ddd, 2))
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2
                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv
                frenet_paths.append(tfp)
    return frenet_paths

def calc_global_paths(fplist, csp):
    if False:
        i = 10
        return i + 15
    for fp in fplist:
        for i in range(len(fp.s)):
            (ix, iy) = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
    return fplist

def check_collision(fp, ob):
    if False:
        for i in range(10):
            print('nop')
    for i in range(len(ob[:, 0])):
        d = [(ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2 for (ix, iy) in zip(fp.x, fp.y)]
        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])
        if collision:
            return False
    return True

def check_paths(fplist, ob):
    if False:
        print('Hello World!')
    ok_ind = []
    for (i, _) in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):
            continue
        elif not check_collision(fplist[i], ob):
            continue
        ok_ind.append(i)
    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    if False:
        for i in range(10):
            print('nop')
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)
    min_cost = float('inf')
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    return best_path

def generate_target_course(x, y):
    if False:
        return 10
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    (rx, ry, ryaw, rk) = ([], [], [], [])
    for i_s in s:
        (ix, iy) = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    return (rx, ry, ryaw, rk, csp)

def main():
    if False:
        i = 10
        return i + 15
    print(__file__ + ' start!!')
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    ob = np.array([[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]])
    (tx, ty, tyaw, tc, csp) = generate_target_course(wx, wy)
    c_speed = 10.0 / 3.6
    c_accel = 0.0
    c_d = 2.0
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0
    area = 20.0
    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print('Goal')
            break
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], 'xk')
            plt.plot(path.x[1:], path.y[1:], '-or')
            plt.plot(path.x[1], path.y[1], 'vc')
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title('v[km/h]:' + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)
    print('Finish')
    if show_animation:
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()
if __name__ == '__main__':
    main()