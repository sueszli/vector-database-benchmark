"""

Graph based SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Ref

[A Tutorial on Graph-Based SLAM]
(http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import copy
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([0.1, np.deg2rad(10.0)]) ** 2
DT = 2.0
SIM_TIME = 100.0
MAX_RANGE = 30.0
STATE_SIZE = 3
C_SIGMA1 = 0.1
C_SIGMA2 = 0.1
C_SIGMA3 = np.deg2rad(1.0)
MAX_ITR = 20
show_graph_d_time = 20.0
show_animation = True

class Edge:

    def __init__(self):
        if False:
            return 10
        self.e = np.zeros((3, 1))
        self.omega = np.zeros((3, 3))
        self.d1 = 0.0
        self.d2 = 0.0
        self.yaw1 = 0.0
        self.yaw2 = 0.0
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.id1 = 0
        self.id2 = 0

def cal_observation_sigma():
    if False:
        print('Hello World!')
    sigma = np.zeros((3, 3))
    sigma[0, 0] = C_SIGMA1 ** 2
    sigma[1, 1] = C_SIGMA2 ** 2
    sigma[2, 2] = C_SIGMA3 ** 2
    return sigma

def calc_3d_rotational_matrix(angle):
    if False:
        for i in range(10):
            print('nop')
    return Rot.from_euler('z', angle).as_matrix()

def calc_edge(x1, y1, yaw1, x2, y2, yaw2, d1, angle1, d2, angle2, t1, t2):
    if False:
        return 10
    edge = Edge()
    tangle1 = pi_2_pi(yaw1 + angle1)
    tangle2 = pi_2_pi(yaw2 + angle2)
    tmp1 = d1 * math.cos(tangle1)
    tmp2 = d2 * math.cos(tangle2)
    tmp3 = d1 * math.sin(tangle1)
    tmp4 = d2 * math.sin(tangle2)
    edge.e[0, 0] = x2 - x1 - tmp1 + tmp2
    edge.e[1, 0] = y2 - y1 - tmp3 + tmp4
    edge.e[2, 0] = 0
    Rt1 = calc_3d_rotational_matrix(tangle1)
    Rt2 = calc_3d_rotational_matrix(tangle2)
    sig1 = cal_observation_sigma()
    sig2 = cal_observation_sigma()
    edge.omega = np.linalg.inv(Rt1 @ sig1 @ Rt1.T + Rt2 @ sig2 @ Rt2.T)
    (edge.d1, edge.d2) = (d1, d2)
    (edge.yaw1, edge.yaw2) = (yaw1, yaw2)
    (edge.angle1, edge.angle2) = (angle1, angle2)
    (edge.id1, edge.id2) = (t1, t2)
    return edge

def calc_edges(x_list, z_list):
    if False:
        return 10
    edges = []
    cost = 0.0
    z_ids = list(itertools.combinations(range(len(z_list)), 2))
    for (t1, t2) in z_ids:
        (x1, y1, yaw1) = (x_list[0, t1], x_list[1, t1], x_list[2, t1])
        (x2, y2, yaw2) = (x_list[0, t2], x_list[1, t2], x_list[2, t2])
        if z_list[t1] is None or z_list[t2] is None:
            continue
        for iz1 in range(len(z_list[t1][:, 0])):
            for iz2 in range(len(z_list[t2][:, 0])):
                if z_list[t1][iz1, 3] == z_list[t2][iz2, 3]:
                    d1 = z_list[t1][iz1, 0]
                    (angle1, _) = (z_list[t1][iz1, 1], z_list[t1][iz1, 2])
                    d2 = z_list[t2][iz2, 0]
                    (angle2, _) = (z_list[t2][iz2, 1], z_list[t2][iz2, 2])
                    edge = calc_edge(x1, y1, yaw1, x2, y2, yaw2, d1, angle1, d2, angle2, t1, t2)
                    edges.append(edge)
                    cost += (edge.e.T @ edge.omega @ edge.e)[0, 0]
    print('cost:', cost, ',n_edge:', len(edges))
    return edges

def calc_jacobian(edge):
    if False:
        print('Hello World!')
    t1 = edge.yaw1 + edge.angle1
    A = np.array([[-1.0, 0, edge.d1 * math.sin(t1)], [0, -1.0, -edge.d1 * math.cos(t1)], [0, 0, 0]])
    t2 = edge.yaw2 + edge.angle2
    B = np.array([[1.0, 0, -edge.d2 * math.sin(t2)], [0, 1.0, edge.d2 * math.cos(t2)], [0, 0, 0]])
    return (A, B)

def fill_H_and_b(H, b, edge):
    if False:
        for i in range(10):
            print('nop')
    (A, B) = calc_jacobian(edge)
    id1 = edge.id1 * STATE_SIZE
    id2 = edge.id2 * STATE_SIZE
    H[id1:id1 + STATE_SIZE, id1:id1 + STATE_SIZE] += A.T @ edge.omega @ A
    H[id1:id1 + STATE_SIZE, id2:id2 + STATE_SIZE] += A.T @ edge.omega @ B
    H[id2:id2 + STATE_SIZE, id1:id1 + STATE_SIZE] += B.T @ edge.omega @ A
    H[id2:id2 + STATE_SIZE, id2:id2 + STATE_SIZE] += B.T @ edge.omega @ B
    b[id1:id1 + STATE_SIZE] += A.T @ edge.omega @ edge.e
    b[id2:id2 + STATE_SIZE] += B.T @ edge.omega @ edge.e
    return (H, b)

def graph_based_slam(x_init, hz):
    if False:
        while True:
            i = 10
    print('start graph based slam')
    z_list = copy.deepcopy(hz)
    x_opt = copy.deepcopy(x_init)
    nt = x_opt.shape[1]
    n = nt * STATE_SIZE
    for itr in range(MAX_ITR):
        edges = calc_edges(x_opt, z_list)
        H = np.zeros((n, n))
        b = np.zeros((n, 1))
        for edge in edges:
            (H, b) = fill_H_and_b(H, b, edge)
        H[0:STATE_SIZE, 0:STATE_SIZE] += np.identity(STATE_SIZE)
        dx = -np.linalg.inv(H) @ b
        for i in range(nt):
            x_opt[0:3, i] += dx[i * 3:i * 3 + 3, 0]
        diff = (dx.T @ dx)[0, 0]
        print('iteration: %d, diff: %f' % (itr + 1, diff))
        if diff < 1e-05:
            break
    return x_opt

def calc_input():
    if False:
        return 10
    v = 1.0
    yaw_rate = 0.1
    u = np.array([[v, yaw_rate]]).T
    return u

def observation(xTrue, xd, u, RFID):
    if False:
        for i in range(10):
            print('nop')
    xTrue = motion_model(xTrue, u)
    z = np.zeros((0, 4))
    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx)) - xTrue[2, 0]
        phi = pi_2_pi(math.atan2(dy, dx))
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0]
            angle_noise = np.random.randn() * Q_sim[1, 1]
            angle += angle_noise
            phi += angle_noise
            zi = np.array([dn, angle, phi, i])
            z = np.vstack((z, zi))
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1]
    ud = np.array([[ud1, ud2]]).T
    xd = motion_model(xd, ud)
    return (xTrue, z, xd, ud)

def motion_model(x, u):
    if False:
        while True:
            i = 10
    F = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT]])
    x = F @ x + B @ u
    return x

def pi_2_pi(angle):
    if False:
        return 10
    return (angle + math.pi) % (2 * math.pi) - math.pi

def main():
    if False:
        print('Hello World!')
    print(__file__ + ' start!!')
    time = 0.0
    RFID = np.array([[10.0, -2.0, 0.0], [15.0, 10.0, 0.0], [3.0, 15.0, 0.0], [-5.0, 20.0, 0.0], [-5.0, 5.0, 0.0]])
    xTrue = np.zeros((STATE_SIZE, 1))
    xDR = np.zeros((STATE_SIZE, 1))
    hxTrue = []
    hxDR = []
    hz = []
    d_time = 0.0
    init = False
    while SIM_TIME >= time:
        if not init:
            hxTrue = xTrue
            hxDR = xTrue
            init = True
        else:
            hxDR = np.hstack((hxDR, xDR))
            hxTrue = np.hstack((hxTrue, xTrue))
        time += DT
        d_time += DT
        u = calc_input()
        (xTrue, z, xDR, ud) = observation(xTrue, xDR, u, RFID)
        hz.append(z)
        if d_time >= show_graph_d_time:
            x_opt = graph_based_slam(hxDR, hz)
            d_time = 0.0
            if show_animation:
                plt.cla()
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(RFID[:, 0], RFID[:, 1], '*k')
                plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), '-b')
                plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), '-k')
                plt.plot(x_opt[0, :].flatten(), x_opt[1, :].flatten(), '-r')
                plt.axis('equal')
                plt.grid(True)
                plt.title('Time' + str(time)[0:5])
                plt.pause(1.0)
if __name__ == '__main__':
    main()