"""

Ensemble Kalman Filter(EnKF) localization sample

author: Ryohei Sasaki(rsasaki0109)

Ref:
Ensemble Kalman filtering
(https://rmets.onlinelibrary.wiley.com/doi/10.1256/qj.05.135)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.angle import rot_mat_2d
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2
DT = 0.1
SIM_TIME = 50.0
MAX_RANGE = 20.0
NP = 20
show_animation = True

def calc_input():
    if False:
        return 10
    v = 1.0
    yaw_rate = 0.1
    u = np.array([[v, yaw_rate]]).T
    return u

def observation(xTrue, xd, u, RFID):
    if False:
        print('Hello World!')
    xTrue = motion_model(xTrue, u)
    z = np.zeros((0, 4))
    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5
            angle_with_noise = angle + np.random.randn() * Q_sim[1, 1] ** 0.5
            zi = np.array([dn, angle_with_noise, RFID[i, 0], RFID[i, 1]])
            z = np.vstack((z, zi))
    ud = np.array([[u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5, u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T
    xd = motion_model(xd, ud)
    return (xTrue, z, xd, ud)

def motion_model(x, u):
    if False:
        while True:
            i = 10
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT], [1.0, 0.0]])
    x = F.dot(x) + B.dot(u)
    return x

def observe_landmark_position(x, landmarks):
    if False:
        while True:
            i = 10
    landmarks_pos = np.zeros((2 * landmarks.shape[0], 1))
    for (i, lm) in enumerate(landmarks):
        index = 2 * i
        q = Q_sim[0, 0] ** 0.5
        landmarks_pos[index] = x[0, 0] + lm[0] * math.cos(x[2, 0] + lm[1]) + np.random.randn() * q / np.sqrt(2)
        landmarks_pos[index + 1] = x[1, 0] + lm[0] * math.sin(x[2, 0] + lm[1]) + np.random.randn() * q / np.sqrt(2)
    return landmarks_pos

def calc_covariance(xEst, px):
    if False:
        while True:
            i = 10
    cov = np.zeros((3, 3))
    for i in range(px.shape[1]):
        dx = (px[:, i] - xEst)[0:3]
        cov += dx.dot(dx.T)
    cov /= NP
    return cov

def enkf_localization(px, z, u):
    if False:
        return 10
    '\n    Localization with Ensemble Kalman filter\n    '
    pz = np.zeros((z.shape[0] * 2, NP))
    for ip in range(NP):
        x = np.array([px[:, ip]]).T
        ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud)
        px[:, ip] = x[:, 0]
        z_pos = observe_landmark_position(x, z)
        pz[:, ip] = z_pos[:, 0]
    x_ave = np.mean(px, axis=1)
    x_dif = px - np.tile(x_ave, (NP, 1)).T
    z_ave = np.mean(pz, axis=1)
    z_dif = pz - np.tile(z_ave, (NP, 1)).T
    U = 1 / (NP - 1) * x_dif @ z_dif.T
    V = 1 / (NP - 1) * z_dif @ z_dif.T
    K = U @ np.linalg.inv(V)
    z_lm_pos = z[:, [2, 3]].reshape(-1)
    px_hat = px + K @ (np.tile(z_lm_pos, (NP, 1)).T - pz)
    xEst = np.average(px_hat, axis=1).reshape(4, 1)
    PEst = calc_covariance(xEst, px_hat)
    return (xEst, PEst, px_hat)

def plot_covariance_ellipse(xEst, PEst):
    if False:
        i = 10
        return i + 15
    Pxy = PEst[0:2, 0:2]
    (eig_val, eig_vec) = np.linalg.eig(Pxy)
    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0
    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    fx = np.stack([x, y]).T @ rot_mat_2d(angle)
    px = np.array(fx[:, 0] + xEst[0, 0]).flatten()
    py = np.array(fx[:, 1] + xEst[1, 0]).flatten()
    plt.plot(px, py, '--r')

def pi_2_pi(angle):
    if False:
        print('Hello World!')
    return (angle + math.pi) % (2 * math.pi) - math.pi

def main():
    if False:
        i = 10
        return i + 15
    print(__file__ + ' start!!')
    time = 0.0
    RF_ID = np.array([[10.0, 0.0], [10.0, 10.0], [0.0, 15.0], [-5.0, 20.0]])
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    px = np.zeros((4, NP))
    xDR = np.zeros((4, 1))
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        (xTrue, z, xDR, ud) = observation(xTrue, xDR, u, RF_ID)
        (xEst, PEst, px) = enkf_localization(px, z, ud)
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(len(z[:, 0])):
                plt.plot([xTrue[0, 0], z[i, 2]], [xTrue[1, 0], z[i, 3]], '-k')
            plt.plot(RF_ID[:, 0], RF_ID[:, 1], '*k')
            plt.plot(px[0, :], px[1, :], '.r')
            plt.plot(np.array(hxTrue[0, :]).flatten(), np.array(hxTrue[1, :]).flatten(), '-b')
            plt.plot(np.array(hxDR[0, :]).flatten(), np.array(hxDR[1, :]).flatten(), '-k')
            plt.plot(np.array(hxEst[0, :]).flatten(), np.array(hxEst[1, :]).flatten(), '-r')
            plot_covariance_ellipse(xEst, PEst)
            plt.axis('equal')
            plt.grid(True)
            plt.pause(0.001)
if __name__ == '__main__':
    main()