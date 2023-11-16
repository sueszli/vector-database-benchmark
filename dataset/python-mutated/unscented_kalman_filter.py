"""

Unscented kalman filter (UKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from utils.angle import rot_mat_2d
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
R = np.diag([1.0, 1.0]) ** 2
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
DT = 0.1
SIM_TIME = 50.0
ALPHA = 0.001
BETA = 2
KAPPA = 0
show_animation = True

def calc_input():
    if False:
        return 10
    v = 1.0
    yawRate = 0.1
    u = np.array([[v, yawRate]]).T
    return u

def observation(xTrue, xd, u):
    if False:
        print('Hello World!')
    xTrue = motion_model(xTrue, u)
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    xd = motion_model(xd, ud)
    return (xTrue, z, xd, ud)

def motion_model(x, u):
    if False:
        for i in range(10):
            print('nop')
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT], [1.0, 0.0]])
    x = F @ x + B @ u
    return x

def observation_model(x):
    if False:
        i = 10
        return i + 15
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    z = H @ x
    return z

def generate_sigma_points(xEst, PEst, gamma):
    if False:
        print('Hello World!')
    sigma = xEst
    Psqrt = scipy.linalg.sqrtm(PEst)
    n = len(xEst[:, 0])
    for i in range(n):
        sigma = np.hstack((sigma, xEst + gamma * Psqrt[:, i:i + 1]))
    for i in range(n):
        sigma = np.hstack((sigma, xEst - gamma * Psqrt[:, i:i + 1]))
    return sigma

def predict_sigma_motion(sigma, u):
    if False:
        return 10
    '\n        Sigma Points prediction with motion model\n    '
    for i in range(sigma.shape[1]):
        sigma[:, i:i + 1] = motion_model(sigma[:, i:i + 1], u)
    return sigma

def predict_sigma_observation(sigma):
    if False:
        for i in range(10):
            print('nop')
    '\n        Sigma Points prediction with observation model\n    '
    for i in range(sigma.shape[1]):
        sigma[0:2, i] = observation_model(sigma[:, i])
    sigma = sigma[0:2, :]
    return sigma

def calc_sigma_covariance(x, sigma, wc, Pi):
    if False:
        for i in range(10):
            print('nop')
    nSigma = sigma.shape[1]
    d = sigma - x[0:sigma.shape[0]]
    P = Pi
    for i in range(nSigma):
        P = P + wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
    return P

def calc_pxz(sigma, x, z_sigma, zb, wc):
    if False:
        for i in range(10):
            print('nop')
    nSigma = sigma.shape[1]
    dx = sigma - x
    dz = z_sigma - zb[0:2]
    P = np.zeros((dx.shape[0], dz.shape[0]))
    for i in range(nSigma):
        P = P + wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T
    return P

def ukf_estimation(xEst, PEst, z, u, wm, wc, gamma):
    if False:
        while True:
            i = 10
    sigma = generate_sigma_points(xEst, PEst, gamma)
    sigma = predict_sigma_motion(sigma, u)
    xPred = (wm @ sigma.T).T
    PPred = calc_sigma_covariance(xPred, sigma, wc, Q)
    zPred = observation_model(xPred)
    y = z - zPred
    sigma = generate_sigma_points(xPred, PPred, gamma)
    zb = (wm @ sigma.T).T
    z_sigma = predict_sigma_observation(sigma)
    st = calc_sigma_covariance(zb, z_sigma, wc, R)
    Pxz = calc_pxz(sigma, xPred, z_sigma, zb, wc)
    K = Pxz @ np.linalg.inv(st)
    xEst = xPred + K @ y
    PEst = PPred - K @ st @ K.T
    return (xEst, PEst)

def plot_covariance_ellipse(xEst, PEst):
    if False:
        while True:
            i = 10
    Pxy = PEst[0:2, 0:2]
    (eigval, eigvec) = np.linalg.eig(Pxy)
    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ np.array([x, y])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, '--r')

def setup_ukf(nx):
    if False:
        for i in range(10):
            print('nop')
    lamb = ALPHA ** 2 * (nx + KAPPA) - nx
    wm = [lamb / (lamb + nx)]
    wc = [lamb / (lamb + nx) + (1 - ALPHA ** 2 + BETA)]
    for i in range(2 * nx):
        wm.append(1.0 / (2 * (nx + lamb)))
        wc.append(1.0 / (2 * (nx + lamb)))
    gamma = math.sqrt(nx + lamb)
    wm = np.array([wm])
    wc = np.array([wc])
    return (wm, wc, gamma)

def main():
    if False:
        for i in range(10):
            print('nop')
    print(__file__ + ' start!!')
    nx = 4
    xEst = np.zeros((nx, 1))
    xTrue = np.zeros((nx, 1))
    PEst = np.eye(nx)
    xDR = np.zeros((nx, 1))
    (wm, wc, gamma) = setup_ukf(nx)
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    time = 0.0
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        (xTrue, z, xDR, ud) = observation(xTrue, xDR, u)
        (xEst, PEst) = ukf_estimation(xEst, PEst, z, ud, wm, wc, gamma)
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], '.g')
            plt.plot(np.array(hxTrue[0, :]).flatten(), np.array(hxTrue[1, :]).flatten(), '-b')
            plt.plot(np.array(hxDR[0, :]).flatten(), np.array(hxDR[1, :]).flatten(), '-k')
            plt.plot(np.array(hxEst[0, :]).flatten(), np.array(hxEst[1, :]).flatten(), '-r')
            plot_covariance_ellipse(xEst, PEst)
            plt.axis('equal')
            plt.grid(True)
            plt.pause(0.001)
if __name__ == '__main__':
    main()