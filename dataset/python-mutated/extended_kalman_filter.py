"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import plot_covariance_ellipse
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
R = np.diag([1.0, 1.0]) ** 2
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
DT = 0.1
SIM_TIME = 50.0
show_animation = True

def calc_input():
    if False:
        for i in range(10):
            print('nop')
    v = 1.0
    yawrate = 0.1
    u = np.array([[v], [yawrate]])
    return u

def observation(xTrue, xd, u):
    if False:
        i = 10
        return i + 15
    xTrue = motion_model(xTrue, u)
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    xd = motion_model(xd, ud)
    return (xTrue, z, xd, ud)

def motion_model(x, u):
    if False:
        while True:
            i = 10
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT], [1.0, 0.0]])
    x = F @ x + B @ u
    return x

def observation_model(x):
    if False:
        while True:
            i = 10
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    z = H @ x
    return z

def jacob_f(x, u):
    if False:
        while True:
            i = 10
    '\n    Jacobian of Motion Model\n\n    motion model\n    x_{t+1} = x_t+v*dt*cos(yaw)\n    y_{t+1} = y_t+v*dt*sin(yaw)\n    yaw_{t+1} = yaw_t+omega*dt\n    v_{t+1} = v{t}\n    so\n    dx/dyaw = -v*dt*sin(yaw)\n    dx/dv = dt*cos(yaw)\n    dy/dyaw = v*dt*cos(yaw)\n    dy/dv = dt*sin(yaw)\n    '
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([[1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)], [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return jF

def jacob_h():
    if False:
        i = 10
        return i + 15
    jH = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    return jH

def ekf_estimation(xEst, PEst, z, u):
    if False:
        return 10
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return (xEst, PEst)

def main():
    if False:
        while True:
            i = 10
    print(__file__ + ' start!!')
    time = 0.0
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)
    xDR = np.zeros((4, 1))
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        (xTrue, z, xDR, ud) = observation(xTrue, xDR, u)
        (xEst, PEst) = ekf_estimation(xEst, PEst, z, ud)
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], '.g')
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), '-b')
            plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), '-k')
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), '-r')
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis('equal')
            plt.grid(True)
            plt.pause(0.001)
if __name__ == '__main__':
    main()