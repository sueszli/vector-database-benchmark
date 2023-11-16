"""
Iterative Closest Point (ICP) SLAM example
author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı, Shamil Gemuev
"""
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
EPS = 0.0001
MAX_ITER = 100
show_animation = True

def icp_matching(previous_points, current_points):
    if False:
        i = 10
        return i + 15
    '\n    Iterative Closest Point matching\n    - input\n    previous_points: 2D or 3D points in the previous frame\n    current_points: 2D or 3D points in the current frame\n    - output\n    R: Rotation matrix\n    T: Translation vector\n    '
    H = None
    dError = np.inf
    preError = np.inf
    count = 0
    if show_animation:
        fig = plt.figure()
        if previous_points.shape[0] == 3:
            fig.add_subplot(111, projection='3d')
    while dError >= EPS:
        count += 1
        if show_animation:
            plot_points(previous_points, current_points, fig)
            plt.pause(0.1)
        (indexes, error) = nearest_neighbor_association(previous_points, current_points)
        (Rt, Tt) = svd_motion_estimation(previous_points[:, indexes], current_points)
        current_points = Rt @ current_points + Tt[:, np.newaxis]
        dError = preError - error
        print('Residual:', error)
        if dError < 0:
            print('Not Converge...', preError, dError, count)
            break
        preError = error
        H = update_homogeneous_matrix(H, Rt, Tt)
        if dError <= EPS:
            print('Converge', error, dError, count)
            break
        elif MAX_ITER <= count:
            print('Not Converge...', error, dError, count)
            break
    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])
    return (R, T)

def update_homogeneous_matrix(Hin, R, T):
    if False:
        while True:
            i = 10
    r_size = R.shape[0]
    H = np.zeros((r_size + 1, r_size + 1))
    H[0:r_size, 0:r_size] = R
    H[0:r_size, r_size] = T
    H[r_size, r_size] = 1.0
    if Hin is None:
        return H
    else:
        return Hin @ H

def nearest_neighbor_association(previous_points, current_points):
    if False:
        i = 10
        return i + 15
    delta_points = previous_points - current_points
    d = np.linalg.norm(delta_points, axis=0)
    error = sum(d)
    d = np.linalg.norm(np.repeat(current_points, previous_points.shape[1], axis=1) - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
    indexes = np.argmin(d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)
    return (indexes, error)

def svd_motion_estimation(previous_points, current_points):
    if False:
        for i in range(10):
            print('nop')
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)
    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]
    W = c_shift @ p_shift.T
    (u, s, vh) = np.linalg.svd(W)
    R = (u @ vh).T
    t = pm - R @ cm
    return (R, t)

def plot_points(previous_points, current_points, figure):
    if False:
        while True:
            i = 10
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
    if previous_points.shape[0] == 3:
        plt.clf()
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(previous_points[0, :], previous_points[1, :], previous_points[2, :], c='r', marker='.')
        axes.scatter(current_points[0, :], current_points[1, :], current_points[2, :], c='b', marker='.')
        axes.scatter(0.0, 0.0, 0.0, c='r', marker='x')
        figure.canvas.draw()
    else:
        plt.cla()
        plt.plot(previous_points[0, :], previous_points[1, :], '.r')
        plt.plot(current_points[0, :], current_points[1, :], '.b')
        plt.plot(0.0, 0.0, 'xr')
        plt.axis('equal')

def main():
    if False:
        i = 10
        return i + 15
    print(__file__ + ' start!!')
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]
    nsim = 3
    for _ in range(nsim):
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))
        cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0] for (x, y) in zip(px, py)]
        cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1] for (x, y) in zip(px, py)]
        current_points = np.vstack((cx, cy))
        (R, T) = icp_matching(previous_points, current_points)
        print('R:', R)
        print('T:', T)

def main_3d_points():
    if False:
        return 10
    print(__file__ + ' start!!')
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, -5, np.deg2rad(-10.0)]
    nsim = 3
    for _ in range(nsim):
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        pz = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py, pz))
        cx = [math.cos(motion[3]) * x - math.sin(motion[3]) * z + motion[0] for (x, z) in zip(px, pz)]
        cy = [y + motion[1] for y in py]
        cz = [math.sin(motion[3]) * x + math.cos(motion[3]) * z + motion[2] for (x, z) in zip(px, pz)]
        current_points = np.vstack((cx, cy, cz))
        (R, T) = icp_matching(previous_points, current_points)
        print('R:', R)
        print('T:', T)
if __name__ == '__main__':
    main()
    main_3d_points()