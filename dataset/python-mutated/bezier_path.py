"""

Path planning with Bezier curve.

author: Atsushi Sakai(@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
show_animation = True

def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset):
    if False:
        return 10
    '\n    Compute control points and path given start and end position.\n\n    :param sx: (float) x-coordinate of the starting point\n    :param sy: (float) y-coordinate of the starting point\n    :param syaw: (float) yaw angle at start\n    :param ex: (float) x-coordinate of the ending point\n    :param ey: (float) y-coordinate of the ending point\n    :param eyaw: (float) yaw angle at the end\n    :param offset: (float)\n    :return: (numpy array, numpy array)\n    '
    dist = np.hypot(sx - ex, sy - ey) / offset
    control_points = np.array([[sx, sy], [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)], [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)], [ex, ey]])
    path = calc_bezier_path(control_points, n_points=100)
    return (path, control_points)

def calc_bezier_path(control_points, n_points=100):
    if False:
        i = 10
        return i + 15
    '\n    Compute bezier path (trajectory) given control points.\n\n    :param control_points: (numpy array)\n    :param n_points: (int) number of points in the trajectory\n    :return: (numpy array)\n    '
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))
    return np.array(traj)

def bernstein_poly(n, i, t):
    if False:
        return 10
    '\n    Bernstein polynom.\n\n    :param n: (int) polynom degree\n    :param i: (int)\n    :param t: (float)\n    :return: (float)\n    '
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier(t, control_points):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return one point on the bezier curve.\n\n    :param t: (float) number in [0, 1]\n    :param control_points: (numpy array)\n    :return: (numpy array) Coordinates of the point\n    '
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def bezier_derivatives_control_points(control_points, n_derivatives):
    if False:
        print('Hello World!')
    '\n    Compute control points of the successive derivatives of a given bezier curve.\n\n    A derivative of a bezier curve is a bezier curve.\n    See https://pomax.github.io/bezierinfo/#derivatives\n    for detailed explanations\n\n    :param control_points: (numpy array)\n    :param n_derivatives: (int)\n    e.g., n_derivatives=2 -> compute control points for first and second derivatives\n    :return: ([numpy array])\n    '
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j]) for j in range(n - 1)])
    return w

def curvature(dx, dy, ddx, ddy):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute curvature at one point given first and second derivatives.\n\n    :param dx: (float) First derivative along x axis\n    :param dy: (float)\n    :param ddx: (float) Second derivative along x axis\n    :param ddy: (float)\n    :return: (float)\n    '
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc='r', ec='k'):
    if False:
        i = 10
        return i + 15
    'Plot arrow.'
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    if False:
        i = 10
        return i + 15
    'Plot an example bezier curve.'
    start_x = 10.0
    start_y = 1.0
    start_yaw = np.radians(180.0)
    end_x = -0.0
    end_y = -3.0
    end_yaw = np.radians(-45.0)
    offset = 3.0
    (path, control_points) = calc_4points_bezier_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
    t = 0.86
    (x_target, y_target) = bezier(t, control_points)
    derivatives_cp = bezier_derivatives_control_points(control_points, 2)
    point = bezier(t, control_points)
    dt = bezier(t, derivatives_cp[1])
    ddt = bezier(t, derivatives_cp[2])
    radius = 1 / curvature(dt[0], dt[1], ddt[0], ddt[1])
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [-dt[1], dt[0]]])
    curvature_center = point + np.array([-dt[1], dt[0]]) * radius
    circle = plt.Circle(tuple(curvature_center), radius, color=(0, 0.8, 0.8), fill=False, linewidth=1)
    assert path.T[0][0] == start_x, 'path is invalid'
    assert path.T[1][0] == start_y, 'path is invalid'
    assert path.T[0][-1] == end_x, 'path is invalid'
    assert path.T[1][-1] == end_y, 'path is invalid'
    if show_animation:
        (fig, ax) = plt.subplots()
        ax.plot(path.T[0], path.T[1], label='Bezier Path')
        ax.plot(control_points.T[0], control_points.T[1], '--o', label='Control Points')
        ax.plot(x_target, y_target)
        ax.plot(tangent[:, 0], tangent[:, 1], label='Tangent')
        ax.plot(normal[:, 0], normal[:, 1], label='Normal')
        ax.add_artist(circle)
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        plt.show()

def main2():
    if False:
        i = 10
        return i + 15
    'Show the effect of the offset.'
    start_x = 10.0
    start_y = 1.0
    start_yaw = np.radians(180.0)
    end_x = -0.0
    end_y = -3.0
    end_yaw = np.radians(-45.0)
    for offset in np.arange(1.0, 5.0, 1.0):
        (path, control_points) = calc_4points_bezier_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
        assert path.T[0][0] == start_x, 'path is invalid'
        assert path.T[1][0] == start_y, 'path is invalid'
        assert path.T[0][-1] == end_x, 'path is invalid'
        assert path.T[1][-1] == end_y, 'path is invalid'
        if show_animation:
            plt.plot(path.T[0], path.T[1], label='Offset=' + str(offset))
    if show_animation:
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
if __name__ == '__main__':
    main()