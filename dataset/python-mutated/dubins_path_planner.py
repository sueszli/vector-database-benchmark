"""

Dubins path planner sample code

author Atsushi Sakai(@Atsushi_twi)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from math import sin, cos, atan2, sqrt, acos, pi, hypot
import numpy as np
from utils.angle import angle_mod, rot_mat_2d
show_animation = True

def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature, step_size=0.1, selected_types=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Plan dubins path\n\n    Parameters\n    ----------\n    s_x : float\n        x position of the start point [m]\n    s_y : float\n        y position of the start point [m]\n    s_yaw : float\n        yaw angle of the start point [rad]\n    g_x : float\n        x position of the goal point [m]\n    g_y : float\n        y position of the end point [m]\n    g_yaw : float\n        yaw angle of the end point [rad]\n    curvature : float\n        curvature for curve [1/m]\n    step_size : float (optional)\n        step size between two path points [m]. Default is 0.1\n    selected_types : a list of string or None\n        selected path planning types. If None, all types are used for\n        path planning, and minimum path length result is returned.\n        You can select used path plannings types by a string list.\n        e.g.: ["RSL", "RSR"]\n\n    Returns\n    -------\n    x_list: array\n        x positions of the path\n    y_list: array\n        y positions of the path\n    yaw_list: array\n        yaw angles of the path\n    modes: array\n        mode list of the path\n    lengths: array\n        arrow_length list of the path segments.\n\n    Examples\n    --------\n    You can generate a dubins path.\n\n    >>> start_x = 1.0  # [m]\n    >>> start_y = 1.0  # [m]\n    >>> start_yaw = np.deg2rad(45.0)  # [rad]\n    >>> end_x = -3.0  # [m]\n    >>> end_y = -3.0  # [m]\n    >>> end_yaw = np.deg2rad(-45.0)  # [rad]\n    >>> curvature = 1.0\n    >>> path_x, path_y, path_yaw, mode, _ = plan_dubins_path(\n                start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)\n    >>> plt.plot(path_x, path_y, label="final course " + "".join(mode))\n    >>> plot_arrow(start_x, start_y, start_yaw)\n    >>> plot_arrow(end_x, end_y, end_yaw)\n    >>> plt.legend()\n    >>> plt.grid(True)\n    >>> plt.axis("equal")\n    >>> plt.show()\n\n    .. image:: dubins_path.jpg\n    '
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw
    (lp_x, lp_y, lp_yaw, modes, lengths) = _dubins_path_planning_from_origin(local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size, planning_funcs)
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)
    return (x_list, y_list, yaw_list, modes, lengths)

def _mod2pi(theta):
    if False:
        return 10
    return angle_mod(theta, zero_2_2pi=True)

def _calc_trig_funcs(alpha, beta):
    if False:
        while True:
            i = 10
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return (sin_a, sin_b, cos_a, cos_b, cos_ab)

def _LSL(alpha, beta, d):
    if False:
        print('Hello World!')
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    mode = ['L', 'S', 'L']
    p_squared = 2 + d ** 2 - 2 * cos_ab + 2 * d * (sin_a - sin_b)
    if p_squared < 0:
        return (None, None, None, mode)
    tmp = atan2(cos_b - cos_a, d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return (d1, d2, d3, mode)

def _RSR(alpha, beta, d):
    if False:
        return 10
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    mode = ['R', 'S', 'R']
    p_squared = 2 + d ** 2 - 2 * cos_ab + 2 * d * (sin_b - sin_a)
    if p_squared < 0:
        return (None, None, None, mode)
    tmp = atan2(cos_a - cos_b, d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return (d1, d2, d3, mode)

def _LSR(alpha, beta, d):
    if False:
        print('Hello World!')
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + 2 * cos_ab + 2 * d * (sin_a + sin_b)
    mode = ['L', 'S', 'R']
    if p_squared < 0:
        return (None, None, None, mode)
    d1 = sqrt(p_squared)
    tmp = atan2(-cos_a - cos_b, d + sin_a + sin_b) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return (d2, d1, d3, mode)

def _RSL(alpha, beta, d):
    if False:
        while True:
            i = 10
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + 2 * cos_ab - 2 * d * (sin_a + sin_b)
    mode = ['R', 'S', 'L']
    if p_squared < 0:
        return (None, None, None, mode)
    d1 = sqrt(p_squared)
    tmp = atan2(cos_a + cos_b, d - sin_a - sin_b) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return (d2, d1, d3, mode)

def _RLR(alpha, beta, d):
    if False:
        return 10
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    mode = ['R', 'L', 'R']
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return (None, None, None, mode)
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return (d1, d2, d3, mode)

def _LRL(alpha, beta, d):
    if False:
        for i in range(10):
            print('nop')
    (sin_a, sin_b, cos_a, cos_b, cos_ab) = _calc_trig_funcs(alpha, beta)
    mode = ['L', 'R', 'L']
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (-sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return (None, None, None, mode)
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return (d1, d2, d3, mode)
_PATH_TYPE_MAP = {'LSL': _LSL, 'RSR': _RSR, 'LSR': _LSR, 'RSL': _RSL, 'RLR': _RLR, 'LRL': _LRL}

def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature, step_size, planning_funcs):
    if False:
        for i in range(10):
            print('nop')
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature
    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)
    best_cost = float('inf')
    (b_d1, b_d2, b_d3, b_mode) = (None, None, None, None)
    for planner in planning_funcs:
        (d1, d2, d3, mode) = planner(alpha, beta, d)
        if d1 is None:
            continue
        cost = abs(d1) + abs(d2) + abs(d3)
        if best_cost > cost:
            (b_d1, b_d2, b_d3, b_mode, best_cost) = (d1, d2, d3, mode, cost)
    lengths = [b_d1, b_d2, b_d3]
    (x_list, y_list, yaw_list) = _generate_local_course(lengths, b_mode, curvature, step_size)
    lengths = [length / curvature for length in lengths]
    return (x_list, y_list, yaw_list, b_mode, lengths)

def _interpolate(length, mode, max_curvature, origin_x, origin_y, origin_yaw, path_x, path_y, path_yaw):
    if False:
        for i in range(10):
            print('nop')
    if mode == 'S':
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == 'L':
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == 'R':
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)
        if mode == 'L':
            path_yaw.append(origin_yaw + length)
        elif mode == 'R':
            path_yaw.append(origin_yaw - length)
    return (path_x, path_y, path_yaw)

def _generate_local_course(lengths, modes, max_curvature, step_size):
    if False:
        i = 10
        return i + 15
    (p_x, p_y, p_yaw) = ([0.0], [0.0], [0.0])
    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue
        (origin_x, origin_y, origin_yaw) = (p_x[-1], p_y[-1], p_yaw[-1])
        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            (p_x, p_y, p_yaw) = _interpolate(current_length, mode, max_curvature, origin_x, origin_y, origin_yaw, p_x, p_y, p_yaw)
            current_length += step_size
        (p_x, p_y, p_yaw) = _interpolate(length, mode, max_curvature, origin_x, origin_y, origin_yaw, p_x, p_y, p_yaw)
    return (p_x, p_y, p_yaw)

def main():
    if False:
        while True:
            i = 10
    print('Dubins path planner sample start!!')
    import matplotlib.pyplot as plt
    from utils.plot import plot_arrow
    start_x = 1.0
    start_y = 1.0
    start_yaw = np.deg2rad(45.0)
    end_x = -3.0
    end_y = -3.0
    end_yaw = np.deg2rad(-45.0)
    curvature = 1.0
    (path_x, path_y, path_yaw, mode, lengths) = plan_dubins_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
    if show_animation:
        plt.plot(path_x, path_y, label=''.join(mode))
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
if __name__ == '__main__':
    main()