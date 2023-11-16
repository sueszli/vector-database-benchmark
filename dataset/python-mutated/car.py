"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""
import sys
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
from math import cos, sin, tan, pi
import matplotlib.pyplot as plt
import numpy as np
from utils.angle import rot_mat_2d
WB = 3.0
W = 2.0
LF = 3.3
LB = 1.0
MAX_STEER = 0.6
BUBBLE_DIST = (LF - LB) / 2.0
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    if False:
        print('Hello World!')
    for (i_x, i_y, i_yaw) in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)
        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)
        if not ids:
            continue
        if not rectangle_check(i_x, i_y, i_yaw, [ox[i] for i in ids], [oy[i] for i in ids]):
            return False
    return True

def rectangle_check(x, y, yaw, ox, oy):
    if False:
        print('Hello World!')
    rot = rot_mat_2d(yaw)
    for (iox, ioy) in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        (rx, ry) = (converted_xy[0], converted_xy[1])
        if not (rx > LF or rx < -LB or ry > W / 2.0 or (ry < -W / 2.0)):
            return False
    return True

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc='r', ec='k'):
    if False:
        return 10
    'Plot arrow.'
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

def plot_car(x, y, yaw):
    if False:
        print('Hello World!')
    car_color = '-k'
    (c, s) = (cos(yaw), sin(yaw))
    rot = rot_mat_2d(-yaw)
    (car_outline_x, car_outline_y) = ([], [])
    for (rx, ry) in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)
    (arrow_x, arrow_y, arrow_yaw) = (c * 1.5 + x, s * 1.5 + y, yaw)
    plot_arrow(arrow_x, arrow_y, arrow_yaw)
    plt.plot(car_outline_x, car_outline_y, car_color)

def pi_2_pi(angle):
    if False:
        for i in range(10):
            print('nop')
    return (angle + pi) % (2 * pi) - pi

def move(x, y, yaw, distance, steer, L=WB):
    if False:
        i = 10
        return i + 15
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)
    return (x, y, yaw)

def main():
    if False:
        return 10
    (x, y, yaw) = (0.0, 0.0, 1.0)
    plt.axis('equal')
    plot_car(x, y, yaw)
    plt.show()
if __name__ == '__main__':
    main()