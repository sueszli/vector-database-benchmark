"""Utilities for generating and applying rotation matrices.
"""
import numpy as np
ANGLE_EPS = 0.001

def normalize(v):
    if False:
        i = 10
        return i + 15
    return v / np.linalg.norm(v)

def get_r_matrix(ax_, angle):
    if False:
        i = 10
        return i + 15
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]], dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + (1 - np.cos(angle)) * np.linalg.matrix_power(S_hat, 2)
    else:
        R = np.eye(3)
    return R

def r_between(v_from_, v_to_):
    if False:
        print('Hello World!')
    v_from = normalize(v_from_)
    v_to = normalize(v_to_)
    ax = normalize(np.cross(v_from, v_to))
    angle = np.arccos(np.dot(v_from, v_to))
    return get_r_matrix(ax, angle)

def rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to):
    if False:
        while True:
            i = 10
    inputs = [up_from, lookat_from, up_to, lookat_to]
    for i in range(4):
        inputs[i] = normalize(np.array(inputs[i]).reshape((-1,)))
    (up_from, lookat_from, up_to, lookat_to) = inputs
    r1 = r_between(lookat_from, lookat_to)
    new_x = np.dot(r1, np.array([1, 0, 0]).reshape((-1, 1))).reshape(-1)
    to_x = normalize(np.cross(lookat_to, up_to))
    angle = np.arccos(np.dot(new_x, to_x))
    if angle > ANGLE_EPS:
        if angle < np.pi - ANGLE_EPS:
            ax = normalize(np.cross(new_x, to_x))
            flip = np.dot(lookat_to, ax)
            if flip > 0:
                r2 = get_r_matrix(lookat_to, angle)
            elif flip < 0:
                r2 = get_r_matrix(lookat_to, -1.0 * angle)
        else:
            r2 = get_r_matrix(lookat_to, angle)
    else:
        r2 = np.eye(3)
    return np.dot(r2, r1)