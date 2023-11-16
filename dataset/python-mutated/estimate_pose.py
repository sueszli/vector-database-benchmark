"""
Reference: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
"""
from math import cos, sin, atan2, asin, sqrt
import numpy as np
from .params import param_mean, param_std

def parse_pose(param):
    if False:
        print('Hello World!')
    param = param * param_std + param_mean
    Ps = param[:12].reshape(3, -1)
    (s, R, t3d) = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)
    pose = matrix2angle(R)
    return (P, pose)

def matrix2angle(R):
    if False:
        i = 10
        return i + 15
    ' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf\n    Args:\n        R: (3,3). rotation matrix\n    Returns:\n        x: yaw\n        y: pitch\n        z: roll\n    '
    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    else:
        z = 0
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])
    return (x, y, z)

def P2sRt(P):
    if False:
        i = 10
        return i + 15
    ' decompositing camera matrix P.\n    Args:\n        P: (3, 4). Affine Camera Matrix.\n    Returns:\n        s: scale factor.\n        R: (3, 3). rotation matrix.\n        t2d: (2,). 2d translation.\n    '
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return (s, R, t3d)

def main():
    if False:
        return 10
    pass
if __name__ == '__main__':
    main()