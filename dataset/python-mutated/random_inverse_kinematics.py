"""
Inverse Kinematics for an n-link arm in 3D
Author: Takayuki Murooka (takayuki5168)
"""
import math
from NLinkArm3d import NLinkArm
import random

def random_val(min_val, max_val):
    if False:
        i = 10
        return i + 15
    return min_val + random.random() * (max_val - min_val)

def main():
    if False:
        for i in range(10):
            print('nop')
    print('Start solving Inverse Kinematics 10 times')
    n_link_arm = NLinkArm([[0.0, -math.pi / 2, 0.1, 0.0], [math.pi / 2, math.pi / 2, 0.0, 0.0], [0.0, -math.pi / 2, 0.0, 0.4], [0.0, math.pi / 2, 0.0, 0.0], [0.0, -math.pi / 2, 0.0, 0.321], [0.0, math.pi / 2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    for _ in range(10):
        n_link_arm.inverse_kinematics([random_val(-0.5, 0.5), random_val(-0.5, 0.5), random_val(-0.5, 0.5), random_val(-0.5, 0.5), random_val(-0.5, 0.5), random_val(-0.5, 0.5)], plot=True)
if __name__ == '__main__':
    main()