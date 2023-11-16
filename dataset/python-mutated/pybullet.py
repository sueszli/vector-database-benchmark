import numpy as np
import pybullet as p

def add_line(start, end, color=[0, 0, 0], width=1, lifetime=None, parent=-1, parent_link=-1):
    if False:
        i = 10
        return i + 15
    assert len(start) == 3 and len(end) == 3
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width, parentObjectUniqueId=parent, parentLinkIndex=parent_link)

def draw_point(point, size=0.01, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size / 2 * axis
        p2 = np.array(point) + size / 2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines