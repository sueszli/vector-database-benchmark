"""Adapted from rllab maze_env_utils.py."""
import numpy as np
import math

class Move(object):
    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17
    SpinXY = 18

def can_move_x(movable):
    if False:
        return 10
    return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ, Move.SpinXY]

def can_move_y(movable):
    if False:
        while True:
            i = 10
    return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ, Move.SpinXY]

def can_move_z(movable):
    if False:
        i = 10
        return i + 15
    return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]

def can_spin(movable):
    if False:
        for i in range(10):
            print('nop')
    return movable in [Move.SpinXY]

def can_move(movable):
    if False:
        i = 10
        return i + 15
    return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)

def construct_maze(maze_id='Maze'):
    if False:
        for i in range(10):
            print('nop')
    if maze_id == 'Maze':
        structure = [[1, 1, 1, 1, 1], [1, 'r', 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
    elif maze_id == 'Push':
        structure = [[1, 1, 1, 1, 1], [1, 0, 'r', 1, 1], [1, 0, Move.XY, 0, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
    elif maze_id == 'Fall':
        structure = [[1, 1, 1, 1], [1, 'r', 0, 1], [1, 0, Move.YZ, 1], [1, -1, -1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    elif maze_id == 'Block':
        O = 'r'
        structure = [[1, 1, 1, 1, 1], [1, O, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
    elif maze_id == 'BlockMaze':
        O = 'r'
        structure = [[1, 1, 1, 1], [1, O, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    else:
        raise NotImplementedError('The provided MazeId %s is not recognized' % maze_id)
    return structure

def line_intersect(pt1, pt2, ptA, ptB):
    if False:
        print('Hello World!')
    '\n  Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html\n\n  this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)\n  '
    DET_TOLERANCE = 1e-08
    (x1, y1) = pt1
    (x2, y2) = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1
    (x, y) = ptA
    (xB, yB) = ptB
    dx = xB - x
    dy = yB - y
    DET = -dx1 * dy + dy1 * dx
    if math.fabs(DET) < DET_TOLERANCE:
        return (0, 0, 0, 0, 0)
    DETinv = 1.0 / DET
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)

def ray_segment_intersect(ray, segment):
    if False:
        for i in range(10):
            print('nop')
    '\n  Check if the ray originated from (x, y) with direction theta intersects the line segment (x1, y1) -- (x2, y2),\n  and return the intersection point if there is one\n  '
    ((x, y), theta) = ray
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    (xo, yo, valid, r, s) = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and (0 <= s <= 1):
        return (xo, yo)
    return None

def point_distance(p1, p2):
    if False:
        print('Hello World!')
    (x1, y1) = p1
    (x2, y2) = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5