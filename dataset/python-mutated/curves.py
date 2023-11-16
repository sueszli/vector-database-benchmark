import math
import numpy as np
curve_distance_epsilon = 1e-30
curve_collinearity_epsilon = 1e-30
curve_angle_tolerance_epsilon = 0.01
curve_recursion_limit = 32
m_cusp_limit = 0.0
m_angle_tolerance = 10 * math.pi / 180.0
m_approximation_scale = 1.0
m_distance_tolerance_square = (0.5 / m_approximation_scale) ** 2

def calc_sq_distance(x1, y1, x2, y2):
    if False:
        return 10
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy

def _curve3_recursive_bezier(points, x1, y1, x2, y2, x3, y3, level=0):
    if False:
        return 10
    if level > curve_recursion_limit:
        return
    x12 = (x1 + x2) / 2.0
    y12 = (y1 + y2) / 2.0
    x23 = (x2 + x3) / 2.0
    y23 = (y2 + y3) / 2.0
    x123 = (x12 + x23) / 2.0
    y123 = (y12 + y23) / 2.0
    dx = x3 - x1
    dy = y3 - y1
    d = math.fabs((x2 - x3) * dy - (y2 - y3) * dx)
    if d > curve_collinearity_epsilon:
        if d * d <= m_distance_tolerance_square * (dx * dx + dy * dy):
            if m_angle_tolerance < curve_angle_tolerance_epsilon:
                points.append((x123, y123))
                return
            da = math.fabs(math.atan2(y3 - y2, x3 - x2) - math.atan2(y2 - y1, x2 - x1))
            if da >= math.pi:
                da = 2 * math.pi - da
            if da < m_angle_tolerance:
                points.append((x123, y123))
                return
    else:
        da = dx * dx + dy * dy
        if da == 0:
            d = calc_sq_distance(x1, y1, x2, y2)
        else:
            d = ((x2 - x1) * dx + (y2 - y1) * dy) / da
            if d > 0 and d < 1:
                return
            if d <= 0:
                d = calc_sq_distance(x2, y2, x1, y1)
            elif d >= 1:
                d = calc_sq_distance(x2, y2, x3, y3)
            else:
                d = calc_sq_distance(x2, y2, x1 + d * dx, y1 + d * dy)
        if d < m_distance_tolerance_square:
            points.append((x2, y2))
            return
    _curve3_recursive_bezier(points, x1, y1, x12, y12, x123, y123, level + 1)
    _curve3_recursive_bezier(points, x123, y123, x23, y23, x3, y3, level + 1)

def _curve4_recursive_bezier(points, x1, y1, x2, y2, x3, y3, x4, y4, level=0):
    if False:
        while True:
            i = 10
    if level > curve_recursion_limit:
        return
    x12 = (x1 + x2) / 2.0
    y12 = (y1 + y2) / 2.0
    x23 = (x2 + x3) / 2.0
    y23 = (y2 + y3) / 2.0
    x34 = (x3 + x4) / 2.0
    y34 = (y3 + y4) / 2.0
    x123 = (x12 + x23) / 2.0
    y123 = (y12 + y23) / 2.0
    x234 = (x23 + x34) / 2.0
    y234 = (y23 + y34) / 2.0
    x1234 = (x123 + x234) / 2.0
    y1234 = (y123 + y234) / 2.0
    dx = x4 - x1
    dy = y4 - y1
    d2 = math.fabs((x2 - x4) * dy - (y2 - y4) * dx)
    d3 = math.fabs((x3 - x4) * dy - (y3 - y4) * dx)
    s = int((d2 > curve_collinearity_epsilon) << 1) + int(d3 > curve_collinearity_epsilon)
    if s == 0:
        k = dx * dx + dy * dy
        if k == 0:
            d2 = calc_sq_distance(x1, y1, x2, y2)
            d3 = calc_sq_distance(x4, y4, x3, y3)
        else:
            k = 1.0 / k
            da1 = x2 - x1
            da2 = y2 - y1
            d2 = k * (da1 * dx + da2 * dy)
            da1 = x3 - x1
            da2 = y3 - y1
            d3 = k * (da1 * dx + da2 * dy)
            if d2 > 0 and d2 < 1 and (d3 > 0) and (d3 < 1):
                return
            if d2 <= 0:
                d2 = calc_sq_distance(x2, y2, x1, y1)
            elif d2 >= 1:
                d2 = calc_sq_distance(x2, y2, x4, y4)
            else:
                d2 = calc_sq_distance(x2, y2, x1 + d2 * dx, y1 + d2 * dy)
            if d3 <= 0:
                d3 = calc_sq_distance(x3, y3, x1, y1)
            elif d3 >= 1:
                d3 = calc_sq_distance(x3, y3, x4, y4)
            else:
                d3 = calc_sq_distance(x3, y3, x1 + d3 * dx, y1 + d3 * dy)
        if d2 > d3:
            if d2 < m_distance_tolerance_square:
                points.append((x2, y2))
                return
        elif d3 < m_distance_tolerance_square:
            points.append((x3, y3))
            return
    elif s == 1:
        if d3 * d3 <= m_distance_tolerance_square * (dx * dx + dy * dy):
            if m_angle_tolerance < curve_angle_tolerance_epsilon:
                points.append((x23, y23))
                return
            da1 = math.fabs(math.atan2(y4 - y3, x4 - x3) - math.atan2(y3 - y2, x3 - x2))
            if da1 >= math.pi:
                da1 = 2 * math.pi - da1
            if da1 < m_angle_tolerance:
                points.extend([(x2, y2), (x3, y3)])
                return
            if m_cusp_limit != 0.0:
                if da1 > m_cusp_limit:
                    points.append((x3, y3))
                    return
    elif s == 2:
        if d2 * d2 <= m_distance_tolerance_square * (dx * dx + dy * dy):
            if m_angle_tolerance < curve_angle_tolerance_epsilon:
                points.append((x23, y23))
                return
            da1 = math.fabs(math.atan2(y3 - y2, x3 - x2) - math.atan2(y2 - y1, x2 - x1))
            if da1 >= math.pi:
                da1 = 2 * math.pi - da1
            if da1 < m_angle_tolerance:
                points.extend([(x2, y2), (x3, y3)])
                return
            if m_cusp_limit != 0.0:
                if da1 > m_cusp_limit:
                    points.append((x2, y2))
                    return
    elif s == 3:
        if (d2 + d3) * (d2 + d3) <= m_distance_tolerance_square * (dx * dx + dy * dy):
            if m_angle_tolerance < curve_angle_tolerance_epsilon:
                points.append((x23, y23))
                return
            k = math.atan2(y3 - y2, x3 - x2)
            da1 = math.fabs(k - math.atan2(y2 - y1, x2 - x1))
            da2 = math.fabs(math.atan2(y4 - y3, x4 - x3) - k)
            if da1 >= math.pi:
                da1 = 2 * math.pi - da1
            if da2 >= math.pi:
                da2 = 2 * math.pi - da2
            if da1 + da2 < m_angle_tolerance:
                points.append((x23, y23))
                return
            if m_cusp_limit != 0.0:
                if da1 > m_cusp_limit:
                    points.append((x2, y2))
                    return
                if da2 > m_cusp_limit:
                    points.append((x3, y3))
                    return
    _curve4_recursive_bezier(points, x1, y1, x12, y12, x123, y123, x1234, y1234, level + 1)
    _curve4_recursive_bezier(points, x1234, y1234, x234, y234, x34, y34, x4, y4, level + 1)

def curve3_bezier(p1, p2, p3):
    if False:
        i = 10
        return i + 15
    '\n    Generate the vertices for a quadratic Bezier curve.\n\n    The vertices returned by this function can be passed to a LineVisual or\n    ArrowVisual.\n\n    Parameters\n    ----------\n    p1 : array\n        2D coordinates of the start point\n    p2 : array\n        2D coordinates of the first curve point\n    p3 : array\n        2D coordinates of the end point\n\n    Returns\n    -------\n    coords : list\n        Vertices for the Bezier curve.\n\n    See Also\n    --------\n    curve4_bezier\n\n    Notes\n    -----\n    For more information about Bezier curves please refer to the `Wikipedia`_\n    page.\n\n    .. _Wikipedia: https://en.wikipedia.org/wiki/B%C3%A9zier_curve\n    '
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    points = []
    _curve3_recursive_bezier(points, x1, y1, x2, y2, x3, y3)
    (dx, dy) = (points[0][0] - x1, points[0][1] - y1)
    if dx * dx + dy * dy > 1e-10:
        points.insert(0, (x1, y1))
    (dx, dy) = (points[-1][0] - x3, points[-1][1] - y3)
    if dx * dx + dy * dy > 1e-10:
        points.append((x3, y3))
    return np.array(points).reshape(len(points), 2)

def curve4_bezier(p1, p2, p3, p4):
    if False:
        return 10
    '\n    Generate the vertices for a third order Bezier curve.\n\n    The vertices returned by this function can be passed to a LineVisual or\n    ArrowVisual.\n\n    Parameters\n    ----------\n    p1 : array\n        2D coordinates of the start point\n    p2 : array\n        2D coordinates of the first curve point\n    p3 : array\n        2D coordinates of the second curve point\n    p4 : array\n        2D coordinates of the end point\n\n    Returns\n    -------\n    coords : list\n        Vertices for the Bezier curve.\n\n    See Also\n    --------\n    curve3_bezier\n\n    Notes\n    -----\n    For more information about Bezier curves please refer to the `Wikipedia`_\n    page.\n\n    .. _Wikipedia: https://en.wikipedia.org/wiki/B%C3%A9zier_curve\n    '
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    (x4, y4) = p4
    points = []
    _curve4_recursive_bezier(points, x1, y1, x2, y2, x3, y3, x4, y4)
    (dx, dy) = (points[0][0] - x1, points[0][1] - y1)
    if dx * dx + dy * dy > 1e-10:
        points.insert(0, (x1, y1))
    (dx, dy) = (points[-1][0] - x4, points[-1][1] - y4)
    if dx * dx + dy * dy > 1e-10:
        points.append((x4, y4))
    return np.array(points).reshape(len(points), 2)