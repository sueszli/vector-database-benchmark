"""

Path Planner with B-Spline

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from utils.plot import plot_curvature

def approximate_b_spline_path(x: list, y: list, n_path_points: int, degree: int=3, s=None) -> tuple:
    if False:
        while True:
            i = 10
    '\n    Approximate points with a B-Spline path\n\n    Parameters\n    ----------\n    x : array_like\n        x position list of approximated points\n    y : array_like\n        y position list of approximated points\n    n_path_points : int\n        number of path points\n    degree : int, optional\n        B Spline curve degree. Must be 2<= k <= 5. Default: 3.\n    s : int, optional\n        smoothing parameter. If this value is bigger, the path will be\n        smoother, but it will be less accurate. If this value is smaller,\n        the path will be more accurate, but it will be less smooth.\n        When `s` is 0, it is equivalent to the interpolation. Default is None,\n        in this case `s` will be `len(x)`.\n\n    Returns\n    -------\n    x : array\n        x positions of the result path\n    y : array\n        y positions of the result path\n    heading : array\n        heading of the result path\n    curvature : array\n        curvature of the result path\n\n    '
    distances = _calc_distance_vector(x, y)
    spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
    spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)
    sampled = np.linspace(0.0, distances[-1], n_path_points)
    return _evaluate_spline(sampled, spl_i_x, spl_i_y)

def interpolate_b_spline_path(x, y, n_path_points: int, degree: int=3) -> tuple:
    if False:
        return 10
    '\n    Interpolate x-y points with a B-Spline path\n\n    Parameters\n    ----------\n    x : array_like\n        x positions of interpolated points\n    y : array_like\n        y positions of interpolated points\n    n_path_points : int\n        number of path points\n    degree : int, optional\n        B-Spline degree. Must be 2<= k <= 5. Default: 3\n\n    Returns\n    -------\n    x : array\n        x positions of the result path\n    y : array\n        y positions of the result path\n    heading : array\n        heading of the result path\n    curvature : array\n        curvature of the result path\n\n    '
    return approximate_b_spline_path(x, y, n_path_points, degree, s=0.0)

def _calc_distance_vector(x, y):
    if False:
        print('Hello World!')
    (dx, dy) = (np.diff(x), np.diff(y))
    distances = np.cumsum([np.hypot(idx, idy) for (idx, idy) in zip(dx, dy)])
    distances = np.concatenate(([0.0], distances))
    distances /= distances[-1]
    return distances

def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    if False:
        while True:
            i = 10
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)
    dx = spl_i_x.derivative(1)(sampled)
    dy = spl_i_y.derivative(1)(sampled)
    heading = np.arctan2(dy, dx)
    ddx = spl_i_x.derivative(2)(sampled)
    ddy = spl_i_y.derivative(2)(sampled)
    curvature = (ddy * dx - ddx * dy) / np.power(dx * dx + dy * dy, 2.0 / 3.0)
    return (np.array(x), y, heading, curvature)

def main():
    if False:
        i = 10
        return i + 15
    print(__file__ + ' start!!')
    way_point_x = [-1.0, 3.0, 4.0, 2.0, 1.0]
    way_point_y = [0.0, -3.0, 1.0, 1.0, 3.0]
    n_course_point = 50
    plt.subplots()
    (rax, ray, heading, curvature) = approximate_b_spline_path(way_point_x, way_point_y, n_course_point, s=0.5)
    plt.plot(rax, ray, '-r', label='Approximated B-Spline path')
    plot_curvature(rax, ray, heading, curvature)
    plt.title('B-Spline approximation')
    plt.plot(way_point_x, way_point_y, '-og', label='way points')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.subplots()
    (rix, riy, heading, curvature) = interpolate_b_spline_path(way_point_x, way_point_y, n_course_point)
    plt.plot(rix, riy, '-b', label='Interpolated B-Spline path')
    plot_curvature(rix, riy, heading, curvature)
    plt.title('B-Spline interpolation')
    plt.plot(way_point_x, way_point_y, '-og', label='way points')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
if __name__ == '__main__':
    main()