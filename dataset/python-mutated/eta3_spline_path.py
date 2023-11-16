"""

eta^3 polynomials planner

author: Joe Dinius, Ph.D (https://jwdinius.github.io)
        Atsushi Sakai (@Atsushi_twi)

Ref:
- [eta^3-Splines for the Smooth Path Generation of Wheeled Mobile Robots]
(https://ieeexplore.ieee.org/document/4339545/)

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
show_animation = True

class Eta3Path(object):
    """
    Eta3Path

    input
        segments: a list of `Eta3PathSegment` instances
        defining a continuous path
    """

    def __init__(self, segments):
        if False:
            return 10
        assert isinstance(segments, list) and isinstance(segments[0], Eta3PathSegment)
        for (r, s) in zip(segments[:-1], segments[1:]):
            assert np.array_equal(r.end_pose, s.start_pose)
        self.segments = segments

    def calc_path_point(self, u):
        if False:
            while True:
                i = 10
        '\n        Eta3Path::calc_path_point\n\n        input\n            normalized interpolation point along path object, 0 <= u <= len(self.segments)\n        returns\n            2d (x,y) position vector\n        '
        assert 0 <= u <= len(self.segments)
        if np.isclose(u, len(self.segments)):
            segment_idx = len(self.segments) - 1
            u = 1.0
        else:
            segment_idx = int(np.floor(u))
            u -= segment_idx
        return self.segments[segment_idx].calc_point(u)

class Eta3PathSegment(object):
    """
    Eta3PathSegment - constructs an eta^3 path segment based on desired
    shaping, eta, and curvature vector, kappa. If either, or both,
    of eta and kappa are not set during initialization,
    they will default to zeros.

    input
        start_pose - starting pose array  (x, y, 	heta)
        end_pose - ending pose array (x, y, 	heta)
        eta - shaping parameters, default=None
        kappa - curvature parameters, default=None
    """

    def __init__(self, start_pose, end_pose, eta=None, kappa=None):
        if False:
            for i in range(10):
                print('nop')
        assert len(start_pose) == 3 and len(start_pose) == len(end_pose)
        self.start_pose = start_pose
        self.end_pose = end_pose
        if not eta:
            eta = np.zeros((6,))
        else:
            assert len(eta) == 6
        if not kappa:
            kappa = np.zeros((4,))
        else:
            assert len(kappa) == 4
        ca = np.cos(start_pose[2])
        sa = np.sin(start_pose[2])
        cb = np.cos(end_pose[2])
        sb = np.sin(end_pose[2])
        self.coeffs = np.empty((2, 8))
        self.coeffs[0, 0] = start_pose[0]
        self.coeffs[1, 0] = start_pose[1]
        self.coeffs[0, 1] = eta[0] * ca
        self.coeffs[1, 1] = eta[0] * sa
        self.coeffs[0, 2] = 1.0 / 2 * eta[2] * ca - 1.0 / 2 * eta[0] ** 2 * kappa[0] * sa
        self.coeffs[1, 2] = 1.0 / 2 * eta[2] * sa + 1.0 / 2 * eta[0] ** 2 * kappa[0] * ca
        self.coeffs[0, 3] = 1.0 / 6 * eta[4] * ca - 1.0 / 6 * (eta[0] ** 3 * kappa[1] + 3.0 * eta[0] * eta[2] * kappa[0]) * sa
        self.coeffs[1, 3] = 1.0 / 6 * eta[4] * sa + 1.0 / 6 * (eta[0] ** 3 * kappa[1] + 3.0 * eta[0] * eta[2] * kappa[0]) * ca
        tmp1 = 35.0 * (end_pose[0] - start_pose[0])
        tmp2 = (20.0 * eta[0] + 5 * eta[2] + 2.0 / 3 * eta[4]) * ca
        tmp3 = (5.0 * eta[0] ** 2 * kappa[0] + 2.0 / 3 * eta[0] ** 3 * kappa[1] + 2.0 * eta[0] * eta[2] * kappa[0]) * sa
        tmp4 = (15.0 * eta[1] - 5.0 / 2 * eta[3] + 1.0 / 6 * eta[5]) * cb
        tmp5 = (5.0 / 2 * eta[1] ** 2 * kappa[2] - 1.0 / 6 * eta[1] ** 3 * kappa[3] - 1.0 / 2 * eta[1] * eta[3] * kappa[2]) * sb
        self.coeffs[0, 4] = tmp1 - tmp2 + tmp3 - tmp4 - tmp5
        tmp1 = 35.0 * (end_pose[1] - start_pose[1])
        tmp2 = (20.0 * eta[0] + 5.0 * eta[2] + 2.0 / 3 * eta[4]) * sa
        tmp3 = (5.0 * eta[0] ** 2 * kappa[0] + 2.0 / 3 * eta[0] ** 3 * kappa[1] + 2.0 * eta[0] * eta[2] * kappa[0]) * ca
        tmp4 = (15.0 * eta[1] - 5.0 / 2 * eta[3] + 1.0 / 6 * eta[5]) * sb
        tmp5 = (5.0 / 2 * eta[1] ** 2 * kappa[2] - 1.0 / 6 * eta[1] ** 3 * kappa[3] - 1.0 / 2 * eta[1] * eta[3] * kappa[2]) * cb
        self.coeffs[1, 4] = tmp1 - tmp2 - tmp3 - tmp4 + tmp5
        tmp1 = -84.0 * (end_pose[0] - start_pose[0])
        tmp2 = (45.0 * eta[0] + 10.0 * eta[2] + eta[4]) * ca
        tmp3 = (10.0 * eta[0] ** 2 * kappa[0] + eta[0] ** 3 * kappa[1] + 3.0 * eta[0] * eta[2] * kappa[0]) * sa
        tmp4 = (39.0 * eta[1] - 7.0 * eta[3] + 1.0 / 2 * eta[5]) * cb
        tmp5 = +(7.0 * eta[1] ** 2 * kappa[2] - 1.0 / 2 * eta[1] ** 3 * kappa[3] - 3.0 / 2 * eta[1] * eta[3] * kappa[2]) * sb
        self.coeffs[0, 5] = tmp1 + tmp2 - tmp3 + tmp4 + tmp5
        tmp1 = -84.0 * (end_pose[1] - start_pose[1])
        tmp2 = (45.0 * eta[0] + 10.0 * eta[2] + eta[4]) * sa
        tmp3 = (10.0 * eta[0] ** 2 * kappa[0] + eta[0] ** 3 * kappa[1] + 3.0 * eta[0] * eta[2] * kappa[0]) * ca
        tmp4 = (39.0 * eta[1] - 7.0 * eta[3] + 1.0 / 2 * eta[5]) * sb
        tmp5 = -(7.0 * eta[1] ** 2 * kappa[2] - 1.0 / 2 * eta[1] ** 3 * kappa[3] - 3.0 / 2 * eta[1] * eta[3] * kappa[2]) * cb
        self.coeffs[1, 5] = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
        tmp1 = 70.0 * (end_pose[0] - start_pose[0])
        tmp2 = (36.0 * eta[0] + 15.0 / 2 * eta[2] + 2.0 / 3 * eta[4]) * ca
        tmp3 = +(15.0 / 2 * eta[0] ** 2 * kappa[0] + 2.0 / 3 * eta[0] ** 3 * kappa[1] + 2.0 * eta[0] * eta[2] * kappa[0]) * sa
        tmp4 = (34.0 * eta[1] - 13.0 / 2 * eta[3] + 1.0 / 2 * eta[5]) * cb
        tmp5 = -(13.0 / 2 * eta[1] ** 2 * kappa[2] - 1.0 / 2 * eta[1] ** 3 * kappa[3] - 3.0 / 2 * eta[1] * eta[3] * kappa[2]) * sb
        self.coeffs[0, 6] = tmp1 - tmp2 + tmp3 - tmp4 + tmp5
        tmp1 = 70.0 * (end_pose[1] - start_pose[1])
        tmp2 = -(36.0 * eta[0] + 15.0 / 2 * eta[2] + 2.0 / 3 * eta[4]) * sa
        tmp3 = -(15.0 / 2 * eta[0] ** 2 * kappa[0] + 2.0 / 3 * eta[0] ** 3 * kappa[1] + 2.0 * eta[0] * eta[2] * kappa[0]) * ca
        tmp4 = -(34.0 * eta[1] - 13.0 / 2 * eta[3] + 1.0 / 2 * eta[5]) * sb
        tmp5 = +(13.0 / 2 * eta[1] ** 2 * kappa[2] - 1.0 / 2 * eta[1] ** 3 * kappa[3] - 3.0 / 2 * eta[1] * eta[3] * kappa[2]) * cb
        self.coeffs[1, 6] = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
        tmp1 = -20.0 * (end_pose[0] - start_pose[0])
        tmp2 = (10.0 * eta[0] + 2.0 * eta[2] + 1.0 / 6 * eta[4]) * ca
        tmp3 = -(2.0 * eta[0] ** 2 * kappa[0] + 1.0 / 6 * eta[0] ** 3 * kappa[1] + 1.0 / 2 * eta[0] * eta[2] * kappa[0]) * sa
        tmp4 = (10.0 * eta[1] - 2.0 * eta[3] + 1.0 / 6 * eta[5]) * cb
        tmp5 = (2.0 * eta[1] ** 2 * kappa[2] - 1.0 / 6 * eta[1] ** 3 * kappa[3] - 1.0 / 2 * eta[1] * eta[3] * kappa[2]) * sb
        self.coeffs[0, 7] = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
        tmp1 = -20.0 * (end_pose[1] - start_pose[1])
        tmp2 = (10.0 * eta[0] + 2.0 * eta[2] + 1.0 / 6 * eta[4]) * sa
        tmp3 = (2.0 * eta[0] ** 2 * kappa[0] + 1.0 / 6 * eta[0] ** 3 * kappa[1] + 1.0 / 2 * eta[0] * eta[2] * kappa[0]) * ca
        tmp4 = (10.0 * eta[1] - 2.0 * eta[3] + 1.0 / 6 * eta[5]) * sb
        tmp5 = -(2.0 * eta[1] ** 2 * kappa[2] - 1.0 / 6 * eta[1] ** 3 * kappa[3] - 1.0 / 2 * eta[1] * eta[3] * kappa[2]) * cb
        self.coeffs[1, 7] = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
        self.s_dot = lambda u: max(np.linalg.norm(self.coeffs[:, 1:].dot(np.array([1, 2.0 * u, 3.0 * u ** 2, 4.0 * u ** 3, 5.0 * u ** 4, 6.0 * u ** 5, 7.0 * u ** 6]))), 1e-06)
        self.f_length = lambda ue: quad(lambda u: self.s_dot(u), 0, ue)
        self.segment_length = self.f_length(1)[0]

    def calc_point(self, u):
        if False:
            i = 10
            return i + 15
        '\n        Eta3PathSegment::calc_point\n\n        input\n            u - parametric representation of a point along the segment, 0 <= u <= 1\n        returns\n            (x,y) of point along the segment\n        '
        assert 0 <= u <= 1
        return self.coeffs.dot(np.array([1, u, u ** 2, u ** 3, u ** 4, u ** 5, u ** 6, u ** 7]))

    def calc_deriv(self, u, order=1):
        if False:
            print('Hello World!')
        '\n        Eta3PathSegment::calc_deriv\n\n        input\n            u - parametric representation of a point along the segment, 0 <= u <= 1\n        returns\n            (d^nx/du^n,d^ny/du^n) of point along the segment, for 0 < n <= 2\n        '
        assert 0 <= u <= 1
        assert 0 < order <= 2
        if order == 1:
            return self.coeffs[:, 1:].dot(np.array([1, 2.0 * u, 3.0 * u ** 2, 4.0 * u ** 3, 5.0 * u ** 4, 6.0 * u ** 5, 7.0 * u ** 6]))
        return self.coeffs[:, 2:].dot(np.array([2, 6.0 * u, 12.0 * u ** 2, 20.0 * u ** 3, 30.0 * u ** 4, 42.0 * u ** 5]))

def test1():
    if False:
        return 10
    for i in range(10):
        path_segments = []
        start_pose = [0, 0, 0]
        end_pose = [4, 3.0, 0]
        kappa = [0, 0, 0, 0]
        eta = [i, i, 0, 0, 0, 0]
        path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
        path = Eta3Path(path_segments)
        ui = np.linspace(0, len(path_segments), 1001)
        pos = np.empty((2, ui.size))
        for (j, u) in enumerate(ui):
            pos[:, j] = path.calc_path_point(u)
        if show_animation:
            plt.plot(pos[0, :], pos[1, :])
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.pause(1.0)
    if show_animation:
        plt.close('all')

def test2():
    if False:
        return 10
    for i in range(10):
        path_segments = []
        start_pose = [0, 0, 0]
        end_pose = [4, 3.0, 0]
        kappa = [0, 0, 0, 0]
        eta = [0, 0, (i - 5) * 20, (5 - i) * 20, 0, 0]
        path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
        path = Eta3Path(path_segments)
        ui = np.linspace(0, len(path_segments), 1001)
        pos = np.empty((2, ui.size))
        for (j, u) in enumerate(ui):
            pos[:, j] = path.calc_path_point(u)
        if show_animation:
            plt.plot(pos[0, :], pos[1, :])
            plt.pause(1.0)
    if show_animation:
        plt.close('all')

def test3():
    if False:
        while True:
            i = 10
    path_segments = []
    start_pose = [0, 0, 0]
    end_pose = [4, 1.5, 0]
    kappa = [0, 0, 0, 0]
    eta = [4.27, 4.27, 0, 0, 0, 0]
    path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
    start_pose = [4, 1.5, 0]
    end_pose = [5.5, 1.5, 0]
    kappa = [0, 0, 0, 0]
    eta = [0, 0, 0, 0, 0, 0]
    path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
    start_pose = [5.5, 1.5, 0]
    end_pose = [7.4377, 1.8235, 0.6667]
    kappa = [0, 0, 1, 1]
    eta = [1.88, 1.88, 0, 0, 0, 0]
    path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
    start_pose = [7.4377, 1.8235, 0.6667]
    end_pose = [7.8, 4.3, 1.8]
    kappa = [1, 1, 0.5, 0]
    eta = [7, 10, 10, -10, 4, 4]
    path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
    start_pose = [7.8, 4.3, 1.8]
    end_pose = [5.4581, 5.8064, 3.3416]
    kappa = [0.5, 0, 0.5, 0]
    eta = [2.98, 2.98, 0, 0, 0, 0]
    path_segments.append(Eta3PathSegment(start_pose=start_pose, end_pose=end_pose, eta=eta, kappa=kappa))
    path = Eta3Path(path_segments)
    ui = np.linspace(0, len(path_segments), 1001)
    pos = np.empty((2, ui.size))
    for (i, u) in enumerate(ui):
        pos[:, i] = path.calc_path_point(u)
    if show_animation:
        plt.figure('Path from Reference')
        plt.plot(pos[0, :], pos[1, :])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Path')
        plt.pause(1.0)
        plt.show()

def main():
    if False:
        i = 10
        return i + 15
    '\n    recreate path from reference (see Table 1)\n    '
    test1()
    test2()
    test3()
if __name__ == '__main__':
    main()