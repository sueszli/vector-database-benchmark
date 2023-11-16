import numpy as np
import conftest
from PathPlanning.ReedsSheppPath import reeds_shepp_path_planning as m

def check_edge_condition(px, py, pyaw, start_x, start_y, start_yaw, end_x, end_y, end_yaw):
    if False:
        return 10
    assert abs(px[0] - start_x) <= 0.01
    assert abs(py[0] - start_y) <= 0.01
    assert abs(pyaw[0] - start_yaw) <= 0.01
    assert abs(px[-1] - end_x) <= 0.01
    assert abs(py[-1] - end_y) <= 0.01
    assert abs(pyaw[-1] - end_yaw) <= 0.01

def check_path_length(px, py, lengths):
    if False:
        i = 10
        return i + 15
    sum_len = sum((abs(length) for length in lengths))
    dpx = np.diff(px)
    dpy = np.diff(py)
    actual_len = sum((np.hypot(dx, dy) for (dx, dy) in zip(dpx, dpy)))
    diff_len = sum_len - actual_len
    assert diff_len <= 0.01

def test1():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.main()

def test2():
    if False:
        while True:
            i = 10
    N_TEST = 10
    np.random.seed(1234)
    for i in range(N_TEST):
        start_x = (np.random.rand() - 0.5) * 10.0
        start_y = (np.random.rand() - 0.5) * 10.0
        start_yaw = np.deg2rad((np.random.rand() - 0.5) * 180.0)
        end_x = (np.random.rand() - 0.5) * 10.0
        end_y = (np.random.rand() - 0.5) * 10.0
        end_yaw = np.deg2rad((np.random.rand() - 0.5) * 180.0)
        curvature = 1.0 / (np.random.rand() * 5.0)
        (px, py, pyaw, mode, lengths) = m.reeds_shepp_path_planning(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
        check_edge_condition(px, py, pyaw, start_x, start_y, start_yaw, end_x, end_y, end_yaw)
        check_path_length(px, py, lengths)
if __name__ == '__main__':
    conftest.run_this_test(__file__)