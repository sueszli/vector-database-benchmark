import conftest
from SLAM.EKFSLAM import ekf_slam as m

def test_1():
    if False:
        while True:
            i = 10
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)