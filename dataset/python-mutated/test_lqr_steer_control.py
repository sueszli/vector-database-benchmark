import conftest
from PathTracking.lqr_steer_control import lqr_steer_control as m

def test1():
    if False:
        return 10
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)