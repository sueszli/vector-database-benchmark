import conftest
from PathPlanning.LQRPlanner import lqr_planner as m

def test_1():
    if False:
        return 10
    m.SHOW_ANIMATION = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)