import conftest
from PathPlanning.FrenetOptimalTrajectory import frenet_optimal_trajectory as m

def test1():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.SIM_LOOP = 5
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)