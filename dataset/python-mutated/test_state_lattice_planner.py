import conftest
from PathPlanning.StateLatticePlanner import state_lattice_planner as m

def test1():
    if False:
        print('Hello World!')
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)