import conftest
from PathPlanning.DepthFirstSearch import depth_first_search as m

def test_1():
    if False:
        i = 10
        return i + 15
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)