import conftest
from PathPlanning.DStarLite import d_star_lite as m

def test_1():
    if False:
        print('Hello World!')
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)