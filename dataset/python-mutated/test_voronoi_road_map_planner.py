import conftest
from PathPlanning.VisibilityRoadMap import visibility_road_map as m

def test1():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)