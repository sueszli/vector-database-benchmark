import conftest
from PathPlanning.RRTStar import rrt_star as m

def test1():
    if False:
        return 10
    m.show_animation = False
    m.main()

def test_no_obstacle():
    if False:
        while True:
            i = 10
    obstacle_list = []
    rrt_star = m.RRTStar(start=[0, 0], goal=[6, 10], rand_area=[-2, 15], obstacle_list=obstacle_list)
    path = rrt_star.planning(animation=False)
    assert path is not None

def test_no_obstacle_and_robot_radius():
    if False:
        return 10
    obstacle_list = []
    rrt_star = m.RRTStar(start=[0, 0], goal=[6, 10], rand_area=[-2, 15], obstacle_list=obstacle_list, robot_radius=0.8)
    path = rrt_star.planning(animation=False)
    assert path is not None
if __name__ == '__main__':
    conftest.run_this_test(__file__)