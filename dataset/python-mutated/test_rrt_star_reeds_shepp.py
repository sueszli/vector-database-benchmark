import conftest
from PathPlanning.RRTStarReedsShepp import rrt_star_reeds_shepp as m

def test1():
    if False:
        while True:
            i = 10
    m.show_animation = False
    m.main(max_iter=5)
obstacleList = [(5, 5, 1), (4, 6, 1), (4, 8, 1), (4, 10, 1), (6, 5, 1), (7, 5, 1), (8, 6, 1), (8, 8, 1), (8, 10, 1)]
start = [0.0, 0.0, m.np.deg2rad(0.0)]
goal = [6.0, 7.0, m.np.deg2rad(90.0)]

def test2():
    if False:
        print('Hello World!')
    step_size = 0.2
    rrt_star_reeds_shepp = m.RRTStarReedsShepp(start, goal, obstacleList, [-2.0, 15.0], max_iter=100, step_size=step_size)
    rrt_star_reeds_shepp.set_random_seed(seed=8)
    path = rrt_star_reeds_shepp.planning(animation=False)
    for i in range(len(path) - 1):
        assert m.math.dist(path[i][0:2], path[i + 1][0:2]) < step_size + 1e-14

def test3():
    if False:
        return 10
    step_size = 20
    rrt_star_reeds_shepp = m.RRTStarReedsShepp(start, goal, obstacleList, [-2.0, 15.0], max_iter=100, step_size=step_size)
    rrt_star_reeds_shepp.set_random_seed(seed=8)
    path = rrt_star_reeds_shepp.planning(animation=False)
    for i in range(len(path) - 1):
        assert m.math.dist(path[i][0:2], path[i + 1][0:2]) < step_size + 1e-14
if __name__ == '__main__':
    conftest.run_this_test(__file__)