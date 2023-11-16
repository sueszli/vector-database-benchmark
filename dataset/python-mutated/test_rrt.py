import conftest
import random
from PathPlanning.RRT import rrt as m
from PathPlanning.RRT import rrt_with_pathsmoothing as m1
random.seed(12345)

def test1():
    if False:
        print('Hello World!')
    m.show_animation = False
    m.main(gx=1.0, gy=1.0)

def test2():
    if False:
        while True:
            i = 10
    m1.show_animation = False
    m1.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)