import conftest
from PathPlanning.AStar import a_star_searching_from_two_side as m

def test1():
    if False:
        print('Hello World!')
    m.show_animation = False
    m.main(800)

def test2():
    if False:
        return 10
    m.show_animation = False
    m.main(5000)
if __name__ == '__main__':
    conftest.run_this_test(__file__)