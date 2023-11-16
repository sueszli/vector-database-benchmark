import conftest
from Control.move_to_pose import move_to_pose as m

def test_1():
    if False:
        while True:
            i = 10
    m.show_animation = False
    m.main()

def test_2():
    if False:
        while True:
            i = 10
    '\n    This unit test tests the move_to_pose.py program for a MAX_LINEAR_SPEED and\n    MAX_ANGULAR_SPEED\n    '
    m.show_animation = False
    m.MAX_LINEAR_SPEED = 11
    m.MAX_ANGULAR_SPEED = 5
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)