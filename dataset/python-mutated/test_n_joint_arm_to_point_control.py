import conftest
from ArmNavigation.n_joint_arm_to_point_control import n_joint_arm_to_point_control as m
import random
random.seed(12345)

def test1():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.animation()
if __name__ == '__main__':
    conftest.run_this_test(__file__)