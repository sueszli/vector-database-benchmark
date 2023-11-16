import conftest
from AerialNavigation.drone_3d_trajectory_following import drone_3d_trajectory_following as m

def test1():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)