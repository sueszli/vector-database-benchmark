import conftest
from PathTracking.model_predictive_speed_and_steer_control import model_predictive_speed_and_steer_control as m

def test_1():
    if False:
        i = 10
        return i + 15
    m.show_animation = False
    m.main()

def test_2():
    if False:
        for i in range(10):
            print('nop')
    m.show_animation = False
    m.main2()
if __name__ == '__main__':
    conftest.run_this_test(__file__)