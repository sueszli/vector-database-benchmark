import conftest
from Localization.unscented_kalman_filter import unscented_kalman_filter as m

def test1():
    if False:
        while True:
            i = 10
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)