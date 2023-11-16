import conftest
from PathPlanning.Eta3SplinePath import eta3_spline_path as m

def test_1():
    if False:
        print('Hello World!')
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)