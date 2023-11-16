import conftest
from PathPlanning.PotentialFieldPlanning import potential_field_planning as m

def test1():
    if False:
        i = 10
        return i + 15
    m.show_animation = False
    m.main()
if __name__ == '__main__':
    conftest.run_this_test(__file__)