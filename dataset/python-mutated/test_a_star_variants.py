import PathPlanning.AStar.a_star_variants as a_star
import conftest

def test_1():
    if False:
        while True:
            i = 10
    a_star.show_animation = False
    a_star.use_beam_search = True
    a_star.main()
    reset_all()
    a_star.use_iterative_deepening = True
    a_star.main()
    reset_all()
    a_star.use_dynamic_weighting = True
    a_star.main()
    reset_all()
    a_star.use_theta_star = True
    a_star.main()
    reset_all()
    a_star.use_jump_point = True
    a_star.main()
    reset_all()

def reset_all():
    if False:
        while True:
            i = 10
    a_star.show_animation = False
    a_star.use_beam_search = False
    a_star.use_iterative_deepening = False
    a_star.use_dynamic_weighting = False
    a_star.use_theta_star = False
    a_star.use_jump_point = False
if __name__ == '__main__':
    conftest.run_this_test(__file__)