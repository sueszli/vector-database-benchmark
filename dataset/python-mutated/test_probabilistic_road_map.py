import conftest
import numpy as np
from PathPlanning.ProbabilisticRoadMap import probabilistic_road_map

def test1():
    if False:
        i = 10
        return i + 15
    probabilistic_road_map.show_animation = False
    probabilistic_road_map.main(rng=np.random.default_rng(1233))
if __name__ == '__main__':
    conftest.run_this_test(__file__)