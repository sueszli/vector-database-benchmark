import conftest
from Mapping.normal_vector_estimation import normal_vector_estimation as m
import random
random.seed(12345)

def test_1():
    if False:
        i = 10
        return i + 15
    m.show_animation = False
    m.main1()

def test_2():
    if False:
        i = 10
        return i + 15
    m.show_animation = False
    m.main2()
if __name__ == '__main__':
    conftest.run_this_test(__file__)