from unittest.mock import patch
from nose.tools import assert_equal
from pyecharts.faker import Faker, Collector

def test_rand_color():
    if False:
        while True:
            i = 10
    rand_color = Faker.rand_color()
    assert rand_color is not None

def test_img_path():
    if False:
        for i in range(10):
            print('nop')
    assert_equal(Faker.img_path(path='/usr/local'), '/usr/local')

def test_collector():
    if False:
        print('Hello World!')

    def _add(x, y):
        if False:
            print('Hello World!')
        return x + y
    c = Collector()
    c.funcs(_add)
    assert_equal(c.charts[0][1], '_add')