from nose.tools import assert_equal
from pyecharts import options as opts

def test_area_color_in_item_styles():
    if False:
        i = 10
        return i + 15
    op = opts.ItemStyleOpts(area_color='red')
    assert_equal(op.opts['areaColor'], 'red')