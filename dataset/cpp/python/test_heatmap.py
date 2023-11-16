import random
from unittest.mock import patch

from nose.tools import assert_equal

from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker


@patch("pyecharts.render.engine.write_utf8_html_file")
def test_heatmap_base(fake_writer):
    value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
    c = (
        HeatMap()
        .add_xaxis(Faker.clock)
        .add_yaxis("series0", Faker.week, value)
        .set_global_opts(visualmap_opts=opts.VisualMapOpts())
    )
    c.render()
    _, content = fake_writer.call_args[0]
    assert_equal(c.theme, "white")
    assert_equal(c.renderer, "canvas")
