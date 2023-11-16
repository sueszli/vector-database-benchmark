from typing import Iterable
from unittest.mock import patch
from nose.tools import assert_equal, assert_in, assert_true
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Tab
from pyecharts.commons.utils import OrderedSet
from pyecharts.components import Table
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType

def _create_bar() -> Bar:
    if False:
        return 10
    return Bar(init_opts=opts.InitOpts(theme=ThemeType.ESSOS)).add_xaxis(Faker.week).add_yaxis('商家A', [1, 2, 3, 4, 5, 6, 7])

def _create_line() -> Line:
    if False:
        print('Hello World!')
    return Line().add_xaxis(Faker.week).add_yaxis('商家A', [7, 6, 5, 4, 3, 2, 1])

def _create_table() -> Table:
    if False:
        i = 10
        return i + 15
    table = Table()
    headers = ['City name', 'Area', 'Population', 'Annual Rainfall']
    rows = [['Brisbane', 5905, 1857594, 1146.4], ['Adelaide', 1295, 1158259, 600.5], ['Darwin', 112, 120900, 1714.7]]
    table.add(headers, rows)
    return table

@patch('pyecharts.render.engine.write_utf8_html_file')
def test_tab_base(fake_writer):
    if False:
        while True:
            i = 10
    bar = _create_bar()
    line = _create_line()
    tab = Tab().add(bar, 'bar-example').add(line, 'line-example')
    tab.render()
    (_, content) = fake_writer.call_args[0]
    assert_in('bar-example', content)
    assert_in('line-example', content)

def test_tab_render_embed():
    if False:
        i = 10
        return i + 15
    bar = _create_bar()
    line = _create_line()
    content = Tab().add(bar, 'bar').add(line, 'line').render_embed()
    assert_true(len(content) > 8000)

def test_tab_render_notebook():
    if False:
        while True:
            i = 10
    from pyecharts.globals import CurrentConfig, NotebookType
    CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK
    tab = Tab()
    tab.add(_create_line(), 'line-example')
    tab.add(_create_bar(), 'bar-example')
    tab.add(_create_table(), 'table-example')
    html = tab.render_notebook().__html__()
    assert_in('City name', html)

def test_page_jshost_default():
    if False:
        while True:
            i = 10
    bar = _create_bar()
    tab = Tab().add(bar, 'bar')
    assert_equal(tab.js_host, 'https://assets.pyecharts.org/assets/v5/')

def test_tab_jshost_custom():
    if False:
        for i in range(10):
            print('nop')
    from pyecharts.globals import CurrentConfig
    default_host = CurrentConfig.ONLINE_HOST
    custom_host = 'http://localhost:8888/assets/'
    CurrentConfig.ONLINE_HOST = custom_host
    bar = _create_bar()
    line = _create_line()
    tab = Tab().add(bar, 'bar').add(line, 'line')
    assert_equal(tab.js_host, custom_host)
    CurrentConfig.ONLINE_HOST = default_host

def test_tab_iterable():
    if False:
        print('Hello World!')
    tab = Tab()
    assert_true(isinstance(tab, Iterable))

def test_tab_attr():
    if False:
        return 10
    tab = Tab()
    assert_true(isinstance(tab.js_functions, OrderedSet))
    assert_true(isinstance(tab._charts, list))

def test_tab_with_chart_container():
    if False:
        for i in range(10):
            print('nop')
    tab = Tab(tab_css_opts=opts.TabChartGlobalOpts(is_enable=False, tab_base_css={'overflow': 'hidden'}))
    assert_true(isinstance(tab._charts, list))