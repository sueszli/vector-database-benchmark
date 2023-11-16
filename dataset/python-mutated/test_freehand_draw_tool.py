from __future__ import annotations
import pytest
pytest
import time
from bokeh.application.handlers.function import ModifyDoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, FreehandDrawTool, MultiLine, Plot, Range1d
from tests.support.plugins.project import BokehServerPage, SinglePlotPage
from tests.support.util.compare import cds_data_almost_equal
from tests.support.util.selenium import RECORD
pytest_plugins = ('tests.support.plugins.project',)

def _make_plot(num_objects=0):
    if False:
        return 10
    source = ColumnDataSource(dict(xs=[], ys=[]))
    plot = Plot(height=400, width=400, x_range=Range1d(0, 3), y_range=Range1d(0, 3), min_border=0)
    renderer = plot.add_glyph(source, MultiLine(xs='xs', ys='ys'))
    tool = FreehandDrawTool(num_objects=num_objects, renderers=[renderer])
    plot.add_tools(tool)
    plot.toolbar.active_multi = tool
    code = RECORD('xs', 'source.data.xs', final=False) + RECORD('ys', 'source.data.ys')
    plot.tags.append(CustomJS(name='custom-action', args=dict(source=source), code=code))
    plot.toolbar_sticky = False
    return plot

def _make_server_plot(expected, num_objects=0) -> tuple[ModifyDoc, Plot]:
    if False:
        return 10
    plot = Plot(height=400, width=400, x_range=Range1d(0, 3), y_range=Range1d(0, 3), min_border=0)

    def modify_doc(doc):
        if False:
            while True:
                i = 10
        source = ColumnDataSource(dict(xs=[], ys=[]))
        renderer = plot.add_glyph(source, MultiLine(xs='xs', ys='ys'))
        tool = FreehandDrawTool(num_objects=num_objects, renderers=[renderer])
        plot.add_tools(tool)
        plot.toolbar.active_multi = tool
        div = Div(text='False')

        def cb(attr, old, new):
            if False:
                return 10
            if cds_data_almost_equal(new, expected):
                div.text = 'True'
        source.on_change('data', cb)
        code = RECORD('matches', 'div.text')
        plot.tags.append(CustomJS(name='custom-action', args=dict(div=div), code=code))
        doc.add_root(column(plot, div))
    return (modify_doc, plot)

@pytest.mark.selenium
class Test_FreehandDrawTool:

    def test_selected_by_default(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            while True:
                i = 10
        plot = _make_plot()
        page = single_plot_page(plot)
        [button] = page.get_toolbar_buttons(plot)
        assert 'active' in button.get_attribute('class')
        assert page.has_no_console_errors()

    def test_can_be_deselected_and_selected(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            print('Hello World!')
        plot = _make_plot()
        page = single_plot_page(plot)
        [button] = page.get_toolbar_buttons(plot)
        assert 'active' in button.get_attribute('class')
        [button] = page.get_toolbar_buttons(plot)
        button.click()
        assert 'active' not in button.get_attribute('class')
        [button] = page.get_toolbar_buttons(plot)
        button.click()
        assert 'active' in button.get_attribute('class')
        assert page.has_no_console_errors()

    def test_drag_triggers_draw(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        plot = _make_plot()
        page = single_plot_page(plot)
        page.drag_canvas_at_position(plot, 200, 200, 50, 50)
        page.eval_custom_action()
        expected = {'xs': [[1.6216216216216217, 2.027027027027027, 2.027027027027027, 2.027027027027027]], 'ys': [[1.5, 1.125, 1.125, 1.125]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_num_object_limits_lines(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        plot = _make_plot(num_objects=1)
        page = single_plot_page(plot)
        page.drag_canvas_at_position(plot, 200, 200, 50, 50)
        page.drag_canvas_at_position(plot, 100, 100, 100, 100)
        page.eval_custom_action()
        expected = {'xs': [[0.8108108108108109, 1.6216216216216217, 1.6216216216216217, 1.6216216216216217]], 'ys': [[2.25, 1.5, 1.5, 1.5]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_freehand_draw_syncs_to_server(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            return 10
        expected = {'xs': [[1.6216216216216217, 2.027027027027027, 2.027027027027027, 2.027027027027027]], 'ys': [[1.5, 1.125, 1.125, 1.125]]}
        (modify_doc, plot) = _make_server_plot(expected)
        page = bokeh_server_page(modify_doc)
        page.drag_canvas_at_position(plot, 200, 200, 50, 50)
        page.eval_custom_action()
        assert page.results == {'matches': 'True'}

    def test_line_delete_syncs_to_server(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            print('Hello World!')
        expected = {'xs': [], 'ys': []}
        (modify_doc, plot) = _make_server_plot(expected)
        page = bokeh_server_page(modify_doc)
        page.drag_canvas_at_position(plot, 200, 200, 50, 50)
        page.click_canvas_at_position(plot, 200, 200)
        time.sleep(0.4)
        page.send_keys('\ue003')
        page.eval_custom_action()
        assert page.results == {'matches': 'True'}