from __future__ import annotations
import pytest
pytest
import time
from bokeh.application.handlers.function import ModifyDoc
from bokeh.layouts import column
from bokeh.models import Circle, ColumnDataSource, CustomJS, Div, MultiLine, Plot, PolyDrawTool, Range1d
from tests.support.plugins.project import BokehServerPage, SinglePlotPage
from tests.support.util.compare import cds_data_almost_equal
from tests.support.util.selenium import RECORD
pytest_plugins = ('tests.support.plugins.project',)

def _make_plot(num_objects=0, drag=True, vertices=False):
    if False:
        i = 10
        return i + 15
    source = ColumnDataSource(dict(xs=[[1, 2]], ys=[[1, 1]]))
    plot = Plot(height=400, width=400, x_range=Range1d(0, 3), y_range=Range1d(0, 3), min_border=0)
    renderer = plot.add_glyph(source, MultiLine(xs='xs', ys='ys'))
    tool = PolyDrawTool(num_objects=num_objects, drag=drag, renderers=[renderer])
    if vertices:
        psource = ColumnDataSource(dict(x=[], y=[]))
        prenderer = plot.add_glyph(psource, Circle(x='x', y='y', size=10))
        tool.vertex_renderer = prenderer
    plot.add_tools(tool)
    plot.toolbar.active_multi = tool
    code = RECORD('xs', 'source.data.xs', final=False) + RECORD('ys', 'source.data.ys')
    plot.tags.append(CustomJS(name='custom-action', args=dict(source=source), code=code))
    plot.toolbar_sticky = False
    return plot

def _make_server_plot(expected) -> tuple[ModifyDoc, Plot]:
    if False:
        for i in range(10):
            print('nop')
    plot = Plot(height=400, width=400, x_range=Range1d(0, 3), y_range=Range1d(0, 3), min_border=0)

    def modify_doc(doc):
        if False:
            for i in range(10):
                print('nop')
        source = ColumnDataSource(dict(xs=[[1, 2]], ys=[[1, 1]]))
        renderer = plot.add_glyph(source, MultiLine(xs='xs', ys='ys'))
        tool = PolyDrawTool(renderers=[renderer])
        plot.add_tools(tool)
        plot.toolbar.active_multi = tool
        div = Div(text='False')

        def cb(attr, old, new):
            if False:
                print('Hello World!')
            if cds_data_almost_equal(new, expected):
                div.text = 'True'
        source.on_change('data', cb)
        code = RECORD('matches', 'div.text')
        plot.tags.append(CustomJS(name='custom-action', args=dict(div=div), code=code))
        doc.add_root(column(plot, div))
    return (modify_doc, plot)

@pytest.mark.selenium
class Test_PolyDrawTool:

    def test_selected_by_default(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            print('Hello World!')
        plot = _make_plot()
        page = single_plot_page(plot)
        [button] = page.get_toolbar_buttons(plot)
        assert 'active' in button.get_attribute('class')
        assert page.has_no_console_errors()

    def test_can_be_deselected_and_selected(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            while True:
                i = 10
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

    def test_double_click_triggers_draw(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            i = 10
            return i + 15
        plot = _make_plot()
        page = single_plot_page(plot)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.5)
        page.eval_custom_action()
        expected = {'xs': [[1, 2], [1.6216216216216217, 2.4324324324324325]], 'ys': [[1, 1], [1.5, 0.75]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_click_snaps_to_vertex(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        plot = _make_plot(vertices=True)
        page = single_plot_page(plot)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.click_canvas_at_position(plot, 300, 300)
        time.sleep(0.5)
        page.double_click_canvas_at_position(plot, 201, 201)
        time.sleep(0.5)
        page.eval_custom_action()
        expected = {'xs': [[1, 2], [1.6216216216216217, 2.4324324324324325, 1.6216216216216217]], 'ys': [[1, 1], [1.5, 0.75, 1.5]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_drag_moves_multi_line(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            while True:
                i = 10
        plot = _make_plot()
        page = single_plot_page(plot)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.4)
        page.drag_canvas_at_position(plot, 200, 200, 70, 50)
        page.eval_custom_action()
        expected = {'xs': [[1, 2], [2.1891891891891895, 3]], 'ys': [[1, 1], [1.125, 0.375]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_drag_does_not_move_multi_line(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        plot = _make_plot(drag=False)
        page = single_plot_page(plot)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.4)
        page.drag_canvas_at_position(plot, 200, 200, 70, 53)
        page.eval_custom_action()
        expected = {'xs': [[1, 2], [1.6216216216216217, 2.4324324324324325]], 'ys': [[1, 1], [1.5, 0.75]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_num_object_limits_multi_lines(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            while True:
                i = 10
        plot = _make_plot(num_objects=1)
        page = single_plot_page(plot)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.4)
        page.drag_canvas_at_position(plot, 200, 200, 70, 50)
        page.eval_custom_action()
        expected = {'xs': [[2.1891891891891895, 3]], 'ys': [[1.125, 0.375]]}
        assert cds_data_almost_equal(page.results, expected)
        assert page.has_no_console_errors()

    def test_poly_draw_syncs_to_server(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            print('Hello World!')
        expected = {'xs': [[1, 2], [1.6216216216216217, 2.4324324324324325]], 'ys': [[1, 1], [1.5, 0.75]]}
        (modify_doc, plot) = _make_server_plot(expected)
        page = bokeh_server_page(modify_doc)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.5)
        page.eval_custom_action()
        assert page.results == {'matches': 'True'}

    def test_poly_drag_syncs_to_server(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            return 10
        expected = {'xs': [[1, 2], [2.1891891891891895, 3]], 'ys': [[1, 1], [1.125, 0.375]]}
        (modify_doc, plot) = _make_server_plot(expected)
        page = bokeh_server_page(modify_doc)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.4)
        page.drag_canvas_at_position(plot, 200, 200, 70, 50)
        page.eval_custom_action()
        assert page.results == {'matches': 'True'}

    def test_poly_delete_syncs_to_server(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            i = 10
            return i + 15
        expected = {'xs': [[1, 2]], 'ys': [[1, 1]]}
        (modify_doc, plot) = _make_server_plot(expected)
        page = bokeh_server_page(modify_doc)
        page.double_click_canvas_at_position(plot, 200, 200)
        page.double_click_canvas_at_position(plot, 300, 300)
        time.sleep(0.4)
        page.click_canvas_at_position(plot, 200, 200)
        time.sleep(0.4)
        page.send_keys('\ue003')
        time.sleep(0.4)
        page.eval_custom_action()
        assert page.results == {'matches': 'True'}