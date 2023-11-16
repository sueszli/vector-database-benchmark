from __future__ import annotations
import pytest
pytest
from bokeh.events import RangesUpdate
from bokeh.models import Circle, ColumnDataSource, CustomJS, Plot, Range1d, Rect, ResetTool, ZoomInTool
from tests.support.plugins.project import SinglePlotPage
from tests.support.util.selenium import RECORD
pytest_plugins = ('tests.support.plugins.project',)

def _make_plot():
    if False:
        i = 10
        return i + 15
    source = ColumnDataSource(dict(x=[1, 2], y=[1, 1]))
    plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
    plot.add_glyph(source, Rect(x='x', y='y', width=0.9, height=0.9))
    plot.add_tools(ResetTool(), ZoomInTool())
    code = RECORD('xrstart', 'p.x_range.start', final=False) + RECORD('xrend', 'p.x_range.end', final=False) + RECORD('yrstart', 'p.y_range.start', final=False) + RECORD('yrend', 'p.y_range.end')
    plot.tags.append(CustomJS(name='custom-action', args=dict(p=plot), code=code))
    plot.toolbar_sticky = False
    return plot

@pytest.mark.selenium
class Test_ResetTool:

    def test_deselected_by_default(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            i = 10
            return i + 15
        plot = _make_plot()
        page = single_plot_page(plot)
        [reset, _zoom_in] = page.get_toolbar_buttons(plot)
        assert 'active' not in reset.get_attribute('class')
        assert page.has_no_console_errors()

    def test_clicking_resets_range(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            return 10
        plot = _make_plot()
        page = single_plot_page(plot)
        [reset, zoom_in] = page.get_toolbar_buttons(plot)
        zoom_in.click()
        page.eval_custom_action()
        results = page.results
        assert results['xrstart'] != 0
        assert results['xrend'] != 1
        assert results['yrstart'] != 0
        assert results['yrend'] != 1
        reset.click()
        page.eval_custom_action()
        results = page.results
        assert results['xrstart'] == 0
        assert results['xrend'] == 1
        assert results['yrstart'] == 0
        assert results['yrend'] == 1
        assert page.has_no_console_errors()

    def test_clicking_resets_selection(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            while True:
                i = 10
        source = ColumnDataSource(dict(x=[1, 2], y=[1, 1]))
        source.selected.indices = [0]
        source.selected.line_indices = [0]
        source.selected.multiline_indices = {'0': [0]}
        plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
        plot.add_glyph(source, Circle(x='x', y='y', size=20))
        plot.add_tools(ResetTool())
        code = RECORD('indices', 's.selected.indices') + RECORD('line_indices', 's.selected.line_indices') + RECORD('multiline_indices', 's.selected.multiline_indices')
        plot.tags.append(CustomJS(name='custom-action', args=dict(s=source), code=code))
        plot.toolbar_sticky = False
        page = single_plot_page(plot)
        page.eval_custom_action()
        results = page.results
        assert results['indices'] == [0]
        assert results['line_indices'] == [0]
        assert results['multiline_indices'] == {'0': [0]}
        [reset] = page.get_toolbar_buttons(plot)
        reset.click()
        page.eval_custom_action()
        results = page.results
        assert results['indices'] == []
        assert results['line_indices'] == []
        assert results['multiline_indices'] == {}
        assert page.has_no_console_errors()

    def test_ranges_udpate(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            i = 10
            return i + 15
        source = ColumnDataSource(dict(x=[1, 2], y=[1, 1]))
        plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
        plot.add_glyph(source, Rect(x='x', y='y', width=0.9, height=0.9))
        plot.add_tools(ResetTool(), ZoomInTool())
        code = RECORD('event_name', 'cb_obj.event_name', final=False) + RECORD('x0', 'cb_obj.x0', final=False) + RECORD('x1', 'cb_obj.x1', final=False) + RECORD('y0', 'cb_obj.y0', final=False) + RECORD('y1', 'cb_obj.y1')
        plot.js_on_event(RangesUpdate, CustomJS(code=code))
        plot.tags.append(CustomJS(name='custom-action', code=''))
        plot.toolbar_sticky = False
        page = single_plot_page(plot)
        [reset, zoom_in] = page.get_toolbar_buttons(plot)
        zoom_in.click()
        reset.click()
        page.eval_custom_action()
        results = page.results
        assert results['event_name'] == 'rangesupdate'
        assert results['x0'] == 0
        assert results['x1'] == 1
        assert results['y0'] == 0
        assert results['y1'] == 1
        assert page.has_no_console_errors()