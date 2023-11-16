from __future__ import annotations
import pytest
pytest
from bokeh.models import CustomJS, TapTool
from bokeh.plotting import figure
from tests.support.plugins.project import SinglePlotPage
from tests.support.util.selenium import RECORD
pytest_plugins = ('tests.support.plugins.project',)

@pytest.mark.selenium
class Test_TapTool:

    def test_tap_triggers_no_callback_without_hit(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            return 10
        plot = figure(height=800, width=1000, tools='')
        plot.rect(x=[1, 2], y=[1, 1], width=1, height=1)
        plot.add_tools(TapTool(callback=CustomJS(code=RECORD('indices', 'cb_data.source.selected.indices'))))
        plot.tags.append(CustomJS(name='custom-action', args=dict(p=plot), code=RECORD('junk', '10')))
        page = single_plot_page(plot)
        page.click_canvas_at_position(plot, 50, 50)
        page.eval_custom_action()
        assert page.results == {'junk': 10}
        assert page.has_no_console_errors()

    def test_tap_triggers_callback_with_indices(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            return 10
        plot = figure(height=800, width=1000, tools='')
        plot.rect(x=[1, 2], y=[1, 1], width=1, height=1)
        plot.add_tools(TapTool(callback=CustomJS(code=RECORD('indices', 'cb_data.source.selected.indices'))))
        page = single_plot_page(plot)
        page.click_canvas_at_position(plot, 400, 500)
        assert page.results['indices'] == [0]
        page.click_canvas_at_position(plot, 600, 300)
        assert page.results['indices'] == [1]
        assert page.has_no_console_errors()

    def test_tap_reports_all_indices_on_overlap(self, single_plot_page: SinglePlotPage) -> None:
        if False:
            print('Hello World!')
        plot = figure(height=800, width=1000, tools='')
        plot.rect(x=[1, 1], y=[1, 1], width=1, height=1)
        plot.add_tools(TapTool(callback=CustomJS(code=RECORD('indices', 'cb_data.source.selected.indices'))))
        page = single_plot_page(plot)
        page.click_canvas_at_position(plot, 400, 500)
        assert set(page.results['indices']) == {0, 1}
        assert page.has_no_console_errors()