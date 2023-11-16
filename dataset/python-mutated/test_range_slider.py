from __future__ import annotations
import pytest
pytest
from time import sleep
from bokeh.layouts import column
from bokeh.models import Circle, ColumnDataSource, CustomJS, Plot, Range1d, RangeSlider
from bokeh.models.formatters import BasicTickFormatter
from tests.support.plugins.project import BokehModelPage, BokehServerPage
from tests.support.util.selenium import RECORD, Keys, drag_range_slider, find_element_for, find_elements_for, get_slider_bar_color, get_slider_title_text, get_slider_title_value, select_element_and_press_key
pytest_plugins = ('tests.support.plugins.project',)

@pytest.mark.selenium
class Test_RangeSlider:

    def test_display(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            i = 10
            return i + 15
        slider = RangeSlider(start=0, end=10, value=(1, 5), width=300)
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert page.has_no_console_errors()

    def test_displays_title(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        slider = RangeSlider(start=0, end=10, value=(1, 5), title='bar', width=300)
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert get_slider_title_text(page.driver, slider) == 'bar: 1 .. 5'
        assert get_slider_title_value(page.driver, slider) == '1 .. 5'
        assert page.has_no_console_errors()

    def test_displays_title_scientific(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            return 10
        slider = RangeSlider(start=0, end=1e-05, step=1e-06, value=(1e-06, 8e-06), title='bar', format=BasicTickFormatter(precision=2), width=300)
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        t0 = get_slider_title_text(page.driver, slider)
        t1 = get_slider_title_value(page.driver, slider)
        assert t0 == 'bar: 1.00e−6 .. 8.00e−6'
        assert t1 == '1.00e−6 .. 8.00e−6'
        assert page.has_no_console_errors()

    def test_title_updates(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        slider = RangeSlider(start=0, end=10, value=(1, 9), title='bar', width=300)
        page = bokeh_model_page(slider)
        assert get_slider_title_value(page.driver, slider) == '1 .. 9'
        drag_range_slider(page.driver, slider, 'lower', 50)
        value = get_slider_title_value(page.driver, slider).split()[0]
        assert float(value) > 1
        assert float(value) == int(value)
        drag_range_slider(page.driver, slider, 'lower', 50)
        value = get_slider_title_value(page.driver, slider).split()[0]
        assert float(value) > 2
        drag_range_slider(page.driver, slider, 'lower', -135)
        value = get_slider_title_value(page.driver, slider).split()[0]
        assert float(value) == 0
        assert page.has_no_console_errors()

    def test_keypress_event(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        slider = RangeSlider(start=0, end=10, value=(1, 5), title='bar', width=300)
        page = bokeh_model_page(slider)
        handle_lower = find_element_for(page.driver, slider, '.noUi-handle-lower')
        handle_upper = find_element_for(page.driver, slider, '.noUi-handle-upper')
        select_element_and_press_key(page.driver, handle_lower, Keys.ARROW_RIGHT, press_number=1)
        assert get_slider_title_value(page.driver, slider) == '2 .. 5'
        select_element_and_press_key(page.driver, handle_lower, Keys.ARROW_LEFT, press_number=5)
        assert get_slider_title_value(page.driver, slider) == '0 .. 5'
        select_element_and_press_key(page.driver, handle_lower, Keys.ARROW_RIGHT, press_number=11)
        assert get_slider_title_value(page.driver, slider) == '5 .. 5'
        select_element_and_press_key(page.driver, handle_upper, Keys.ARROW_RIGHT, press_number=1)
        assert get_slider_title_value(page.driver, slider) == '5 .. 6'
        select_element_and_press_key(page.driver, handle_upper, Keys.ARROW_LEFT, press_number=2)
        assert get_slider_title_value(page.driver, slider) == '5 .. 5'
        select_element_and_press_key(page.driver, handle_upper, Keys.ARROW_RIGHT, press_number=6)
        assert get_slider_title_value(page.driver, slider) == '5 .. 10'
        assert page.has_no_console_errors()

    def test_displays_bar_color(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        slider = RangeSlider(start=0, end=10, value=(1, 5), title='bar', width=300, bar_color='red')
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert get_slider_bar_color(page.driver, slider) == 'rgba(255, 0, 0, 1)'
        assert page.has_no_console_errors()

    def test_js_on_change_executes(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            i = 10
            return i + 15
        slider = RangeSlider(start=0, end=10, value=(1, 5), title='bar', width=300)
        slider.js_on_change('value', CustomJS(code=RECORD('value', 'cb_obj.value')))
        page = bokeh_model_page(slider)
        drag_range_slider(page.driver, slider, 'lower', 150)
        results = page.results
        assert float(results['value'][0]) > 1
        assert float(results['value'][1]) == 5
        drag_range_slider(page.driver, slider, 'lower', 150)
        results = page.results
        assert float(results['value'][0]) > 1
        assert float(results['value'][1]) > 5
        assert page.has_no_console_errors()

    def test_server_on_change_round_trip(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        slider = RangeSlider(start=0, end=10, value=(1, 9), title='bar', width=300)

        def modify_doc(doc):
            if False:
                return 10
            source = ColumnDataSource(dict(x=[1, 2], y=[1, 1], val=['a', 'b']))
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
            plot.add_glyph(source, Circle(x='x', y='y', size=20))
            plot.tags.append(CustomJS(name='custom-action', args=dict(s=source), code=RECORD('data', 's.data')))

            def cb(attr, old, new):
                if False:
                    while True:
                        i = 10
                source.data['val'] = [old, new]
            slider.on_change('value', cb)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_range_slider(page.driver, slider, 'lower', 50)
        page.eval_custom_action()
        results = page.results
        (old, new) = results['data']['val']
        assert float(old[0]) == 1
        assert float(new[0]) > 1
        drag_range_slider(page.driver, slider, 'lower', 50)
        page.eval_custom_action()
        results = page.results
        (old, new) = results['data']['val']
        assert float(new[0]) > 2
        drag_range_slider(page.driver, slider, 'lower', -135)
        page.eval_custom_action()
        results = page.results
        (old, new) = results['data']['val']
        assert float(new[0]) == 0

    def test_server_bar_color_updates(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            while True:
                i = 10
        slider = RangeSlider(start=0, end=10, value=(1, 5), title='bar', width=300)

        def modify_doc(doc):
            if False:
                return 10
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)

            def cb(attr, old, new):
                if False:
                    i = 10
                    return i + 15
                slider.bar_color = 'rgba(255, 255, 0, 1)'
            slider.on_change('value', cb)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_range_slider(page.driver, slider, 'lower', 150)
        sleep(1)
        assert get_slider_bar_color(page.driver, slider) == 'rgba(255, 255, 0, 1)'