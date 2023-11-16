from __future__ import annotations
import pytest
pytest
from datetime import date, datetime, timedelta
from time import sleep
from bokeh.layouts import column
from bokeh.models import Circle, ColumnDataSource, CustomJS, DateSlider, Plot, Range1d
from tests.support.plugins.project import BokehModelPage, BokehServerPage
from tests.support.util.selenium import RECORD, drag_slider, find_elements_for, get_slider_bar_color, get_slider_title_text, get_slider_title_value
pytest_plugins = ('tests.support.plugins.project',)
start = date(2017, 8, 3)
end = date(2017, 8, 10)
value = start + timedelta(days=1)

@pytest.mark.selenium
class Test_DateSlider:

    def test_display(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        slider = DateSlider(start=start, end=end, value=value, width=300)
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert page.has_no_console_errors()

    def test_displays_title(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        slider = DateSlider(start=start, end=end, value=value, width=300)
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert get_slider_title_text(page.driver, slider) == '04 Aug 2017'
        assert get_slider_title_value(page.driver, slider) == '04 Aug 2017'
        assert page.has_no_console_errors()

    def test_title_updates(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            return 10
        slider = DateSlider(start=start, end=end, value=value, width=300)
        page = bokeh_model_page(slider)
        assert get_slider_title_value(page.driver, slider) == '04 Aug 2017'
        drag_slider(page.driver, slider, 50)
        assert get_slider_title_value(page.driver, slider) > '04 Aug 2017'
        drag_slider(page.driver, slider, -70)
        assert get_slider_title_value(page.driver, slider) == '03 Aug 2017'
        assert page.has_no_console_errors()

    def test_displays_bar_color(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        slider = DateSlider(start=start, end=end, value=value, width=300, bar_color='red')
        page = bokeh_model_page(slider)
        children = find_elements_for(page.driver, slider, 'div.bk-input-group > div')
        assert len(children) == 2
        assert get_slider_bar_color(page.driver, slider) == 'rgba(255, 0, 0, 1)'
        assert page.has_no_console_errors()

    def test_js_on_change_executes(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        slider = DateSlider(start=start, end=end, value=value, width=300)
        slider.js_on_change('value', CustomJS(code=RECORD('value', 'cb_obj.value')))
        page = bokeh_model_page(slider)
        drag_slider(page.driver, slider, 150)
        results = page.results
        assert datetime.fromtimestamp(results['value'] / 1000) > datetime(*date.fromisoformat('2017-08-04').timetuple()[:3])
        assert page.has_no_console_errors()

    def test_server_on_change_round_trip(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            return 10
        slider = DateSlider(start=start, end=end, value=value, width=300, step=1)

        def modify_doc(doc):
            if False:
                print('Hello World!')
            source = ColumnDataSource(dict(x=[1, 2], y=[1, 1], val=['a', 'b']))
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
            plot.add_glyph(source, Circle(x='x', y='y', size=20))
            plot.tags.append(CustomJS(name='custom-action', args=dict(s=source), code=RECORD('data', 's.data')))

            def cb(attr, old, new):
                if False:
                    print('Hello World!')
                iso_date = slider.value_as_date.isoformat()
                source.data['val'] = [iso_date, iso_date]
            slider.on_change('value', cb)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_slider(page.driver, slider, 50)
        page.eval_custom_action()
        results = page.results
        new = results['data']['val']
        assert new[0] > '2017-08-04'
        drag_slider(page.driver, slider, -70)
        page.eval_custom_action()
        results = page.results
        new = results['data']['val']
        assert new[0] == '2017-08-03'

    def test_server_callback_value_vs_value_throttled(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            print('Hello World!')
        junk = dict(v=0, vt=0)
        slider = DateSlider(start=start, end=end, value=value, width=300)

        def modify_doc(doc):
            if False:
                for i in range(10):
                    print('nop')
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)

            def cbv(attr, old, new):
                if False:
                    i = 10
                    return i + 15
                junk['v'] += 1

            def cbvt(attr, old, new):
                if False:
                    return 10
                junk['vt'] += 1
            slider.on_change('value', cbv)
            slider.on_change('value_throttled', cbvt)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_slider(page.driver, slider, 30, release=False)
        sleep(1)
        drag_slider(page.driver, slider, 30, release=False)
        sleep(1)
        drag_slider(page.driver, slider, 30, release=False)
        sleep(1)
        drag_slider(page.driver, slider, 30, release=True)
        sleep(1)
        assert junk['v'] == 4
        assert junk['vt'] == 1

    def test_server_bar_color_updates(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        slider = DateSlider(start=start, end=end, value=value, width=300, bar_color='red')

        def modify_doc(doc):
            if False:
                for i in range(10):
                    print('nop')
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)

            def cb(attr, old, new):
                if False:
                    while True:
                        i = 10
                slider.bar_color = 'rgba(255, 255, 0, 1)'
            slider.on_change('value', cb)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_slider(page.driver, slider, 150)
        sleep(1)
        assert get_slider_bar_color(page.driver, slider) == 'rgba(255, 255, 0, 1)'

    def test_server_title_updates(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            while True:
                i = 10
        slider = DateSlider(start=start, end=end, value=value, width=300)

        def modify_doc(doc):
            if False:
                i = 10
                return i + 15
            plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)

            def cb(attr, old, new):
                if False:
                    for i in range(10):
                        print('nop')
                slider.title = 'baz'
            slider.on_change('value', cb)
            doc.add_root(column(slider, plot))
        page = bokeh_server_page(modify_doc)
        drag_slider(page.driver, slider, 150)
        sleep(1)
        assert get_slider_title_text(page.driver, slider) > '04 Aug 2017'