from __future__ import annotations
import pytest
pytest
from bokeh.layouts import column
from bokeh.models import Circle, ColumnDataSource, CustomJS, Plot, Range1d, Spinner
from tests.support.plugins.project import BokehModelPage, BokehServerPage
from tests.support.util.selenium import RECORD, ActionChains, Keys, enter_text_in_element, find_element_for
pytest_plugins = ('tests.support.plugins.project',)

def mk_modify_doc(spinner: Spinner):
    if False:
        i = 10
        return i + 15

    def modify_doc(doc):
        if False:
            while True:
                i = 10
        source = ColumnDataSource(dict(x=[1, 2], y=[1, 1], val=['a', 'b']))
        plot = Plot(height=400, width=400, x_range=Range1d(0, 1), y_range=Range1d(0, 1), min_border=0)
        plot.add_glyph(source, Circle(x='x', y='y'))
        plot.tags.append(CustomJS(name='custom-action', args=dict(s=source), code=RECORD('data', 's.data')))

        def cb(attr, old, new):
            if False:
                return 10
            source.data['val'] = [old, new]
        spinner.on_change('value', cb)
        doc.add_root(column(spinner, plot))
        return doc
    return modify_doc

@pytest.mark.selenium
class Test_Spinner:

    def test_spinner_display(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        spinner = Spinner()
        page = bokeh_model_page(spinner)
        input_el = find_element_for(page.driver, spinner, 'input')
        btn_up_el = find_element_for(page.driver, spinner, '.bk-spin-btn-up')
        btn_down_el = find_element_for(page.driver, spinner, '.bk-spin-btn-down')
        assert input_el.get_attribute('type') == 'text'
        assert btn_up_el.tag_name == 'button'
        assert btn_down_el.tag_name == 'button'
        assert page.has_no_console_errors()

    def test_spinner_display_title(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        spinner = Spinner(title='title')
        page = bokeh_model_page(spinner)
        label_el = find_element_for(page.driver, spinner, 'label')
        assert label_el.text == 'title'
        input_el = find_element_for(page.driver, spinner, 'input')
        assert input_el.get_attribute('type') == 'text'
        assert page.has_no_console_errors()

    def test_spinner_value_format(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        spinner = Spinner(value=1, low=0, high=10, step=1, format='0.00')
        page = bokeh_model_page(spinner)
        input_el = find_element_for(page.driver, spinner, 'input')
        assert input_el.get_attribute('value') == '1.00'
        assert page.has_no_console_errors()

    def test_spinner_smallest_step(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        spinner = Spinner(value=0, low=0, high=1, step=1e-16)
        spinner.js_on_change('value', CustomJS(code=RECORD('value', 'cb_obj.value')))
        page = bokeh_model_page(spinner)
        input_el = find_element_for(page.driver, spinner, 'input')
        enter_text_in_element(page.driver, input_el, '0.43654644333534')
        results = page.results
        assert results['value'] == 0.43654644333534
        enter_text_in_element(page.driver, input_el, '1e-16', click=2)
        results = page.results
        assert results['value'] == 1e-16
        assert page.has_no_console_errors()

    def test_spinner_spinning_events(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        spinner = Spinner(value=0, low=0, high=1, step=0.01)
        spinner.js_on_change('value', CustomJS(code=RECORD('value', 'cb_obj.value')))
        page = bokeh_model_page(spinner)
        input_el = find_element_for(page.driver, spinner, 'input')
        btn_up_el = find_element_for(page.driver, spinner, '.bk-spin-btn-up')
        btn_down_el = find_element_for(page.driver, spinner, '.bk-spin-btn-down')
        enter_text_in_element(page.driver, input_el, '0.5')
        results = page.results
        assert results['value'] == 0.5
        actions = ActionChains(page.driver)
        actions.click(on_element=btn_up_el)
        actions.perform()
        results = page.results
        assert results['value'] == 0.51
        actions = ActionChains(page.driver)
        actions.double_click(on_element=btn_down_el)
        actions.perform()
        results = page.results
        assert results['value'] == 0.49
        actions = ActionChains(page.driver)
        actions.click(on_element=input_el)
        actions.send_keys(Keys.ARROW_UP)
        actions.perform()
        results = page.results
        assert results['value'] == 0.5
        actions = ActionChains(page.driver)
        actions.click(on_element=input_el)
        actions.key_down(Keys.ARROW_DOWN)
        actions.perform()
        results = page.results
        assert results['value'] == 0.49
        actions = ActionChains(page.driver)
        actions.click(on_element=input_el)
        actions.key_down(Keys.PAGE_UP)
        actions.perform()
        results = page.results
        assert results['value'] == 0.59
        actions = ActionChains(page.driver)
        actions.click(on_element=input_el)
        actions.key_down(Keys.PAGE_DOWN)
        actions.perform()
        results = page.results
        assert results['value'] == 0.49
        assert page.has_no_console_errors()

    def test_server_on_change_round_trip(self, bokeh_server_page: BokehServerPage) -> None:
        if False:
            i = 10
            return i + 15
        spinner = Spinner(low=-1, high=10, step=0.1, value=4, format='0[.]0')
        page = bokeh_server_page(mk_modify_doc(spinner))
        input_el = find_element_for(page.driver, spinner, 'input')
        enter_text_in_element(page.driver, input_el, '4', click=2)
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == ['a', 'b']
        enter_text_in_element(page.driver, input_el, '5', click=2)
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == [4, 5]
        enter_text_in_element(page.driver, input_el, '11', click=2)
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == [5, 10]
        enter_text_in_element(page.driver, input_el, '-2', click=2)
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == [10, -1]
        input_el.clear()
        enter_text_in_element(page.driver, input_el, '5.1')
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == [None, 5.1]
        enter_text_in_element(page.driver, input_el, '5.19', click=2)
        page.eval_custom_action()
        results = page.results
        assert results['data']['val'] == [5.1, 5.19]
        assert input_el.get_attribute('value') == '5.2'