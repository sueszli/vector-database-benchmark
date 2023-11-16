""" Provide tools for executing Selenium tests.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Sequence
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
if TYPE_CHECKING:
    from selenium.webdriver.common.keys import _KeySeq
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement
from bokeh.models import Button
if TYPE_CHECKING:
    from bokeh.model import Model
    from bokeh.models.callbacks import Callback
    from bokeh.models.plots import Plot
    from bokeh.models.widgets import Slider
    from bokeh.models.widgets.tables import DataTable
__all__ = ('alt_click', 'ButtonWrapper', 'copy_table_rows', 'COUNT', 'drag_range_slider', 'drag_slider', 'element_to_finish_resizing', 'element_to_start_resizing', 'enter_text_in_cell', 'enter_text_in_cell_with_click_enter', 'enter_text_in_element', 'get_slider_bar_color', 'get_slider_title_text', 'get_slider_title_value', 'get_table_cell', 'get_table_column_cells', 'get_table_header', 'get_table_row', 'get_table_selected_rows', 'hover_element', 'INIT', 'paste_values', 'RECORD', 'RESULTS', 'SCROLL', 'select_element_and_press_key', 'shift_click', 'sort_table_column')
MATCHES_SCRIPT = '\n    function* descend(el, sel, parent) {\n        if (el.matches(sel)) {\n            yield parent ? el.parentElement : el\n        }\n        if (el.shadowRoot) {\n            for (const child of el.shadowRoot.children) {\n                yield* descend(child, sel, parent)\n            }\n        }\n        for (const child of el.children) {\n            yield* descend(child, sel, parent)\n        }\n    }\n\n    const selector = arguments[0]\n    const root = arguments[1] ?? document.documentElement\n    const parent = arguments[2] ?? false\n\n    return [...descend(root, selector, parent)]\n'

def find_matching_elements(driver: WebDriver, selector: str, *, root: WebElement | None=None, parent: bool=False) -> list[WebElement]:
    if False:
        i = 10
        return i + 15
    return driver.execute_script(MATCHES_SCRIPT, selector, root, parent)

def find_matching_element(driver: WebDriver, selector: str, *, root: WebElement | None=None, parent: bool=False) -> WebElement:
    if False:
        i = 10
        return i + 15
    elements = find_matching_elements(driver, selector, root=root, parent=parent)
    n = len(elements)
    if n == 0:
        raise ValueError('not found')
    else:
        return elements[0]
FIND_VIEW_SCRIPT = '\n    function* find(views, id, fn) {\n        for (const view of views) {\n            if (view.model.id == id) {\n                yield* fn(view)\n            } else if ("child_views" in view) {\n                yield* find(view.child_views, id, fn)\n            } else if ("tool_views" in view) {\n                yield* find(view.tool_views.values(), id, fn)\n            } else if ("renderer_views" in view) {\n                yield* find(view.renderer_views.values(), id, fn)\n            }\n        }\n    }\n\n    function head(iter) {\n        for (const item of iter) {\n            return item\n        }\n        return undefined\n    }\n'

def get_events_el(driver: WebDriver, model: Plot) -> WebElement:
    if False:
        return 10
    script = FIND_VIEW_SCRIPT + '\n    const id = arguments[0]\n    function* fn(view) {\n        yield view.canvas_view.events_el\n    }\n    return head(find(Bokeh.index, id, fn)) ?? null\n    '
    el = driver.execute_script(script, model.id)
    if el is not None:
        return el
    else:
        raise RuntimeError(f"can't resolve a view for {model}")
FIND_SCRIPT = '\n    const id = arguments[0]\n    const selector = arguments[1]\n\n    function* find(views) {\n        for (const view of views) {\n            if (view.model.id == id) {\n                if (selector != null) {\n                    const el = view.shadow_el ?? view.el\n                    yield [...el.querySelectorAll(selector)]\n                } else\n                    yield [view.el]\n            } else if ("child_views" in view) {\n                yield* find(view.child_views)\n            } else if ("tool_views" in view) {\n                yield* find(view.tool_views.values())\n            } else if ("renderer_views" in view) {\n                yield* find(view.renderer_views.values())\n            }\n        }\n    }\n'

def find_elements_for(driver: WebDriver, model: Model, selector: str | None=None) -> list[WebElement]:
    if False:
        while True:
            i = 10
    script = FIND_SCRIPT + '\n    for (const els of find(Bokeh.index)) {\n        return els\n    }\n    return null\n    '
    return driver.execute_script(script, model.id, selector)

def find_element_for(driver: WebDriver, model: Model, selector: str | None=None) -> WebElement:
    if False:
        for i in range(10):
            print('nop')
    script = FIND_SCRIPT + '\n    for (const els of find(Bokeh.index)) {\n        return els[0] ?? null\n    }\n    return null\n    '
    el = driver.execute_script(script, model.id, selector)
    if el is not None:
        return el
    else:
        raise ValueError('not found')

def COUNT(key: str) -> str:
    if False:
        i = 10
        return i + 15
    return 'Bokeh._testing.count(%r);' % key
INIT = 'Bokeh._testing.init();'

def RECORD(key: str, value: Any, *, final: bool=True) -> str:
    if False:
        while True:
            i = 10
    if final:
        return f'Bokeh._testing.record({key!r}, {value});'
    else:
        return f'Bokeh._testing.record0({key!r}, {value});'
RESULTS = 'return Bokeh._testing.results'

def SCROLL(amt: float) -> str:
    if False:
        print('Hello World!')
    return "\n    const elt = Bokeh.index.roots[0].canvas_view.events_el;\n    const event = new WheelEvent('wheel', { deltaY: %f, clientX: 100, clientY: 100} );\n    elt.dispatchEvent(event);\n    " % amt

def alt_click(driver: WebDriver, element: WebElement) -> None:
    if False:
        for i in range(10):
            print('nop')
    actions = ActionChains(driver)
    actions.key_down(Keys.META)
    actions.click(element)
    actions.key_up(Keys.META)
    actions.perform()

class ButtonWrapper:

    def __init__(self, label: str, callback: Callback) -> None:
        if False:
            while True:
                i = 10
        self.obj = Button(label=label)
        self.obj.js_on_event('button_click', callback)

    def click(self, driver: WebDriver) -> None:
        if False:
            i = 10
            return i + 15
        button = find_element_for(driver, self.obj, '.bk-btn')
        button.click()

class element_to_start_resizing:
    """ An expectation for checking if an element has started resizing
    """

    def __init__(self, element: WebElement) -> None:
        if False:
            return 10
        self.element = element
        self.previous_width = self.element.size['width']

    def __call__(self, driver: WebDriver) -> bool:
        if False:
            while True:
                i = 10
        current_width = self.element.size['width']
        if self.previous_width != current_width:
            return True
        else:
            self.previous_width = current_width
            return False

class element_to_finish_resizing:
    """ An expectation for checking if an element has finished resizing

    """

    def __init__(self, element: WebElement) -> None:
        if False:
            print('Hello World!')
        self.element = element
        self.previous_width = self.element.size['width']

    def __call__(self, driver: WebDriver) -> bool:
        if False:
            for i in range(10):
                print('nop')
        current_width = self.element.size['width']
        if self.previous_width == current_width:
            return True
        else:
            self.previous_width = current_width
            return False

def select_element_and_press_key(driver: WebDriver, element: WebElement, key: _KeySeq, press_number: int=1) -> None:
    if False:
        for i in range(10):
            print('nop')
    actions = ActionChains(driver)
    actions.move_to_element(element)
    actions.click()
    for _ in range(press_number):
        actions = ActionChains(driver)
        actions.send_keys_to_element(element, key)
        actions.perform()

def hover_element(driver: WebDriver, element: WebElement) -> None:
    if False:
        print('Hello World!')
    hover = ActionChains(driver).move_to_element(element)
    hover.perform()

def enter_text_in_element(driver: WebDriver, element: WebElement, text: str, click: int=1, enter: bool=True, mod: _KeySeq | None=None) -> None:
    if False:
        print('Hello World!')
    actions = ActionChains(driver)
    actions.move_to_element(element)
    if click == 1:
        actions.click()
    elif click == 2:
        actions.double_click()
    if enter:
        text += Keys.ENTER
    if mod:
        actions.key_down(mod)
    actions.send_keys(text)
    if mod:
        actions.key_up(mod)
    actions.perform()

def enter_text_in_cell(driver: WebDriver, table: DataTable, row: int, col: int, text: str) -> None:
    if False:
        print('Hello World!')
    actions = ActionChains(driver)
    cell = get_table_cell(driver, table, row, col)
    actions.move_to_element(cell)
    actions.double_click()
    actions.perform()
    actions = ActionChains(driver)
    cell = get_table_cell(driver, table, row, col)
    try:
        input = find_matching_element(driver, 'input', root=cell)
    except ValueError:
        return
    actions.move_to_element(input)
    actions.click()
    actions.double_click()
    actions.send_keys(text + Keys.ENTER)
    actions.perform()

def escape_cell(driver: WebDriver, table: DataTable, row: int, col: int) -> None:
    if False:
        i = 10
        return i + 15
    cell = get_table_cell(driver, table, row, col)
    try:
        input = find_matching_element(driver, 'input', root=cell)
    except ValueError:
        return
    actions = ActionChains(driver)
    actions.move_to_element(input)
    actions.send_keys(Keys.ESCAPE)
    actions.perform()

def enter_text_in_cell_with_click_enter(driver: WebDriver, table: DataTable, row: int, col: int, text: str) -> None:
    if False:
        while True:
            i = 10
    actions = ActionChains(driver)
    cell = get_table_cell(driver, table, row, col)
    actions.move_to_element(cell)
    actions.click()
    actions.send_keys(Keys.ENTER + text + Keys.ENTER)
    actions.perform()

def enter_text_with_click_enter(driver: WebDriver, cell: WebElement, text: str) -> None:
    if False:
        while True:
            i = 10
    actions = ActionChains(driver)
    actions.move_to_element(cell)
    actions.click()
    actions.send_keys(Keys.ENTER + text + Keys.ENTER)
    actions.perform()

def copy_table_rows(driver: WebDriver, table: DataTable, rows: Sequence[int]) -> None:
    if False:
        while True:
            i = 10
    actions = ActionChains(driver)
    row = get_table_row(driver, table, rows[0])
    actions.move_to_element(row)
    actions.click()
    actions.key_down(Keys.SHIFT)
    for r in rows[1:]:
        row = get_table_row(driver, table, r)
        actions.move_to_element(row)
        actions.click()
    actions.key_up(Keys.SHIFT)
    actions.key_down(Keys.CONTROL)
    actions.send_keys(Keys.INSERT)
    actions.key_up(Keys.CONTROL)
    actions.perform()

def paste_values(driver: WebDriver, el: WebElement | None=None) -> None:
    if False:
        print('Hello World!')
    actions = ActionChains(driver)
    if el:
        actions.move_to_element(el)
    actions.key_down(Keys.SHIFT)
    actions.send_keys(Keys.INSERT)
    actions.key_up(Keys.SHIFT)
    actions.perform()

def get_table_column_cells(driver: WebDriver, table: DataTable, col: int) -> list[str]:
    if False:
        while True:
            i = 10
    result = []
    rows = find_elements_for(driver, table, '.slick-row')
    for row in rows:
        elt = row.find_element(By.CSS_SELECTOR, '.slick-cell.l%d.r%d' % (col, col))
        result.append(elt.text)
    return result

def get_table_row(driver: WebDriver, table: DataTable, row: int) -> WebElement:
    if False:
        print('Hello World!')
    return find_element_for(driver, table, f'.slick-row:nth-child({row})')

def get_table_selected_rows(driver: WebDriver, table: DataTable) -> set[int]:
    if False:
        for i in range(10):
            print('nop')
    result = set()
    rows = find_elements_for(driver, table, '.slick-row')
    for (i, row) in enumerate(rows):
        elt = row.find_element(By.CSS_SELECTOR, '.slick-cell.l1.r1')
        if 'selected' in elt.get_attribute('class'):
            result.add(i)
    return result

def get_table_cell(driver: WebDriver, table: DataTable, row: int, col: int) -> WebElement:
    if False:
        return 10
    return find_element_for(driver, table, f'.slick-row:nth-child({row}) .r{col}')

def get_table_header(driver: WebDriver, table: DataTable, col: int) -> WebElement:
    if False:
        i = 10
        return i + 15
    return find_element_for(driver, table, f'.slick-header-columns .slick-header-column:nth-child({col})')

def sort_table_column(driver: WebDriver, table: DataTable, col: int, double: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    elt = find_element_for(driver, table, f'.slick-header-columns .slick-header-column:nth-child({col})')
    elt.click()
    if double:
        elt.click()

def shift_click(driver: WebDriver, element: WebElement) -> None:
    if False:
        while True:
            i = 10
    actions = ActionChains(driver)
    actions.key_down(Keys.SHIFT)
    actions.click(element)
    actions.key_up(Keys.SHIFT)
    actions.perform()

def drag_slider(driver: WebDriver, slider: Slider, distance: float, release: bool=True) -> None:
    if False:
        print('Hello World!')
    handle = find_element_for(driver, slider, '.noUi-handle')
    actions = ActionChains(driver)
    actions.move_to_element(handle)
    actions.click_and_hold()
    actions.move_by_offset(distance, 0)
    if release:
        actions.release()
    actions.perform()

def drag_range_slider(driver: WebDriver, slider: Slider, location: str, distance: float) -> None:
    if False:
        print('Hello World!')
    handle = find_element_for(driver, slider, f'.noUi-handle-{location}')
    actions = ActionChains(driver)
    actions.move_to_element(handle)
    actions.click_and_hold()
    actions.move_by_offset(distance, 0)
    actions.release()
    actions.perform()

def get_slider_title_text(driver: WebDriver, slider: Slider) -> str:
    if False:
        while True:
            i = 10
    return find_element_for(driver, slider, 'div.bk-input-group > div.bk-slider-title').text

def get_slider_title_value(driver: WebDriver, slider: Slider) -> str:
    if False:
        while True:
            i = 10
    return find_element_for(driver, slider, 'div.bk-input-group > div > span.bk-slider-value').text

def get_slider_bar_color(driver: WebDriver, slider: Slider) -> str:
    if False:
        for i in range(10):
            print('nop')
    bar_el = find_element_for(driver, slider, '.noUi-connect')
    return bar_el.value_of_css_property('background-color')