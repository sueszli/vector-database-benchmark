from __future__ import annotations
import pytest
pytest
from time import sleep
from bokeh.models import ColumnDataSource, CustomJS, DataTable, IntEditor, NumberEditor, StringEditor, TableColumn
from tests.support.plugins.project import BokehModelPage
from tests.support.util.selenium import RECORD, enter_text_in_cell, enter_text_in_cell_with_click_enter, escape_cell, get_table_cell
pytest_plugins = ('tests.support.plugins.project',)

def make_table(editor, values):
    if False:
        while True:
            i = 10
    source = ColumnDataSource({'values': values})
    column = TableColumn(field='values', title='values', editor=editor())
    table = DataTable(source=source, columns=[column], editable=True, width=600)
    source.selected.js_on_change('indices', CustomJS(args=dict(s=source), code=RECORD('values', 's.data.values')))
    return table

@pytest.mark.selenium
class Test_IntEditor:
    values = [1, 2]
    editor = IntEditor

    def test_editing_does_not_update_source_on_noneditable_table(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        table.editable = False
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, '33')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        assert page.has_no_console_errors()

    @pytest.mark.parametrize('bad', ['1.1', 'text'])
    def test_editing_does_not_update_source_on_bad_values(self, bad: str, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, bad)
        escape_cell(page.driver, table, 1, 1)
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        assert page.has_no_console_errors()

    def test_editing_updates_source(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            i = 10
            return i + 15
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, '33')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        sleep(0.5)
        results = page.results
        assert results['values'] == [33, 2]
        assert page.has_no_console_errors()

@pytest.mark.selenium
class Test_NumberEditor:
    values = [1.1, 2.2]
    editor = NumberEditor

    def test_editing_does_not_update_source_on_noneditable_table(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        table.editable = False
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, '33.5')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        assert page.has_no_console_errors()

    @pytest.mark.parametrize('bad', ['text'])
    def test_editing_does_not_update_source_on_bad_values(self, bad, bokeh_model_page: BokehModelPage) -> None:
        if False:
            for i in range(10):
                print('nop')
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, bad)
        escape_cell(page.driver, table, 1, 1)
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        assert page.has_no_console_errors()

    def test_editing_updates_source(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            return 10
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, '33.5')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == [33.5, 2.2]
        assert page.has_no_console_errors()

@pytest.mark.selenium
class Test_StringEditor:
    values = ['foo', 'bar']
    editor = StringEditor

    def test_editing_does_not_update_source_on_noneditable_table(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        table.editable = False
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, 'baz')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        assert page.has_no_console_errors()

    def test_editing_updates_source(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell(page.driver, table, 1, 1, 'baz')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == ['baz', 'bar']
        assert page.has_no_console_errors()

    def test_editing_updates_source_with_click_enter(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        table = make_table(self.editor, self.values)
        page = bokeh_model_page(table)
        cell = get_table_cell(page.driver, table, 1, 1)
        cell.click()
        results = page.results
        assert results['values'] == self.values
        enter_text_in_cell_with_click_enter(page.driver, table, 1, 1, 'baz')
        cell = get_table_cell(page.driver, table, 2, 1)
        cell.click()
        results = page.results
        assert results['values'] == ['baz', 'bar']
        assert page.has_no_console_errors()