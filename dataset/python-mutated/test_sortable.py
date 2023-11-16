from __future__ import annotations
import pytest
pytest
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from tests.support.plugins.project import BokehModelPage
from tests.support.util.selenium import get_table_cell, get_table_header
pytest_plugins = ('tests.support.plugins.project',)

@pytest.mark.selenium
class Test_DataTableSortable:

    def test_columns_sortable(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            return 10
        data = {'x': [1, 2, 3, 4], 'y': [4, 3, 2, 1], 'd': ['foo', 'bar', 'baz', 'quux']}
        source = ColumnDataSource(data)
        table = DataTable(columns=[TableColumn(field='x', title='x'), TableColumn(field='y', title='y', sortable=False), TableColumn(field='d', title='d', sortable=True)], source=source)
        page = bokeh_model_page(table)
        h1 = get_table_header(page.driver, table, 1)
        assert 'slick-header-sortable' in h1.get_attribute('class')
        h2 = get_table_header(page.driver, table, 2)
        assert 'slick-header-sortable' in h2.get_attribute('class')
        h3 = get_table_header(page.driver, table, 3)
        assert 'slick-header-sortable' not in h3.get_attribute('class')
        h4 = get_table_header(page.driver, table, 4)
        assert 'slick-header-sortable' in h4.get_attribute('class')
        assert page.has_no_console_errors()

    def test_click_nonsortable(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        data = {'x': [1, 2, 3, 4], 'y': [4, 3, 2, 1], 'd': ['foo', 'bar', 'baz', 'quux']}
        source = ColumnDataSource(data)
        table = DataTable(columns=[TableColumn(field='x', title='x'), TableColumn(field='y', title='y', sortable=False), TableColumn(field='d', title='d', sortable=True)], source=source)
        page = bokeh_model_page(table)
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        h3 = get_table_header(page.driver, table, 3)
        h3.click()
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        assert page.has_no_console_errors()

    def test_click_sortable(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        data = {'x': [1, 2, 3, 4], 'y': [4, 3, 2, 1], 'd': ['foo', 'bar', 'baz', 'quux']}
        source = ColumnDataSource(data)
        table = DataTable(columns=[TableColumn(field='x', title='x'), TableColumn(field='y', title='y', sortable=False), TableColumn(field='d', title='d', sortable=True)], source=source)
        page = bokeh_model_page(table)
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        h4 = get_table_header(page.driver, table, 4)
        h4.click()
        for (i, x) in enumerate(['bar', 'baz', 'foo', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        h4 = get_table_header(page.driver, table, 4)
        h4.click()
        for (i, x) in enumerate(['quux', 'foo', 'baz', 'bar'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        assert page.has_no_console_errors()

    def test_table_unsortable(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            while True:
                i = 10
        data = {'x': [1, 2, 3, 4], 'y': [4, 3, 2, 1], 'd': ['foo', 'bar', 'baz', 'quux']}
        source = ColumnDataSource(data)
        table = DataTable(columns=[TableColumn(field='x', title='x'), TableColumn(field='y', title='y', sortable=False), TableColumn(field='d', title='d', sortable=True)], sortable=False, source=source)
        page = bokeh_model_page(table)
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        h4 = get_table_header(page.driver, table, 4)
        h4.click()
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        h4 = get_table_header(page.driver, table, 4)
        h4.click()
        for (i, x) in enumerate(['foo', 'bar', 'baz', 'quux'], 1):
            elt = get_table_cell(page.driver, table, i, 3)
            assert elt.text == x
        assert page.has_no_console_errors()