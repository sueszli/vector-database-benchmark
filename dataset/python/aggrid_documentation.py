from nicegui import ui

from ..documentation_tools import text_demo


def main_demo() -> None:
    grid = ui.aggrid({
        'defaultColDef': {'flex': 1},
        'columnDefs': [
            {'headerName': 'Name', 'field': 'name'},
            {'headerName': 'Age', 'field': 'age'},
            {'headerName': 'Parent', 'field': 'parent', 'hide': True},
        ],
        'rowData': [
            {'name': 'Alice', 'age': 18, 'parent': 'David'},
            {'name': 'Bob', 'age': 21, 'parent': 'Eve'},
            {'name': 'Carol', 'age': 42, 'parent': 'Frank'},
        ],
        'rowSelection': 'multiple',
    }).classes('max-h-40')

    def update():
        grid.options['rowData'][0]['age'] += 1
        grid.update()

    ui.button('Update', on_click=update)
    ui.button('Select all', on_click=lambda: grid.call_api_method('selectAll'))
    ui.button('Show parent', on_click=lambda: grid.call_column_api_method('setColumnVisible', 'parent', True))


def more() -> None:
    @text_demo('Select AG Grid Rows', '''
        You can add checkboxes to grid cells to allow the user to select single or multiple rows.

        To retrieve the currently selected rows, use the `get_selected_rows` method.
        This method returns a list of rows as dictionaries.

        If `rowSelection` is set to `'single'` or to get the first selected row,
        you can also use the `get_selected_row` method.
        This method returns a single row as a dictionary or `None` if no row is selected.

        See the [AG Grid documentation](https://www.ag-grid.com/javascript-data-grid/row-selection/#example-single-row-selection) for more information.
    ''')
    def aggrid_with_selectable_rows():
        grid = ui.aggrid({
            'columnDefs': [
                {'headerName': 'Name', 'field': 'name', 'checkboxSelection': True},
                {'headerName': 'Age', 'field': 'age'},
            ],
            'rowData': [
                {'name': 'Alice', 'age': 18},
                {'name': 'Bob', 'age': 21},
                {'name': 'Carol', 'age': 42},
            ],
            'rowSelection': 'multiple',
        }).classes('max-h-40')

        async def output_selected_rows():
            rows = await grid.get_selected_rows()
            if rows:
                for row in rows:
                    ui.notify(f"{row['name']}, {row['age']}")
            else:
                ui.notify('No rows selected.')

        async def output_selected_row():
            row = await grid.get_selected_row()
            if row:
                ui.notify(f"{row['name']}, {row['age']}")
            else:
                ui.notify('No row selected!')

        ui.button('Output selected rows', on_click=output_selected_rows)
        ui.button('Output selected row', on_click=output_selected_row)

    @text_demo('Filter Rows using Mini Filters', '''
        You can add [mini filters](https://ag-grid.com/javascript-data-grid/filter-set-mini-filter/)
        to the header of each column to filter the rows.
        
        Note how the "agTextColumnFilter" matches individual characters, like "a" in "Alice" and "Carol",
        while the "agNumberColumnFilter" matches the entire number, like "18" and "21", but not "1".
    ''')
    def aggrid_with_minifilters():
        ui.aggrid({
            'columnDefs': [
                {'headerName': 'Name', 'field': 'name', 'filter': 'agTextColumnFilter', 'floatingFilter': True},
                {'headerName': 'Age', 'field': 'age', 'filter': 'agNumberColumnFilter', 'floatingFilter': True},
            ],
            'rowData': [
                {'name': 'Alice', 'age': 18},
                {'name': 'Bob', 'age': 21},
                {'name': 'Carol', 'age': 42},
            ],
        }).classes('max-h-40')

    @text_demo('AG Grid with Conditional Cell Formatting', '''
        This demo shows how to use [cellClassRules](https://www.ag-grid.com/javascript-grid-cell-styles/#cell-class-rules)
        to conditionally format cells based on their values.
    ''')
    def aggrid_with_conditional_cell_formatting():
        ui.aggrid({
            'columnDefs': [
                {'headerName': 'Name', 'field': 'name'},
                {'headerName': 'Age', 'field': 'age', 'cellClassRules': {
                    'bg-red-300': 'x < 21',
                    'bg-green-300': 'x >= 21',
                }},
            ],
            'rowData': [
                {'name': 'Alice', 'age': 18},
                {'name': 'Bob', 'age': 21},
                {'name': 'Carol', 'age': 42},
            ],
        })

    @text_demo('Create Grid from Pandas DataFrame', '''
        You can create an AG Grid from a Pandas DataFrame using the `from_pandas` method.
        This method takes a Pandas DataFrame as input and returns an AG Grid.
    ''')
    def aggrid_from_pandas():
        import pandas as pd

        df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        ui.aggrid.from_pandas(df).classes('max-h-40')

    @text_demo('Render columns as HTML', '''
        You can render columns as HTML by passing a list of column indices to the `html_columns` argument.
    ''')
    def aggrid_with_html_columns():
        ui.aggrid({
            'columnDefs': [
                {'headerName': 'Name', 'field': 'name'},
                {'headerName': 'URL', 'field': 'url'},
            ],
            'rowData': [
                {'name': 'Google', 'url': '<a href="https://google.com">https://google.com</a>'},
                {'name': 'Facebook', 'url': '<a href="https://facebook.com">https://facebook.com</a>'},
            ],
        }, html_columns=[1])

    @text_demo('Respond to an AG Grid event', '''
        All AG Grid events are passed through to NiceGUI via the AG Grid global listener.
        These events can be subscribed to using the `.on()` method.
    ''')
    def aggrid_respond_to_event():
        ui.aggrid({
            'columnDefs': [
                {'headerName': 'Name', 'field': 'name'},
                {'headerName': 'Age', 'field': 'age'},
            ],
            'rowData': [
                {'name': 'Alice', 'age': 18},
                {'name': 'Bob', 'age': 21},
                {'name': 'Carol', 'age': 42},
            ],
        }).on('cellClicked', lambda event: ui.notify(f'Cell value: {event.args["value"]}'))

    @text_demo('AG Grid with complex objects', '''
        You can use nested complex objects in AG Grid by separating the field names with a period.
        (This is the reason why keys in `rowData` are not allowed to contain periods.)
    ''')
    def aggrid_with_complex_objects():
        ui.aggrid({
            'columnDefs': [
                {'headerName': 'First name', 'field': 'name.first'},
                {'headerName': 'Last name', 'field': 'name.last'},
                {'headerName': 'Age', 'field': 'age'}
            ],
            'rowData': [
                {'name': {'first': 'Alice', 'last': 'Adams'}, 'age': 18},
                {'name': {'first': 'Bob', 'last': 'Brown'}, 'age': 21},
                {'name': {'first': 'Carol', 'last': 'Clark'}, 'age': 42},
            ],
        }).classes('max-h-40')

    @text_demo('AG Grid with dynamic row height', '''
        You can set the height of individual rows by passing a function to the `getRowHeight` argument.
    ''')
    def aggrid_with_dynamic_row_height():
        ui.aggrid({
            'columnDefs': [{'field': 'name'}, {'field': 'age'}],
            'rowData': [
                {'name': 'Alice', 'age': '18'},
                {'name': 'Bob', 'age': '21'},
                {'name': 'Carol', 'age': '42'},
            ],
            ':getRowHeight': 'params => params.data.age > 35 ? 50 : 25',
        }).classes('max-h-40')
