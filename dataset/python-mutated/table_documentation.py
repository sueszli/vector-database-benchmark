from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')
    columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'}, {'name': 'age', 'label': 'Age', 'field': 'age', 'sortable': True}]
    rows = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 21}, {'name': 'Carol'}]
    ui.table(columns=columns, rows=rows, row_key='name')

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Table with expandable rows', '\n        Scoped slots can be used to insert buttons that toggle the expand state of a table row.\n        See the [Quasar documentation](https://quasar.dev/vue-components/table#expanding-rows) for more information.\n    ')
    def table_with_expandable_rows():
        if False:
            while True:
                i = 10
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name'}, {'name': 'age', 'label': 'Age', 'field': 'age'}]
        rows = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 21}, {'name': 'Carol'}]
        table = ui.table(columns=columns, rows=rows, row_key='name').classes('w-72')
        table.add_slot('header', '\n            <q-tr :props="props">\n                <q-th auto-width />\n                <q-th v-for="col in props.cols" :key="col.name" :props="props">\n                    {{ col.label }}\n                </q-th>\n            </q-tr>\n        ')
        table.add_slot('body', '\n            <q-tr :props="props">\n                <q-td auto-width>\n                    <q-btn size="sm" color="accent" round dense\n                        @click="props.expand = !props.expand"\n                        :icon="props.expand ? \'remove\' : \'add\'" />\n                </q-td>\n                <q-td v-for="col in props.cols" :key="col.name" :props="props">\n                    {{ col.value }}\n                </q-td>\n            </q-tr>\n            <q-tr v-show="props.expand" :props="props">\n                <q-td colspan="100%">\n                    <div class="text-left">This is {{ props.row.name }}.</div>\n                </q-td>\n            </q-tr>\n        ')

    @text_demo('Show and hide columns', '\n        Here is an example of how to show and hide columns in a table.\n    ')
    def show_and_hide_columns():
        if False:
            i = 10
            return i + 15
        from typing import Dict
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'}, {'name': 'age', 'label': 'Age', 'field': 'age', 'sortable': True}]
        rows = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 21}, {'name': 'Carol'}]
        table = ui.table(columns=columns, rows=rows, row_key='name')

        def toggle(column: Dict, visible: bool) -> None:
            if False:
                i = 10
                return i + 15
            column['classes'] = '' if visible else 'hidden'
            column['headerClasses'] = '' if visible else 'hidden'
            table.update()
        with ui.button(icon='menu'):
            with ui.menu(), ui.column().classes('gap-0 p-2'):
                for column in columns:
                    ui.switch(column['label'], value=True, on_change=lambda e, column=column: toggle(column, e.value))

    @text_demo('Table with drop down selection', '\n        Here is an example of how to use a drop down selection in a table.\n        After emitting a `rename` event from the scoped slot, the `rename` function updates the table rows.\n    ')
    def table_with_drop_down_selection():
        if False:
            return 10
        from nicegui import events
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name'}, {'name': 'age', 'label': 'Age', 'field': 'age'}]
        rows = [{'id': 0, 'name': 'Alice', 'age': 18}, {'id': 1, 'name': 'Bob', 'age': 21}, {'id': 2, 'name': 'Carol'}]
        name_options = ['Alice', 'Bob', 'Carol']

        def rename(e: events.GenericEventArguments) -> None:
            if False:
                i = 10
                return i + 15
            for row in rows:
                if row['id'] == e.args['id']:
                    row['name'] = e.args['name']
            ui.notify(f'Table.rows is now: {table.rows}')
        table = ui.table(columns=columns, rows=rows, row_key='name').classes('w-full')
        table.add_slot('body', '\n            <q-tr :props="props">\n                <q-td key="name" :props="props">\n                    <q-select\n                        v-model="props.row.name"\n                        :options="' + str(name_options) + '"\n                        @update:model-value="() => $parent.$emit(\'rename\', props.row)"\n                    />\n                </q-td>\n                <q-td key="age" :props="props">\n                    {{ props.row.age }}\n                </q-td>\n            </q-tr>\n        ')
        table.on('rename', rename)

    @text_demo('Table from Pandas DataFrame', '\n        You can create a table from a Pandas DataFrame using the `from_pandas` method. \n        This method takes a Pandas DataFrame as input and returns a table.\n    ')
    def table_from_pandas_demo():
        if False:
            print('Hello World!')
        import pandas as pd
        df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        ui.table.from_pandas(df).classes('max-h-40')

    @text_demo('Adding rows', "\n        It's simple to add new rows with the `add_rows(dict)` method.\n    ")
    def adding_rows():
        if False:
            print('Hello World!')
        import os
        import random

        def add():
            if False:
                return 10
            item = os.urandom(10 // 2).hex()
            table.add_rows({'id': item, 'count': random.randint(0, 100)})
        ui.button('add', on_click=add)
        columns = [{'name': 'id', 'label': 'ID', 'field': 'id'}, {'name': 'count', 'label': 'Count', 'field': 'count'}]
        table = ui.table(columns=columns, rows=[], row_key='id').classes('w-full')

    @text_demo('Custom sorting and formatting', '\n        You can define dynamic column attributes using a `:` prefix.\n        This way you can define custom sorting and formatting functions.\n\n        The following example allows sorting the `name` column by length.\n        The `age` column is formatted to show the age in years.\n    ')
    def custom_formatting():
        if False:
            return 10
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'sortable': True, ':sort': '(a, b, rowA, rowB) => b.length - a.length'}, {'name': 'age', 'label': 'Age', 'field': 'age', ':format': 'value => value + " years"'}]
        rows = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 21}, {'name': 'Carl', 'age': 42}]
        ui.table(columns=columns, rows=rows, row_key='name')

    @text_demo('Toggle fullscreen', '\n        You can toggle the fullscreen mode of a table using the `toggle_fullscreen()` method.\n    ')
    def toggle_fullscreen():
        if False:
            while True:
                i = 10
        table = ui.table(columns=[{'name': 'name', 'label': 'Name', 'field': 'name'}], rows=[{'name': 'Alice'}, {'name': 'Bob'}, {'name': 'Carol'}]).classes('w-full')
        with table.add_slot('top-left'):

            def toggle() -> None:
                if False:
                    for i in range(10):
                        print('nop')
                table.toggle_fullscreen()
                button.props('icon=fullscreen_exit' if table.is_fullscreen else 'icon=fullscreen')
            button = ui.button('Toggle fullscreen', icon='fullscreen', on_click=toggle).props('flat')

    @text_demo('Pagination', '\n        You can provide either a single integer or a dictionary to define pagination.\n\n        The dictionary can contain the following keys:\n\n        - `rowsPerPage`: The number of rows per page.\n        - `sortBy`: The column name to sort by.\n        - `descending`: Whether to sort in descending order.\n        - `page`: The current page (1-based).\n    ')
    def pagination() -> None:
        if False:
            i = 10
            return i + 15
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'}, {'name': 'age', 'label': 'Age', 'field': 'age', 'sortable': True}]
        rows = [{'name': 'Elsa', 'age': 18}, {'name': 'Oaken', 'age': 46}, {'name': 'Hans', 'age': 20}, {'name': 'Sven'}, {'name': 'Olaf', 'age': 4}, {'name': 'Anna', 'age': 17}]
        ui.table(columns=columns, rows=rows, pagination=3)
        ui.table(columns=columns, rows=rows, pagination={'rowsPerPage': 4, 'sortBy': 'age', 'page': 2})

    @text_demo('Computed fields', '\n        You can use functions to compute the value of a column.\n        The function receives the row as an argument.\n        See the [Quasar documentation](https://quasar.dev/vue-components/table#defining-the-columns) for more information.\n    ')
    def computed_fields():
        if False:
            i = 10
            return i + 15
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'}, {'name': 'length', 'label': 'Length', ':field': 'row => row.name.length'}]
        rows = [{'name': 'Alice'}, {'name': 'Bob'}, {'name': 'Christopher'}]
        ui.table(columns=columns, rows=rows, row_key='name')

    @text_demo('Conditional formatting', '\n        You can use scoped slots to conditionally format the content of a cell.\n        See the [Quasar documentation](https://quasar.dev/vue-components/table#example--body-cell-slot)\n        for more information about body-cell slots.\n        \n        In this demo we use a `q-badge` to display the age in red if the person is under 21 years old.\n        We use the `body-cell-age` slot to insert the `q-badge` into the `age` column.\n        The ":color" attribute of the `q-badge` is set to "red" if the age is under 21, otherwise it is set to "green".\n        The colon in front of the "color" attribute indicates that the value is a JavaScript expression.\n    ')
    def conditional_formatting():
        if False:
            print('Hello World!')
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name'}, {'name': 'age', 'label': 'Age', 'field': 'age'}]
        rows = [{'name': 'Alice', 'age': 18}, {'name': 'Bob', 'age': 21}, {'name': 'Carol', 'age': 42}]
        table = ui.table(columns=columns, rows=rows, row_key='name')
        table.add_slot('body-cell-age', '\n            <q-td key="age" :props="props">\n                <q-badge :color="props.value < 21 ? \'red\' : \'green\'">\n                    {{ props.value }}\n                </q-badge>\n            </q-td>\n        ')

    @text_demo('Table cells with links', '\n        Here is a demo of how to insert links into table cells.\n        We use the `body-cell-link` slot to insert an `<a>` tag into the `link` column.\n    ')
    def table_cells_with_links():
        if False:
            return 10
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'}, {'name': 'link', 'label': 'Link', 'field': 'link', 'align': 'left'}]
        rows = [{'name': 'Google', 'link': 'https://google.com'}, {'name': 'Facebook', 'link': 'https://facebook.com'}, {'name': 'Twitter', 'link': 'https://twitter.com'}]
        table = ui.table(columns=columns, rows=rows, row_key='name')
        table.add_slot('body-cell-link', '\n            <q-td :props="props">\n                <a :href="props.value">{{ props.value }}</a>\n            </q-td>\n        ')