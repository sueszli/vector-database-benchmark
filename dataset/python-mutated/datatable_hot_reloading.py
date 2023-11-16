from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import DataTable
CSS_PATH = (Path(__file__) / '../datatable_hot_reloading.tcss').resolve()
CSS_PATH.write_text('DataTable > .datatable--cursor {\n    background: purple;\n}\n\nDataTable > .datatable--fixed {\n    background: red;\n}\n\nDataTable > .datatable--fixed-cursor {\n    background: blue;\n}\n\nDataTable > .datatable--header {\n    background: yellow;\n}\n\nDataTable > .datatable--odd-row {\n    background: pink;\n}\n\nDataTable > .datatable--even-row {\n    background: brown;\n}\n')

class DataTableHotReloadingApp(App[None]):
    CSS_PATH = CSS_PATH

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield DataTable(zebra_stripes=True, cursor_type='row')

    def on_mount(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        dt = self.query_one(DataTable)
        dt.add_column('A', width=10)
        self.c = dt.add_column('B')
        dt.fixed_columns = 1
        dt.add_row('one', 'two')
        dt.add_row('three', 'four')
        dt.add_row('five', 'six')
if __name__ == '__main__':
    app = DataTableHotReloadingApp()
    app.run()