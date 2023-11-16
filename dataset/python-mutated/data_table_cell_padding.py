from textual.app import App, ComposeResult
from textual.widgets import DataTable

class TableApp(App):
    CSS = '\n    DataTable {\n        margin: 1;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        for cell_padding in range(5):
            dt = DataTable(cell_padding=cell_padding)
            dt.add_columns('one', 'two', 'three')
            dt.add_row('value', 'value', 'val')
            yield dt

    def key_a(self):
        if False:
            print('Hello World!')
        self.query(DataTable).last().cell_padding = 20

    def key_b(self):
        if False:
            for i in range(10):
                print('nop')
        self.query(DataTable).last().cell_padding = 10
app = TableApp()
if __name__ == '__main__':
    app.run()