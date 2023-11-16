import csv
import io
from textual.app import App, ComposeResult
from textual.widgets import DataTable
CSV = 'lane,swimmer,country,time\n4,Joseph Schooling,Singapore,50.39\n2,Michael Phelps,United States,51.14\n5,Chad le Clos,South Africa,51.14\n6,László Cseh,Hungary,51.14\n3,Li Zhuhao,China,51.26\n8,Mehdy Metella,France,51.58\n7,Tom Shields,United States,51.73\n1,Aleksandr Sadovnikov,Russia,51.84'

class TableApp(App):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        table = DataTable()
        table.focus()
        table.cursor_type = 'column'
        table.fixed_columns = 1
        table.fixed_rows = 1
        yield table

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        table = self.query_one(DataTable)
        rows = csv.reader(io.StringIO(CSV))
        table.add_columns(*next(rows))
        table.add_rows(rows)
if __name__ == '__main__':
    app = TableApp()
    app.run()