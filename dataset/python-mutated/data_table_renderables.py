from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable
ROWS = [('lane', 'swimmer', 'country', 'time'), (4, 'Joseph Schooling', 'Singapore', 50.39), (2, 'Michael Phelps', 'United States', 51.14), (5, 'Chad le Clos', 'South Africa', 51.14), (6, 'László Cseh', 'Hungary', 51.14), (3, 'Li Zhuhao', 'China', 51.26), (8, 'Mehdy Metella', 'France', 51.58), (7, 'Tom Shields', 'United States', 51.73), (1, 'Aleksandr Sadovnikov', 'Russia', 51.84), (10, 'Darren Burns', 'Scotland', 51.84)]

class TableApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield DataTable()

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        table = self.query_one(DataTable)
        table.add_columns(*ROWS[0])
        for row in ROWS[1:]:
            styled_row = [Text(str(cell), style='italic #03AC13', justify='right') for cell in row]
            table.add_row(*styled_row)
app = TableApp()
if __name__ == '__main__':
    app.run()