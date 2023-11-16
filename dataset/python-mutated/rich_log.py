import csv
import io
from rich.syntax import Syntax
from rich.table import Table
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog
CSV = 'lane,swimmer,country,time\n4,Joseph Schooling,Singapore,50.39\n2,Michael Phelps,United States,51.14\n5,Chad le Clos,South Africa,51.14\n6,László Cseh,Hungary,51.14\n3,Li Zhuhao,China,51.26\n8,Mehdy Metella,France,51.58\n7,Tom Shields,United States,51.73\n1,Aleksandr Sadovnikov,Russia,51.84'
CODE = 'def loop_first_last(values: Iterable[T]) -> Iterable[tuple[bool, bool, T]]:\n    """Iterate and generate a tuple with a flag for first and last value."""\n    iter_values = iter(values)\n    try:\n        previous_value = next(iter_values)\n    except StopIteration:\n        return\n    first = True\n    for value in iter_values:\n        yield first, False, previous_value\n        first = False\n        previous_value = value\n    yield first, True, previous_value'

class RichLogApp(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield RichLog(highlight=True, markup=True)

    def on_ready(self) -> None:
        if False:
            return 10
        'Called  when the DOM is ready.'
        text_log = self.query_one(RichLog)
        text_log.write(Syntax(CODE, 'python', indent_guides=True))
        rows = iter(csv.reader(io.StringIO(CSV)))
        table = Table(*next(rows))
        for row in rows:
            table.add_row(*row)
        text_log.write(table)
        text_log.write('[bold magenta]Write text or any Rich renderable!')

    def on_key(self, event: events.Key) -> None:
        if False:
            while True:
                i = 10
        'Write Key events to log.'
        text_log = self.query_one(RichLog)
        text_log.write(event)
if __name__ == '__main__':
    app = RichLogApp()
    app.run()