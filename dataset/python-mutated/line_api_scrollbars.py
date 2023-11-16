from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import RichLog

class MyWidget(Widget):

    def render(self):
        if False:
            for i in range(10):
                print('nop')
        return Text('\n'.join((f'{n} 0123456789' for n in range(20))), no_wrap=True, overflow='hidden', justify='left')

class ScrollViewApp(App):
    CSS = '\n    Screen {\n        align: center middle;\n    }\n\n    RichLog {\n        width:13;\n        height:10;\n    }\n\n    VerticalScroll {\n        width:13;\n        height: 10;\n        overflow: scroll;\n        overflow-x: auto;\n    }\n\n    MyWidget {\n        width:13;\n        height:auto;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield RichLog()
        yield VerticalScroll(MyWidget())

    def on_ready(self) -> None:
        if False:
            i = 10
            return i + 15
        self.query_one(RichLog).write('\n'.join((f'{n} 0123456789' for n in range(20))))
        self.query_one(VerticalScroll).scroll_end(animate=False)
if __name__ == '__main__':
    app = ScrollViewApp()
    app.run()