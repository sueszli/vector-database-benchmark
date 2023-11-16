from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Label, Static

class MyCustomWidget(Static):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Label(('|\n' * 100)[:-1])
        yield Label('SHOULD BE VISIBLE', id='target')

class MyApp(App):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        with VerticalScroll():
            yield MyCustomWidget()

    def key_t(self) -> None:
        if False:
            i = 10
            return i + 15
        self.query_one('#target').scroll_visible()
if __name__ == '__main__':
    MyApp().run()