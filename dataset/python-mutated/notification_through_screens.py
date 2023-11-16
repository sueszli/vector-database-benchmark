from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Label

class StackableScreen(Screen):
    TARGET_DEPTH = 10

    def __init__(self, count: int=TARGET_DEPTH) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._number = count

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Label(f'Screen {self.TARGET_DEPTH - self._number}')

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._number > 0:
            self.app.push_screen(StackableScreen(self._number - 1))

class NotifyDownScreensApp(App[None]):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Label('Base screen')

    def on_mount(self):
        if False:
            i = 10
            return i + 15
        for n in range(10):
            self.notify(str(n))
        self.push_screen(StackableScreen())
if __name__ == '__main__':
    NotifyDownScreensApp().run()