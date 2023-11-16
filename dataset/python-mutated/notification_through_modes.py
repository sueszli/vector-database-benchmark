from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Label

class Mode(Screen):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Label('This is a mode screen')

class NotifyThroughModesApp(App[None]):
    MODES = {'test': Mode()}

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Label('Base screen')

    def on_mount(self):
        if False:
            i = 10
            return i + 15
        for n in range(10):
            self.notify(str(n))
        self.switch_mode('test')
if __name__ == '__main__':
    NotifyThroughModesApp().run()