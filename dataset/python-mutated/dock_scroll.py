from textual.app import App
from textual.widgets import Header, Label, Footer

class TestApp(App):
    BINDINGS = [('ctrl+q', 'app.quit', 'Quit')]
    CSS = '\n    \n    Label {\n        border: solid red;\n    }\n    Footer {\n        height: 4;\n    }\n    '

    def compose(self):
        if False:
            i = 10
            return i + 15
        text = 'this is a sample sentence and here are some words'.replace(' ', '\n') * 2
        yield Header()
        yield Label(text)
        yield Footer()

    def on_mount(self):
        if False:
            i = 10
            return i + 15
        self.dark = False
if __name__ == '__main__':
    app = TestApp()
    app.run()