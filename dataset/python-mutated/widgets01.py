from textual.app import App, ComposeResult
from textual.widgets import Welcome

class WelcomeApp(App):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Welcome()

    def on_button_pressed(self) -> None:
        if False:
            i = 10
            return i + 15
        self.exit()
if __name__ == '__main__':
    app = WelcomeApp()
    app.run()