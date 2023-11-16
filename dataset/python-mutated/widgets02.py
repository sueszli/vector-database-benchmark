from textual.app import App
from textual.widgets import Welcome

class WelcomeApp(App):

    def on_key(self) -> None:
        if False:
            while True:
                i = 10
        self.mount(Welcome())

    def on_button_pressed(self) -> None:
        if False:
            print('Hello World!')
        self.exit()
if __name__ == '__main__':
    app = WelcomeApp()
    app.run()