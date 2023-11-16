from textual.app import App
from textual.widgets import Button, Welcome

class WelcomeApp(App):

    def on_key(self) -> None:
        if False:
            while True:
                i = 10
        self.mount(Welcome())
        self.query_one(Button).label = 'YES!'
if __name__ == '__main__':
    app = WelcomeApp()
    app.run()