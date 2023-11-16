from textual.app import App
from textual import events

class ActionsApp(App):

    def action_set_background(self, color: str) -> None:
        if False:
            return 10
        self.screen.styles.background = color

    def on_key(self, event: events.Key) -> None:
        if False:
            print('Hello World!')
        if event.key == 'r':
            self.action_set_background('red')
if __name__ == '__main__':
    app = ActionsApp()
    app.run()