from textual.app import App
from textual import events

class EventApp(App):
    COLORS = ['white', 'maroon', 'red', 'purple', 'fuchsia', 'olive', 'yellow', 'navy', 'teal', 'aqua']

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        self.screen.styles.background = 'darkblue'

    def on_key(self, event: events.Key) -> None:
        if False:
            print('Hello World!')
        if event.key.isdecimal():
            self.screen.styles.background = self.COLORS[int(event.key)]
if __name__ == '__main__':
    app = EventApp()
    app.run()