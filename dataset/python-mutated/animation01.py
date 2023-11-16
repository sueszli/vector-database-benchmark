from textual.app import App, ComposeResult
from textual.widgets import Static

class AnimationApp(App):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        self.box = Static('Hello, World!')
        self.box.styles.background = 'red'
        self.box.styles.color = 'black'
        self.box.styles.padding = (1, 2)
        yield self.box

    def on_mount(self):
        if False:
            return 10
        self.box.styles.animate('opacity', value=0.0, duration=2.0)
if __name__ == '__main__':
    app = AnimationApp()
    app.run()