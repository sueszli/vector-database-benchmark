from textual.app import App, ComposeResult
from textual.widgets import Log
from textual.containers import Horizontal

class LogApp(App):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Log()

    def on_ready(self) -> None:
        if False:
            i = 10
            return i + 15
        log = self.query_one(Log)
        log.write('Hello,')
        log.write(' World')
        log.write("!\nWhat's up?")
        log.write('')
        log.write('\n')
        log.write('FOO')
if __name__ == '__main__':
    app = LogApp()
    app.run()