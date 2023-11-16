from textual.app import App, ComposeResult
from textual.containers import Center, Middle
from textual.timer import Timer
from textual.widgets import Footer, ProgressBar

class StyledProgressBar(App[None]):
    BINDINGS = [('s', 'start', 'Start')]
    CSS_PATH = 'progress_bar_styled.tcss'
    progress_timer: Timer
    'Timer to simulate progress happening.'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        with Center():
            with Middle():
                yield ProgressBar()
        yield Footer()

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        'Set up a timer to simulate progess happening.'
        self.progress_timer = self.set_interval(1 / 10, self.make_progress, pause=True)

    def make_progress(self) -> None:
        if False:
            i = 10
            return i + 15
        'Called automatically to advance the progress bar.'
        self.query_one(ProgressBar).advance(1)

    def action_start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start the progress tracking.'
        self.query_one(ProgressBar).update(total=100)
        self.progress_timer.resume()
if __name__ == '__main__':
    StyledProgressBar().run()