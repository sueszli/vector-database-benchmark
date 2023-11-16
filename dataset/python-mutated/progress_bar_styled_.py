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
            while True:
                i = 10
        with Center():
            with Middle():
                yield ProgressBar()
        yield Footer()

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        'Set up a timer to simulate progess happening.'
        self.progress_timer = self.set_interval(1 / 10, self.make_progress, pause=True)

    def make_progress(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called automatically to advance the progress bar.'
        self.query_one(ProgressBar).advance(1)

    def action_start(self) -> None:
        if False:
            return 10
        'Start the progress tracking.'
        self.query_one(ProgressBar).update(total=100)
        self.progress_timer.resume()

    def key_f(self) -> None:
        if False:
            print('Hello World!')
        self.query_one(ProgressBar).query_one('#bar')._get_elapsed_time = lambda : 5

    def key_t(self) -> None:
        if False:
            return 10
        self.query_one(ProgressBar).query_one('#eta')._get_elapsed_time = lambda : 3.9
        self.query_one(ProgressBar).update(total=100, progress=39)

    def key_u(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(ProgressBar).update(total=100, progress=100)
if __name__ == '__main__':
    StyledProgressBar().run()