from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Label
SCREEN_CSS_PATH = (Path(__file__) / '../hot_reloading_app_with_screen_css.tcss').resolve()
SCREEN_CSS_PATH.write_text('\nContainer {\n    align: center middle;\n}\n\nLabel {\n    border: round $primary;\n    padding: 3;\n}\n')

class MyScreen(Screen[None]):
    CSS_PATH = SCREEN_CSS_PATH

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Container(Label('Hello, world!'))

class HotReloadingApp(App[None]):

    def on_mount(self) -> None:
        if False:
            return 10
        self.push_screen(MyScreen())
if __name__ == '__main__':
    HotReloadingApp(watch_css=True).run()