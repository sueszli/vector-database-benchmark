from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Label
CSS_PATH = (Path(__file__) / '../hot_reloading_app.tcss').resolve()
CSS_PATH.write_text('\nContainer {\n    align: center middle;\n}\n\nLabel {\n    border: round $primary;\n    padding: 3;\n}\n')

class HotReloadingApp(App[None]):
    CSS_PATH = CSS_PATH

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Container(Label('Hello, world!'))