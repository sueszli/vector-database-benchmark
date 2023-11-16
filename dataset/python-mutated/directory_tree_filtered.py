from pathlib import Path
from typing import Iterable
from textual.app import App, ComposeResult
from textual.widgets import DirectoryTree

class FilteredDirectoryTree(DirectoryTree):

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        if False:
            i = 10
            return i + 15
        return [path for path in paths if not path.name.startswith('.')]

class DirectoryTreeApp(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield FilteredDirectoryTree('./')
if __name__ == '__main__':
    app = DirectoryTreeApp()
    app.run()