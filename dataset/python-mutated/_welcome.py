"""Provides a Textual welcome widget."""
from rich.markdown import Markdown
from ..app import ComposeResult
from ..containers import Container
from ._button import Button
from ._static import Static
WELCOME_MD = '# Welcome!\n\nTextual is a TUI, or *Text User Interface*, framework for Python inspired by modern web development. **We hope you enjoy using Textual!**\n\n## Dune quote\n\n> "I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain."\n'

class Welcome(Static):
    """A Textual welcome widget.

    This widget can be used as a form of placeholder within a Textual
    application; although also see
    [Placeholder][textual.widgets._placeholder.Placeholder].
    """
    DEFAULT_CSS = '\n        Welcome {\n            width: 100%;\n            height: 100%;\n            background: $surface;\n        }\n\n        Welcome Container {\n            padding: 1;\n            background: $panel;\n            color: $text;\n        }\n\n        Welcome #text {\n            margin:  0 1;\n        }\n\n        Welcome #close {\n            dock: bottom;\n            width: 100%;\n        }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Container(Static(Markdown(WELCOME_MD), id='text'), id='md')
        yield Button('OK', id='close', variant='success')