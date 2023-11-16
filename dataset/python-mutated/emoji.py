import sys
from typing import TYPE_CHECKING, Optional, Union
from .jupyter import JupyterMixin
from .segment import Segment
from .style import Style
from ._emoji_codes import EMOJI
from ._emoji_replace import _emoji_replace
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderResult
EmojiVariant = Literal['emoji', 'text']

class NoEmoji(Exception):
    """No emoji by that name."""

class Emoji(JupyterMixin):
    __slots__ = ['name', 'style', '_char', 'variant']
    VARIANTS = {'text': '︎', 'emoji': '️'}

    def __init__(self, name: str, style: Union[str, Style]='none', variant: Optional[EmojiVariant]=None) -> None:
        if False:
            while True:
                i = 10
        "A single emoji character.\n\n        Args:\n            name (str): Name of emoji.\n            style (Union[str, Style], optional): Optional style. Defaults to None.\n\n        Raises:\n            NoEmoji: If the emoji doesn't exist.\n        "
        self.name = name
        self.style = style
        self.variant = variant
        try:
            self._char = EMOJI[name]
        except KeyError:
            raise NoEmoji(f'No emoji called {name!r}')
        if variant is not None:
            self._char += self.VARIANTS.get(variant, '')

    @classmethod
    def replace(cls, text: str) -> str:
        if False:
            return 10
        'Replace emoji markup with corresponding unicode characters.\n\n        Args:\n            text (str): A string with emojis codes, e.g. "Hello :smiley:!"\n\n        Returns:\n            str: A string with emoji codes replaces with actual emoji.\n        '
        return _emoji_replace(text)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<emoji {self.name!r}>'

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return self._char

    def __rich_console__(self, console: 'Console', options: 'ConsoleOptions') -> 'RenderResult':
        if False:
            return 10
        yield Segment(self._char, console.get_style(self.style))
if __name__ == '__main__':
    import sys
    from rich.columns import Columns
    from rich.console import Console
    console = Console(record=True)
    columns = Columns((f':{name}: {name}' for name in sorted(EMOJI.keys()) if '\u200d' not in name), column_first=True)
    console.print(columns)
    if len(sys.argv) > 1:
        console.save_html(sys.argv[1])