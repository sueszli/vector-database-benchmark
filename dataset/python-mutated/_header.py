"""Provides a Textual application header widget."""
from __future__ import annotations
from datetime import datetime
from rich.text import Text
from ..app import RenderResult
from ..events import Click, Mount
from ..reactive import Reactive
from ..widget import Widget

class HeaderIcon(Widget):
    """Display an 'icon' on the left of the header."""
    DEFAULT_CSS = '\n    HeaderIcon {\n        dock: left;\n        padding: 0 1;\n        width: 8;\n        content-align: left middle;\n    }\n\n    HeaderIcon:hover {\n        background: $foreground 10%;\n    }\n    '
    icon = Reactive('⭘')
    'The character to use as the icon within the header.'

    async def on_click(self, event: Click) -> None:
        """Launch the command palette when icon is clicked."""
        event.stop()
        await self.run_action('command_palette')

    def render(self) -> RenderResult:
        if False:
            i = 10
            return i + 15
        'Render the header icon.\n\n        Returns:\n            The rendered icon.\n        '
        return self.icon

class HeaderClockSpace(Widget):
    """The space taken up by the clock on the right of the header."""
    DEFAULT_CSS = '\n    HeaderClockSpace {\n        dock: right;\n        width: 10;\n        padding: 0 1;\n    }\n    '

    def render(self) -> RenderResult:
        if False:
            while True:
                i = 10
        'Render the header clock space.\n\n        Returns:\n            The rendered space.\n        '
        return ''

class HeaderClock(HeaderClockSpace):
    """Display a clock on the right of the header."""
    DEFAULT_CSS = '\n    HeaderClock {\n        background: $foreground-darken-1 5%;\n        color: $text;\n        text-opacity: 85%;\n        content-align: center middle;\n    }\n    '

    def _on_mount(self, _: Mount) -> None:
        if False:
            return 10
        self.set_interval(1, callback=self.refresh, name=f'update header clock')

    def render(self) -> RenderResult:
        if False:
            return 10
        'Render the header clock.\n\n        Returns:\n            The rendered clock.\n        '
        return Text(datetime.now().time().strftime('%X'))

class HeaderTitle(Widget):
    """Display the title / subtitle in the header."""
    DEFAULT_CSS = '\n    HeaderTitle {\n        content-align: center middle;\n        width: 100%;\n    }\n    '
    text: Reactive[str] = Reactive('')
    'The main title text.'
    sub_text = Reactive('')
    'The sub-title text.'

    def render(self) -> RenderResult:
        if False:
            for i in range(10):
                print('nop')
        'Render the title and sub-title.\n\n        Returns:\n            The value to render.\n        '
        text = Text(self.text, no_wrap=True, overflow='ellipsis')
        if self.sub_text:
            text.append(' — ')
            text.append(self.sub_text, 'dim')
        return text

class Header(Widget):
    """A header widget with icon and clock."""
    DEFAULT_CSS = '\n    Header {\n        dock: top;\n        width: 100%;\n        background: $foreground 5%;\n        color: $text;\n        height: 1;\n    }\n    Header.-tall {\n        height: 3;\n    }\n    '
    DEFAULT_CLASSES = ''
    tall: Reactive[bool] = Reactive(False)
    'Set to `True` for a taller header or `False` for a single line header.'

    def __init__(self, show_clock: bool=False, *, name: str | None=None, id: str | None=None, classes: str | None=None):
        if False:
            i = 10
            return i + 15
        'Initialise the header widget.\n\n        Args:\n            show_clock: ``True`` if the clock should be shown on the right of the header.\n            name: The name of the header widget.\n            id: The ID of the header widget in the DOM.\n            classes: The CSS classes of the header widget.\n        '
        super().__init__(name=name, id=id, classes=classes)
        self._show_clock = show_clock

    def compose(self):
        if False:
            print('Hello World!')
        yield HeaderIcon()
        yield HeaderTitle()
        yield (HeaderClock() if self._show_clock else HeaderClockSpace())

    def watch_tall(self, tall: bool) -> None:
        if False:
            print('Hello World!')
        self.set_class(tall, '-tall')

    def _on_click(self):
        if False:
            i = 10
            return i + 15
        self.toggle_class('-tall')

    @property
    def screen_title(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The title that this header will display.\n\n        This depends on [`Screen.title`][textual.screen.Screen.title] and [`App.title`][textual.app.App.title].\n        '
        screen_title = self.screen.title
        title = screen_title if screen_title is not None else self.app.title
        return title

    @property
    def screen_sub_title(self) -> str:
        if False:
            while True:
                i = 10
        'The sub-title that this header will display.\n\n        This depends on [`Screen.sub_title`][textual.screen.Screen.sub_title] and [`App.sub_title`][textual.app.App.sub_title].\n        '
        screen_sub_title = self.screen.sub_title
        sub_title = screen_sub_title if screen_sub_title is not None else self.app.sub_title
        return sub_title

    def _on_mount(self, _: Mount) -> None:
        if False:
            for i in range(10):
                print('nop')

        def set_title() -> None:
            if False:
                print('Hello World!')
            self.query_one(HeaderTitle).text = self.screen_title

        def set_sub_title(sub_title: str) -> None:
            if False:
                i = 10
                return i + 15
            self.query_one(HeaderTitle).sub_text = self.screen_sub_title
        self.watch(self.app, 'title', set_title)
        self.watch(self.app, 'sub_title', set_sub_title)
        self.watch(self.screen, 'title', set_title)
        self.watch(self.screen, 'sub_title', set_sub_title)