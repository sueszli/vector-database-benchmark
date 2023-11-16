"""Provides the base code and implementations of toggle widgets.

In particular it provides `Checkbox`, `RadioButton` and `RadioSet`.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
from rich.style import Style
from rich.text import Text, TextType
from ..app import RenderResult
from ..binding import Binding, BindingType
from ..events import Click
from ..geometry import Size
from ..message import Message
from ..reactive import reactive
from ._static import Static
if TYPE_CHECKING:
    from typing_extensions import Self

class ToggleButton(Static, can_focus=True):
    """Base toggle button widget.

    Warning:
        `ToggleButton` should be considered to be an internal class; it
        exists to serve as the common core of [Checkbox][textual.widgets.Checkbox] and
        [RadioButton][textual.widgets.RadioButton].
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding('enter,space', 'toggle', 'Toggle', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter, space | Toggle the value. |\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'toggle--button', 'toggle--label'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `toggle--button` | Targets the toggle button itself. |\n    | `toggle--label` | Targets the text label of the toggle button. |\n    '
    DEFAULT_CSS = '\n    ToggleButton {\n        width: auto;\n        border: tall transparent;\n        padding: 0 1;\n        background: $boost;\n    }\n\n    ToggleButton:focus {\n        border: tall $accent;\n    }\n\n    ToggleButton:hover {\n        text-style: bold;\n        background: $boost;\n    }\n\n    ToggleButton:focus > .toggle--label {\n        text-style: underline;\n    }\n\n    /* Base button colours (including in dark mode). */\n\n    ToggleButton > .toggle--button {\n        color: $background;\n        text-style: bold;\n        background: $foreground 15%;\n    }\n\n    ToggleButton:focus > .toggle--button {\n        background: $foreground 25%;\n    }\n\n    ToggleButton.-on > .toggle--button {\n        color: $success;\n    }\n\n    ToggleButton.-on:focus > .toggle--button {\n        background: $foreground 25%;\n    }\n\n    /* Light mode overrides. */\n\n    ToggleButton:light > .toggle--button {\n        color: $background;\n        background: $foreground 10%;\n    }\n\n    ToggleButton:light:focus > .toggle--button {\n        background: $foreground 25%;\n    }\n\n    ToggleButton:light.-on > .toggle--button {\n        color: $primary;\n    }\n    '
    BUTTON_LEFT: str = '▐'
    'The character used for the left side of the toggle button.'
    BUTTON_INNER: str = 'X'
    'The character used for the inside of the button.'
    BUTTON_RIGHT: str = '▌'
    'The character used for the right side of the toggle button.'
    value: reactive[bool] = reactive(False, init=False)
    'The value of the button. `True` for on, `False` for off.'

    def __init__(self, label: TextType='', value: bool=False, button_first: bool=True, *, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Initialise the toggle.\n\n        Args:\n            label: The label for the toggle.\n            value: The initial value of the toggle.\n            button_first: Should the button come before the label, or after?\n            name: The name of the toggle.\n            id: The ID of the toggle in the DOM.\n            classes: The CSS classes of the toggle.\n            disabled: Whether the button is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._button_first = button_first
        with self.prevent(self.Changed):
            self.value = value
        self._label = Text.from_markup(label) if isinstance(label, str) else label
        try:
            self._label = self._label.split()[0]
        except IndexError:
            pass

    @property
    def label(self) -> Text:
        if False:
            i = 10
            return i + 15
        'The label associated with the button.'
        return self._label

    @property
    def _button(self) -> Text:
        if False:
            print('Hello World!')
        'The button, reflecting the current value.'
        button_style = self.get_component_rich_style('toggle--button')
        if not self.value:
            button_style += Style.from_color(self.background_colors[1].rich_color, button_style.bgcolor)
        side_style = Style.from_color(button_style.bgcolor, self.background_colors[1].rich_color)
        return Text.assemble((self.BUTTON_LEFT, side_style), (self.BUTTON_INNER, button_style), (self.BUTTON_RIGHT, side_style))

    def render(self) -> RenderResult:
        if False:
            return 10
        'Render the content of the widget.\n\n        Returns:\n            The content to render for the widget.\n        '
        button = self._button
        label = self._label.copy()
        label.stylize(self.get_component_rich_style('toggle--label', partial=True))
        spacer = ' ' if label else ''
        return Text.assemble(*((button, spacer, label) if self._button_first else (label, spacer, button)), no_wrap=True, overflow='ellipsis')

    def get_content_width(self, container: Size, viewport: Size) -> int:
        if False:
            print('Hello World!')
        return self._button.cell_len + (1 if self._label else 0) + self._label.cell_len

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        if False:
            print('Hello World!')
        return 1

    def toggle(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Toggle the value of the widget.\n\n        Returns:\n            The `ToggleButton` instance.\n        '
        self.value = not self.value
        return self

    def action_toggle(self) -> None:
        if False:
            return 10
        'Toggle the value of the widget when called as an action.\n\n        This would normally be used for a keyboard binding.\n        '
        self.toggle()

    async def _on_click(self, _: Click) -> None:
        """Toggle the value of the widget when clicked with the mouse."""
        self.toggle()

    class Changed(Message):
        """Posted when the value of the toggle button changes."""

        def __init__(self, toggle_button: ToggleButton, value: bool) -> None:
            if False:
                return 10
            'Initialise the message.\n\n            Args:\n                toggle_button: The toggle button sending the message.\n                value: The value of the toggle button.\n            '
            super().__init__()
            self._toggle_button = toggle_button
            'A reference to the toggle button that was changed.'
            self.value = value
            'The value of the toggle button after the change.'

    def watch_value(self) -> None:
        if False:
            i = 10
            return i + 15
        'React to the value being changed.\n\n        When triggered, the CSS class `-on` is applied to the widget if\n        `value` has become `True`, or it is removed if it has become\n        `False`. Subsequently a related `Changed` event will be posted.\n        '
        self.set_class(self.value, '-on')
        self.post_message(self.Changed(self, self.value))