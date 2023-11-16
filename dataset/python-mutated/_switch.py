from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
from rich.console import RenderableType
from ..binding import Binding, BindingType
from ..events import Click
from ..geometry import Size
from ..message import Message
from ..reactive import reactive
from ..scrollbar import ScrollBarRender
from ..widget import Widget
if TYPE_CHECKING:
    from typing_extensions import Self

class Switch(Widget, can_focus=True):
    """A switch widget that represents a boolean value.

    Can be toggled by clicking on it or through its [bindings][textual.widgets.Switch.BINDINGS].

    The switch widget also contains [component classes][textual.widgets.Switch.COMPONENT_CLASSES]
    that enable more customization.
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding('enter,space', 'toggle', 'Toggle', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter,space | Toggle the switch state. |\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'switch--slider'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `switch--slider` | Targets the slider of the switch. |\n    '
    DEFAULT_CSS = '\n    Switch {\n        border: tall transparent;\n        background: $boost;\n        height: auto;\n        width: auto;\n        padding: 0 2;\n    }\n\n    Switch > .switch--slider {\n        background: $panel-darken-2;\n        color: $panel-lighten-2;\n    }\n\n    Switch:hover {\n        border: tall $background;\n    }\n\n    Switch:focus {\n        border: tall $accent;\n    }\n\n    Switch.-on {\n\n    }\n\n    Switch.-on > .switch--slider {\n        color: $success;\n    }\n    '
    value = reactive(False, init=False)
    'The value of the switch; `True` for on and `False` for off.'
    slider_pos = reactive(0.0)
    'The position of the slider.'

    class Changed(Message):
        """Posted when the status of the switch changes.

        Can be handled using `on_switch_changed` in a subclass of `Switch`
        or in a parent widget in the DOM.

        Attributes:
            value: The value that the switch was changed to.
            switch: The `Switch` widget that was changed.
        """

        def __init__(self, switch: Switch, value: bool) -> None:
            if False:
                print('Hello World!')
            super().__init__()
            self.value: bool = value
            self.switch: Switch = switch

        @property
        def control(self) -> Switch:
            if False:
                while True:
                    i = 10
            'Alias for self.switch.'
            return self.switch

    def __init__(self, value: bool=False, *, animate: bool=True, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the switch.\n\n        Args:\n            value: The initial value of the switch.\n            animate: True if the switch should animate when toggled.\n            name: The name of the switch.\n            id: The ID of the switch in the DOM.\n            classes: The CSS classes of the switch.\n            disabled: Whether the switch is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        if value:
            self.slider_pos = 1.0
            self._reactive_value = value
        self._should_animate = animate

    def watch_value(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        target_slider_pos = 1.0 if value else 0.0
        if self._should_animate:
            self.animate('slider_pos', target_slider_pos, duration=0.3)
        else:
            self.slider_pos = target_slider_pos
        self.post_message(self.Changed(self, self.value))

    def watch_slider_pos(self, slider_pos: float) -> None:
        if False:
            i = 10
            return i + 15
        self.set_class(slider_pos == 1, '-on')

    def render(self) -> RenderableType:
        if False:
            return 10
        style = self.get_component_rich_style('switch--slider')
        return ScrollBarRender(virtual_size=100, window_size=50, position=self.slider_pos * 50, style=style, vertical=False)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        if False:
            i = 10
            return i + 15
        return 4

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        if False:
            return 10
        return 1

    async def _on_click(self, event: Click) -> None:
        """Toggle the state of the switch."""
        event.stop()
        self.toggle()

    def action_toggle(self) -> None:
        if False:
            while True:
                i = 10
        'Toggle the state of the switch.'
        self.toggle()

    def toggle(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Toggle the switch value.\n\n        As a result of the value changing, a `Switch.Changed` message will\n        be posted.\n\n        Returns:\n            The `Switch` instance.\n        '
        self.value = not self.value
        return self