"""Provides a RadioSet widget, which groups radio buttons."""
from __future__ import annotations
from typing import ClassVar, Optional
import rich.repr
from ..binding import Binding, BindingType
from ..containers import Container
from ..events import Click, Mount
from ..message import Message
from ..reactive import var
from ._radio_button import RadioButton

class RadioSet(Container, can_focus=True, can_focus_children=False):
    """Widget for grouping a collection of radio buttons into a set.

    When a collection of [`RadioButton`][textual.widgets.RadioButton]s are
    grouped with this widget, they will be treated as a mutually-exclusive
    grouping. If one button is turned on, the previously-on button will be
    turned off.
    """
    DEFAULT_CSS = '\n    RadioSet {\n        border: tall transparent;\n        background: $boost;\n        padding: 0 1 0 0;\n        height: auto;\n        width: auto;\n    }\n\n    RadioSet:focus {\n        border: tall $accent;\n    }\n\n    /* The following rules/styles mimic similar ToggleButton:focus rules in\n     * ToggleButton. If those styles ever get updated, these should be too.\n     */\n\n    RadioSet > * {\n        background: transparent;\n        border: none;\n        padding: 0 1;\n    }\n\n    RadioSet:focus > RadioButton.-selected > .toggle--label {\n        text-style: underline;\n    }\n\n    RadioSet:focus ToggleButton.-selected > .toggle--button {\n        background: $foreground 25%;\n    }\n\n    RadioSet:focus > RadioButton.-on.-selected > .toggle--button {\n        background: $foreground 25%;\n    }\n    '
    BINDINGS: ClassVar[list[BindingType]] = [Binding('down,right', 'next_button', '', show=False), Binding('enter,space', 'toggle', 'Toggle', show=False), Binding('up,left', 'previous_button', '', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter, space | Toggle the currently-selected button. |\n    | left, up | Select the previous radio button in the set. |\n    | right, down | Select the next radio button in the set. |\n    '
    _selected: var[int | None] = var[Optional[int]](None)
    'The index of the currently-selected radio button.'

    @rich.repr.auto
    class Changed(Message):
        """Posted when the pressed button in the set changes.

        This message can be handled using an `on_radio_set_changed` method.
        """
        ALLOW_SELECTOR_MATCH = {'pressed'}
        'Additional message attributes that can be used with the [`on` decorator][textual.on].'

        def __init__(self, radio_set: RadioSet, pressed: RadioButton) -> None:
            if False:
                print('Hello World!')
            'Initialise the message.\n\n            Args:\n                pressed: The radio button that was pressed.\n            '
            super().__init__()
            self.radio_set = radio_set
            'A reference to the [`RadioSet`][textual.widgets.RadioSet] that was changed.'
            self.pressed = pressed
            'The [`RadioButton`][textual.widgets.RadioButton] that was pressed to make the change.'
            self.index = radio_set.pressed_index
            'The index of the [`RadioButton`][textual.widgets.RadioButton] that was pressed to make the change.'

        @property
        def control(self) -> RadioSet:
            if False:
                while True:
                    i = 10
            'A reference to the [`RadioSet`][textual.widgets.RadioSet] that was changed.\n\n            This is an alias for [`Changed.radio_set`][textual.widgets.RadioSet.Changed.radio_set]\n            and is used by the [`on`][textual.on] decorator.\n            '
            return self.radio_set

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                for i in range(10):
                    print('nop')
            yield ('radio_set', self.radio_set)
            yield ('pressed', self.pressed)
            yield ('index', self.index)

    def __init__(self, *buttons: str | RadioButton, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Initialise the radio set.\n\n        Args:\n            buttons: The labels or [`RadioButton`][textual.widgets.RadioButton]s to group together.\n            name: The name of the radio set.\n            id: The ID of the radio set in the DOM.\n            classes: The CSS classes of the radio set.\n            disabled: Whether the radio set is disabled or not.\n\n        Note:\n            When a `str` label is provided, a\n            [RadioButton][textual.widgets.RadioButton] will be created from\n            it.\n        '
        self._pressed_button: RadioButton | None = None
        "Holds the radio buttons we're responsible for."
        super().__init__(*[button if isinstance(button, RadioButton) else RadioButton(button) for button in buttons], name=name, id=id, classes=classes, disabled=disabled)

    def _on_mount(self, _: Mount) -> None:
        if False:
            i = 10
            return i + 15
        'Perform some processing once mounted in the DOM.'
        if self._nodes:
            self._selected = 0
        buttons = list(self.query(RadioButton))
        for button in buttons:
            button.can_focus = False
        switched_on = [button for button in buttons if button.value]
        with self.prevent(RadioButton.Changed):
            for button in switched_on[1:]:
                button.value = False
        if switched_on:
            self._pressed_button = switched_on[0]

    def watch__selected(self) -> None:
        if False:
            i = 10
            return i + 15
        self.query(RadioButton).remove_class('-selected')
        if self._selected is not None:
            self._nodes[self._selected].add_class('-selected')

    def _on_radio_button_changed(self, event: RadioButton.Changed) -> None:
        if False:
            i = 10
            return i + 15
        'Respond to the value of a button in the set being changed.\n\n        Args:\n            event: The event.\n        '
        event.stop()
        with self.prevent(RadioButton.Changed):
            if event.radio_button.value:
                if self._pressed_button is not None and self._pressed_button != event.radio_button:
                    self._pressed_button.value = False
                self._pressed_button = event.radio_button
                self.post_message(self.Changed(self, event.radio_button))
            else:
                event.radio_button.value = True

    def _on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if False:
            while True:
                i = 10
        "Handle a change to which button in the set is pressed.\n\n        This handler ensures that, when a button is pressed, it's also the\n        selected button.\n        "
        self._selected = event.index

    async def _on_click(self, _: Click) -> None:
        """Handle a click on or within the radio set.

        This handler ensures that focus moves to the clicked radio set, even
        if there's a click on one of the radio buttons it contains.
        """
        self.focus()

    @property
    def pressed_button(self) -> RadioButton | None:
        if False:
            i = 10
            return i + 15
        'The currently-pressed [`RadioButton`][textual.widgets.RadioButton], or `None` if none are pressed.'
        return self._pressed_button

    @property
    def pressed_index(self) -> int:
        if False:
            print('Hello World!')
        'The index of the currently-pressed [`RadioButton`][textual.widgets.RadioButton], or -1 if none are pressed.'
        return self._nodes.index(self._pressed_button) if self._pressed_button is not None else -1

    def action_previous_button(self) -> None:
        if False:
            return 10
        'Navigate to the previous button in the set.\n\n        Note that this will wrap around to the end if at the start.\n        '
        if self._nodes:
            if self._selected == 0:
                self._selected = len(self.children) - 1
            elif self._selected is None:
                self._selected = 0
            else:
                self._selected -= 1

    def action_next_button(self) -> None:
        if False:
            while True:
                i = 10
        'Navigate to the next button in the set.\n\n        Note that this will wrap around to the start if at the end.\n        '
        if self._nodes:
            if self._selected is None or self._selected == len(self._nodes) - 1:
                self._selected = 0
            else:
                self._selected += 1

    def action_toggle(self) -> None:
        if False:
            while True:
                i = 10
        'Toggle the state of the currently-selected button.'
        if self._selected is not None:
            button = self._nodes[self._selected]
            assert isinstance(button, RadioButton)
            button.toggle()