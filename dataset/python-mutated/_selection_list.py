"""Provides a selection list widget, allowing one or more items to be selected."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, ClassVar, Generic, Iterable, TypeVar, cast
from rich.repr import Result
from rich.segment import Segment
from rich.style import Style
from rich.text import Text, TextType
from typing_extensions import Self
from ..binding import Binding
from ..messages import Message
from ..strip import Strip
from ._option_list import NewOptionListContent, Option, OptionList
from ._toggle_button import ToggleButton
SelectionType = TypeVar('SelectionType')
'The type for the value of a [`Selection`][textual.widgets.selection_list.Selection] in a [`SelectionList`][textual.widgets.SelectionList]'
MessageSelectionType = TypeVar('MessageSelectionType')
'The type for the value of a [`Selection`][textual.widgets.selection_list.Selection] in a [`SelectionList`][textual.widgets.SelectionList] message.'

class SelectionError(TypeError):
    """Type of an error raised if a selection is badly-formed."""

class Selection(Generic[SelectionType], Option):
    """A selection for a [`SelectionList`][textual.widgets.SelectionList]."""

    def __init__(self, prompt: TextType, value: SelectionType, initial_state: bool=False, id: str | None=None, disabled: bool=False):
        if False:
            return 10
        'Initialise the selection.\n\n        Args:\n            prompt: The prompt for the selection.\n            value: The value for the selection.\n            initial_state: The initial selected state of the selection.\n            id: The optional ID for the selection.\n            disabled: The initial enabled/disabled state. Enabled by default.\n        '
        if isinstance(prompt, str):
            prompt = Text.from_markup(prompt)
        super().__init__(prompt.split()[0], id, disabled)
        self._value: SelectionType = value
        'The value associated with the selection.'
        self._initial_state: bool = initial_state
        'The initial selected state for the selection.'

    @property
    def value(self) -> SelectionType:
        if False:
            for i in range(10):
                print('nop')
        'The value for this selection.'
        return self._value

    @property
    def initial_state(self) -> bool:
        if False:
            return 10
        'The initial selected state for the selection.'
        return self._initial_state

class SelectionList(Generic[SelectionType], OptionList):
    """A vertical selection list that allows making multiple selections."""
    BINDINGS = [Binding('space', 'select')]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | space | Toggle the state of the highlighted selection. |\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'selection-list--button', 'selection-list--button-selected', 'selection-list--button-highlighted', 'selection-list--button-selected-highlighted'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `selection-list--button` | Target the default button style. |\n    | `selection-list--button-selected` | Target a selected button style. |\n    | `selection-list--button-highlighted` | Target a highlighted button style. |\n    | `selection-list--button-selected-highlighted` | Target a highlighted selected button style. |\n    '
    DEFAULT_CSS = '\n    SelectionList {\n        height: auto;\n    }\n\n    SelectionList:light:focus > .selection-list--button-selected {\n        color: $primary;\n    }\n\n    SelectionList:light > .selection-list--button-selected-highlighted {\n        color: $primary;\n    }\n\n    SelectionList:light:focus > .selection-list--button-selected-highlighted {\n        color: $primary;\n    }\n\n    SelectionList > .selection-list--button {\n        text-style: bold;\n        background: $foreground 15%;\n    }\n\n    SelectionList:focus > .selection-list--button {\n        text-style: bold;\n        background: $foreground 25%;\n    }\n\n    SelectionList > .selection-list--button-highlighted {\n        text-style: bold;\n        background: $foreground 15%;\n    }\n\n    SelectionList:focus > .selection-list--button-highlighted {\n        text-style: bold;\n        background: $foreground 25%;\n    }\n\n    SelectionList > .selection-list--button-selected {\n        text-style: bold;\n        color: $success;\n        background: $foreground 15%;\n    }\n\n    SelectionList:focus > .selection-list--button-selected {\n        text-style: bold;\n        color: $success;\n        background: $foreground 25%;\n    }\n\n    SelectionList > .selection-list--button-selected-highlighted {\n        text-style: bold;\n        color: $success;\n        background: $foreground 15%;\n    }\n\n    SelectionList:focus > .selection-list--button-selected-highlighted {\n        text-style: bold;\n        color: $success;\n        background: $foreground 25%;\n    }\n    '

    class SelectionMessage(Generic[MessageSelectionType], Message):
        """Base class for all selection messages."""

        def __init__(self, selection_list: SelectionList, index: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Initialise the selection message.\n\n            Args:\n                selection_list: The selection list that owns the selection.\n                index: The index of the selection that the message relates to.\n            '
            super().__init__()
            self.selection_list: SelectionList[MessageSelectionType] = selection_list
            'The selection list that sent the message.'
            self.selection: Selection[MessageSelectionType] = selection_list.get_option_at_index(index)
            'The highlighted selection.'
            self.selection_index: int = index
            'The index of the selection that the message relates to.'

        @property
        def control(self) -> OptionList:
            if False:
                for i in range(10):
                    print('nop')
            'The selection list that sent the message.\n\n            This is an alias for\n            [`SelectionMessage.selection_list`][textual.widgets.SelectionList.SelectionMessage.selection_list]\n            and is used by the [`on`][textual.on] decorator.\n            '
            return self.selection_list

        def __rich_repr__(self) -> Result:
            if False:
                return 10
            yield ('selection_list', self.selection_list)
            yield ('selection', self.selection)
            yield ('selection_index', self.selection_index)

    class SelectionHighlighted(SelectionMessage):
        """Message sent when a selection is highlighted.

        Can be handled using `on_selection_list_selection_highlighted` in a subclass of
        [`SelectionList`][textual.widgets.SelectionList] or in a parent node in the DOM.
        """

    class SelectionToggled(SelectionMessage):
        """Message sent when a selection is toggled.

        Can be handled using `on_selection_list_selection_toggled` in a subclass of
        [`SelectionList`][textual.widgets.SelectionList] or in a parent node in the DOM.

        Note:
            This message is only sent if the selection is toggled by user
            interaction. See
            [`SelectedChanged`][textual.widgets.SelectionList.SelectedChanged]
            for a message sent when any change (selected or deselected,
            either by user interaction or by API calls) is made to the
            selected values.
        """

    @dataclass
    class SelectedChanged(Generic[MessageSelectionType], Message):
        """Message sent when the collection of selected values changes.

        This message is sent when any change to the collection of selected
        values takes place; either by user interaction or by API calls.
        """
        selection_list: SelectionList[MessageSelectionType]
        'The `SelectionList` that sent the message.'

        @property
        def control(self) -> SelectionList[MessageSelectionType]:
            if False:
                while True:
                    i = 10
            'An alias for `selection_list`.'
            return self.selection_list

    def __init__(self, *selections: Selection | tuple[TextType, SelectionType] | tuple[TextType, SelectionType, bool], name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            print('Hello World!')
        'Initialise the selection list.\n\n        Args:\n            *selections: The content for the selection list.\n            name: The name of the selection list.\n            id: The ID of the selection list in the DOM.\n            classes: The CSS classes of the selection list.\n            disabled: Whether the selection list is disabled or not.\n        '
        self._selected: dict[SelectionType, None] = {}
        'Tracking of which values are selected.'
        self._send_messages = False
        "Keep track of when we're ready to start sending messages."
        super().__init__(*[self._make_selection(selection) for selection in selections], name=name, id=id, classes=classes, disabled=disabled, wrap=False)

    @property
    def selected(self) -> list[SelectionType]:
        if False:
            return 10
        'The selected values.\n\n        This is a list of all of the\n        [values][textual.widgets.selection_list.Selection.value] associated\n        with selections in the list that are currently in the selected\n        state.\n        '
        return list(self._selected.keys())

    def _on_mount(self) -> None:
        if False:
            return 10
        'Configure the list once the DOM is ready.'
        self._send_messages = True

    def _message_changed(self) -> None:
        if False:
            print('Hello World!')
        'Post a message that the selected collection has changed, where appropriate.\n\n        Note:\n            A message will only be sent if `_send_messages` is `True`. This\n            makes this safe to call before the widget is ready for posting\n            messages.\n        '
        if self._send_messages:
            self.post_message(self.SelectedChanged(self))

    def _apply_to_all(self, state_change: Callable[[SelectionType], bool]) -> Self:
        if False:
            print('Hello World!')
        'Apply a selection state change to all selection options in the list.\n\n        Args:\n            state_change: The state change function to apply.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n\n        Note:\n            This method will post a single\n            [`SelectedChanged`][textual.widgets.OptionList.SelectedChanged]\n            message if a change is made in a call to this method.\n        '
        changed = False
        with self.prevent(self.SelectedChanged):
            for selection in self._options:
                changed = state_change(cast(Selection, selection).value) or changed
        if changed:
            self._message_changed()
        self.refresh()
        return self

    def _select(self, value: SelectionType) -> bool:
        if False:
            i = 10
            return i + 15
        'Mark the given value as selected.\n\n        Args:\n            value: The value to mark as selected.\n\n        Returns:\n            `True` if the value was selected, `False` if not.\n        '
        if value not in self._selected:
            self._selected[value] = None
            self._message_changed()
            return True
        return False

    def select(self, selection: Selection[SelectionType] | SelectionType) -> Self:
        if False:
            print('Hello World!')
        'Mark the given selection as selected.\n\n        Args:\n            selection: The selection to mark as selected.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        if self._select(selection.value if isinstance(selection, Selection) else cast(SelectionType, selection)):
            self.refresh()
        return self

    def select_all(self) -> Self:
        if False:
            return 10
        'Select all items.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        return self._apply_to_all(self._select)

    def _deselect(self, value: SelectionType) -> bool:
        if False:
            while True:
                i = 10
        'Mark the given selection as not selected.\n\n        Args:\n            value: The value to mark as not selected.\n\n        Returns:\n            `True` if the value was deselected, `False` if not.\n        '
        try:
            del self._selected[value]
        except KeyError:
            return False
        self._message_changed()
        return True

    def deselect(self, selection: Selection[SelectionType] | SelectionType) -> Self:
        if False:
            return 10
        'Mark the given selection as not selected.\n\n        Args:\n            selection: The selection to mark as not selected.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        if self._deselect(selection.value if isinstance(selection, Selection) else cast(SelectionType, selection)):
            self.refresh()
        return self

    def deselect_all(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Deselect all items.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        return self._apply_to_all(self._deselect)

    def _toggle(self, value: SelectionType) -> bool:
        if False:
            return 10
        'Toggle the selection state of the given value.\n\n        Args:\n            value: The value to toggle.\n\n        Returns:\n            `True`.\n        '
        if value in self._selected:
            self._deselect(value)
        else:
            self._select(value)
        return True

    def toggle(self, selection: Selection[SelectionType] | SelectionType) -> Self:
        if False:
            print('Hello World!')
        'Toggle the selected state of the given selection.\n\n        Args:\n            selection: The selection to toggle.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        self._toggle(selection.value if isinstance(selection, Selection) else cast(SelectionType, selection))
        self.refresh()
        return self

    def toggle_all(self) -> Self:
        if False:
            i = 10
            return i + 15
        'Toggle all items.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n        '
        return self._apply_to_all(self._toggle)

    def _make_selection(self, selection: Selection | tuple[TextType, SelectionType] | tuple[TextType, SelectionType, bool]) -> Selection[SelectionType]:
        if False:
            while True:
                i = 10
        'Turn incoming selection data into a `Selection` instance.\n\n        Args:\n            selection: The selection data.\n\n        Returns:\n            An instance of a `Selection`.\n\n        Raises:\n            SelectionError: If the selection was badly-formed.\n        '
        if isinstance(selection, tuple):
            if len(selection) == 2:
                selection = cast('tuple[TextType, SelectionType, bool]', (*selection, False))
            elif len(selection) != 3:
                raise SelectionError(f'Expected 2 or 3 values, got {len(selection)}')
            selection = Selection[SelectionType](*selection)
        assert isinstance(selection, Selection)
        if selection.initial_state:
            self._select(selection.value)
        return selection

    def _toggle_highlighted_selection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Toggle the state of the highlighted selection.\n\n        If nothing is selected in the list this is a non-operation.\n        '
        if self.highlighted is not None:
            self.toggle(self.get_option_at_index(self.highlighted))

    def _left_gutter_width(self) -> int:
        if False:
            print('Hello World!')
        'Returns the size of any left gutter that should be taken into account.\n\n        Returns:\n            The width of the left gutter.\n        '
        return len(ToggleButton.BUTTON_LEFT + ToggleButton.BUTTON_INNER + ToggleButton.BUTTON_RIGHT + ' ')

    def render_line(self, y: int) -> Strip:
        if False:
            for i in range(10):
                print('nop')
        'Render a line in the display.\n\n        Args:\n            y: The line to render.\n\n        Returns:\n            A [`Strip`][textual.strip.Strip] that is the line to render.\n        '
        prompt = super().render_line(y)
        if not prompt:
            return prompt
        (_, scroll_y) = self.scroll_offset
        selection_index = scroll_y + y
        selection = self.get_option_at_index(selection_index)
        component_style = 'selection-list--button'
        if selection.value in self._selected:
            component_style += '-selected'
        if self.highlighted == selection_index:
            component_style += '-highlighted'
        underlying_style = next(iter(prompt)).style
        assert underlying_style is not None
        button_style = self.get_component_rich_style(component_style)
        if selection.value not in self._selected:
            button_style += Style.from_color(self.background_colors[1].rich_color, button_style.bgcolor)
        side_style = Style.from_color(button_style.bgcolor, underlying_style.bgcolor)
        side_style += Style(meta={'option': selection_index})
        button_style += Style(meta={'option': selection_index})
        return Strip([Segment(ToggleButton.BUTTON_LEFT, style=side_style), Segment(ToggleButton.BUTTON_INNER, style=button_style), Segment(ToggleButton.BUTTON_RIGHT, style=side_style), Segment(' ', style=underlying_style), *prompt])

    def _on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if False:
            print('Hello World!')
        'Capture the `OptionList` highlight event and turn it into a [`SelectionList`][textual.widgets.SelectionList] event.\n\n        Args:\n            event: The event to capture and recreate.\n        '
        event.stop()
        self.post_message(self.SelectionHighlighted(self, event.option_index))

    def _on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if False:
            while True:
                i = 10
        'Capture the `OptionList` selected event and turn it into a [`SelectionList`][textual.widgets.SelectionList] event.\n\n        Args:\n            event: The event to capture and recreate.\n        '
        event.stop()
        self._toggle_highlighted_selection()
        self.post_message(self.SelectionToggled(self, event.option_index))

    def get_option_at_index(self, index: int) -> Selection[SelectionType]:
        if False:
            while True:
                i = 10
        'Get the selection option at the given index.\n\n        Args:\n            index: The index of the selection option to get.\n\n        Returns:\n            The selection option at that index.\n\n        Raises:\n            OptionDoesNotExist: If there is no selection option with the index.\n        '
        return cast('Selection[SelectionType]', super().get_option_at_index(index))

    def get_option(self, option_id: str) -> Selection[SelectionType]:
        if False:
            return 10
        'Get the selection option with the given ID.\n\n        Args:\n            option_id: The ID of the selection option to get.\n\n        Returns:\n            The selection option with the ID.\n\n        Raises:\n            OptionDoesNotExist: If no selection option has the given ID.\n        '
        return cast('Selection[SelectionType]', super().get_option(option_id))

    def _remove_option(self, index: int) -> None:
        if False:
            print('Hello World!')
        'Remove a selection option from the selection option list.\n\n        Args:\n            index: The index of the selection option to remove.\n\n        Raises:\n            IndexError: If there is no selection option of the given index.\n        '
        self._deselect(self.get_option_at_index(index).value)
        return super()._remove_option(index)

    def add_options(self, items: Iterable[NewOptionListContent | Selection | tuple[TextType, SelectionType] | tuple[TextType, SelectionType, bool]]) -> Self:
        if False:
            i = 10
            return i + 15
        'Add new selection options to the end of the list.\n\n        Args:\n            items: The new items to add.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n            SelectionError: If one of the selection options is of the wrong form.\n        '
        cleaned_options: list[Selection] = []
        for item in items:
            if isinstance(item, tuple):
                cleaned_options.append(self._make_selection(cast('tuple[TextType, SelectionType] | tuple[TextType, SelectionType, bool]', item)))
            elif isinstance(item, Selection):
                cleaned_options.append(self._make_selection(item))
            else:
                raise SelectionError('Only Selection or a prompt/value tuple is supported in SelectionList')
        return super().add_options(cleaned_options)

    def add_option(self, item: NewOptionListContent | Selection | tuple[TextType, SelectionType] | tuple[TextType, SelectionType, bool]=None) -> Self:
        if False:
            return 10
        'Add a new selection option to the end of the list.\n\n        Args:\n            item: The new item to add.\n\n        Returns:\n            The [`SelectionList`][textual.widgets.SelectionList] instance.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n            SelectionError: If the selection option is of the wrong form.\n        '
        return self.add_options([item])

    def clear_options(self) -> Self:
        if False:
            i = 10
            return i + 15
        'Clear the content of the selection list.\n\n        Returns:\n            The `SelectionList` instance.\n        '
        self._selected.clear()
        return super().clear_options()