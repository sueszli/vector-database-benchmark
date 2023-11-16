"""Provides the core of a classic vertical bounce-bar option list.

Useful as a lightweight list view (not to be confused with ListView, which
is much richer but uses widgets for the items) and as the base for various
forms of bounce-bar menu.
"""
from __future__ import annotations
from typing import ClassVar, Iterable, NamedTuple
from rich.console import RenderableType
from rich.padding import Padding
from rich.repr import Result
from rich.rule import Rule
from rich.style import Style
from typing_extensions import Literal, Self, TypeAlias
from ..binding import Binding, BindingType
from ..events import Click, Idle, Leave, MouseMove
from ..geometry import Region, Size
from ..message import Message
from ..reactive import reactive
from ..scroll_view import ScrollView
from ..strip import Strip

class DuplicateID(Exception):
    """Exception raised if a duplicate ID is used."""

class OptionDoesNotExist(Exception):
    """Exception raised when a request has been made for an option that doesn't exist."""

class Option:
    """Class that holds the details of an individual option."""

    def __init__(self, prompt: RenderableType, id: str | None=None, disabled: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialise the option.\n\n        Args:\n            prompt: The prompt for the option.\n            id: The optional ID for the option.\n            disabled: The initial enabled/disabled state. Enabled by default.\n        '
        self.__prompt = prompt
        self.__id = id
        self.disabled = disabled

    @property
    def prompt(self) -> RenderableType:
        if False:
            i = 10
            return i + 15
        'The prompt for the option.'
        return self.__prompt

    def set_prompt(self, prompt: RenderableType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the prompt for the option.\n\n        Args:\n            prompt: The new prompt for the option.\n        '
        self.__prompt = prompt

    @property
    def id(self) -> str | None:
        if False:
            i = 10
            return i + 15
        'The optional ID for the option.'
        return self.__id

    def __rich_repr__(self) -> Result:
        if False:
            i = 10
            return i + 15
        yield ('prompt', self.prompt)
        yield ('id', self.id, None)
        yield ('disabled', self.disabled, False)

class Separator:
    """Class used to add a separator to an [OptionList][textual.widgets.OptionList]."""

class Line(NamedTuple):
    """Class that holds a list of segments for the line of a option."""
    segments: Strip
    'The strip of segments that make up the line.'
    option_index: int | None = None
    "The index of the [Option][textual.widgets.option_list.Option] that this line is related to.\n\n    If the line isn't related to an option this will be `None`.\n    "

class OptionLineSpan(NamedTuple):
    """Class that holds the line span information for an option.

    An [Option][textual.widgets.option_list.Option] can have a prompt that
    spans multiple lines. Also, there's no requirement that every option in
    an option list has the same span information. So this structure is used
    to track the line that an option starts on, and how many lines it
    contains.
    """
    first: int
    'The line position for the start of the option..'
    line_count: int
    'The count of lines that make up the option.'

    def __contains__(self, line: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(line, int)
        return line >= self.first and line < self.first + self.line_count
OptionListContent: TypeAlias = 'Option | Separator'
'The type of an item of content in the option list.\n\nThis type represents all of the types that will be found in the list of\ncontent of the option list after it has been processed for addition.\n'
NewOptionListContent: TypeAlias = 'OptionListContent | None | RenderableType'
'The type of a new item of option list content to be added to an option list.\n\nThis type represents all of the types that will be accepted when adding new\ncontent to the option list. This is a superset of `OptionListContent`.\n'

class OptionList(ScrollView, can_focus=True):
    """A vertical option list with bounce-bar highlighting."""
    BINDINGS: ClassVar[list[BindingType]] = [Binding('down', 'cursor_down', 'Down', show=False), Binding('end', 'last', 'Last', show=False), Binding('enter', 'select', 'Select', show=False), Binding('home', 'first', 'First', show=False), Binding('pagedown', 'page_down', 'Page Down', show=False), Binding('pageup', 'page_up', 'Page Up', show=False), Binding('up', 'cursor_up', 'Up', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | down | Move the highlight down. |\n    | end | Move the highlight to the last option. |\n    | enter | Select the current option. |\n    | home | Move the highlight to the first option. |\n    | pagedown | Move the highlight down a page of options. |\n    | pageup | Move the highlight up a page of options. |\n    | up | Move the highlight up. |\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'option-list--option', 'option-list--option-disabled', 'option-list--option-highlighted', 'option-list--option-highlighted-disabled', 'option-list--option-hover', 'option-list--option-hover-disabled', 'option-list--option-hover-highlighted', 'option-list--option-hover-highlighted-disabled', 'option-list--separator'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `option-list--option-disabled` | Target disabled options. |\n    | `option-list--option-highlighted` | Target the highlighted option. |\n    | `option-list--option-highlighted-disabled` | Target a disabled option that is also highlighted. |\n    | `option-list--option-hover` | Target an option that has the mouse over it. |\n    | `option-list--option-hover-disabled` | Target a disabled option that has the mouse over it. |\n    | `option-list--option-hover-highlighted` | Target a highlighted option that has the mouse over it. |\n    | `option-list--option-hover-highlighted-disabled` | Target a disabled highlighted option that has the mouse over it. |\n    | `option-list--separator` | Target the separators. |\n    '
    DEFAULT_CSS = '\n    OptionList {\n        height: auto;\n        background: $boost;\n        color: $text;\n        overflow-x: hidden;\n        border: tall transparent;\n        padding: 0 1;\n    }\n\n    OptionList:focus {\n        border: tall $accent;\n\n    }\n\n    OptionList > .option-list--separator {\n        color: $foreground 15%;\n    }\n\n    OptionList > .option-list--option-highlighted {\n        color: $text;\n        text-style: bold;\n    }\n\n    OptionList:focus > .option-list--option-highlighted {\n        background: $accent;\n    }\n\n    OptionList > .option-list--option-disabled {\n        color: $text-disabled;\n    }\n\n    OptionList > .option-list--option-highlighted-disabled {\n        color: $text-disabled;\n        background: $accent 20%;\n    }\n\n    OptionList:focus > .option-list--option-highlighted-disabled {\n        background: $accent 30%;\n    }\n\n    OptionList > .option-list--option-hover {\n        background: $boost;\n    }\n\n    OptionList > .option-list--option-hover-disabled {\n        color: $text-disabled;\n        background: $boost;\n    }\n\n    OptionList > .option-list--option-hover-highlighted {\n        background: $accent 60%;\n        color: $text;\n        text-style: bold;\n    }\n\n    OptionList:focus > .option-list--option-hover-highlighted {\n        background: $accent;\n        color: $text;\n        text-style: bold;\n    }\n\n    OptionList > .option-list--option-hover-highlighted-disabled {\n        color: $text-disabled;\n        background: $accent 60%;\n    }\n    '
    highlighted: reactive[int | None] = reactive['int | None'](None)
    'The index of the currently-highlighted option, or `None` if no option is highlighted.'

    class OptionMessage(Message):
        """Base class for all option messages."""

        def __init__(self, option_list: OptionList, index: int) -> None:
            if False:
                while True:
                    i = 10
            'Initialise the option message.\n\n            Args:\n                option_list: The option list that owns the option.\n                index: The index of the option that the message relates to.\n            '
            super().__init__()
            self.option_list: OptionList = option_list
            'The option list that sent the message.'
            self.option: Option = option_list.get_option_at_index(index)
            'The highlighted option.'
            self.option_id: str | None = self.option.id
            'The ID of the option that the message relates to.'
            self.option_index: int = index
            'The index of the option that the message relates to.'

        @property
        def control(self) -> OptionList:
            if False:
                while True:
                    i = 10
            'The option list that sent the message.\n\n            This is an alias for [`OptionMessage.option_list`][textual.widgets.OptionList.OptionMessage.option_list]\n            and is used by the [`on`][textual.on] decorator.\n            '
            return self.option_list

        def __rich_repr__(self) -> Result:
            if False:
                for i in range(10):
                    print('nop')
            yield ('option_list', self.option_list)
            yield ('option', self.option)
            yield ('option_id', self.option_id)
            yield ('option_index', self.option_index)

    class OptionHighlighted(OptionMessage):
        """Message sent when an option is highlighted.

        Can be handled using `on_option_list_option_highlighted` in a subclass of
        `OptionList` or in a parent node in the DOM.
        """

    class OptionSelected(OptionMessage):
        """Message sent when an option is selected.

        Can be handled using `on_option_list_option_selected` in a subclass of
        `OptionList` or in a parent node in the DOM.
        """

    def __init__(self, *content: NewOptionListContent, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False, wrap: bool=True):
        if False:
            return 10
        'Initialise the option list.\n\n        Args:\n            *content: The content for the option list.\n            name: The name of the option list.\n            id: The ID of the option list in the DOM.\n            classes: The CSS classes of the option list.\n            disabled: Whether the option list is disabled or not.\n            wrap: Should prompts be auto-wrapped?\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._needs_refresh_content_tracking = False
        self._needs_to_scroll_to_highlight = False
        self._wrap = wrap
        'Should we auto-wrap options?\n\n        If `False` options wider than the list will be truncated.\n        '
        self._contents: list[OptionListContent] = [self._make_content(item) for item in content]
        "A list of the content of the option list.\n\n        This is *every* item that makes up the content of the option list;\n        this includes both the options *and* the separators (and any other\n        decoration we could end up adding -- although I don't anticipate\n        anything else at the moment; but padding around separators could be\n        a thing, perhaps).\n        "
        self._options: list[Option] = [content for content in self._contents if isinstance(content, Option)]
        "A list of the options within the option list.\n\n        This is a list of references to just the options alone, ignoring the\n        separators and potentially any other line-oriented option list\n        content that isn't an option.\n        "
        self._option_ids: dict[str, int] = {}
        'A dictionary of option IDs and the option indexes they relate to.'
        self._lines: list[Line] = []
        'A list of all of the individual lines that make up the option list.\n\n        Note that the size of this list will be at least the same as the number\n        of options, and actually greater if any prompt of any option is\n        multiple lines.\n        '
        self._spans: list[OptionLineSpan] = []
        'A list of the locations and sizes of all options in the option list.\n\n        This will be the same size as the number of prompts; each entry in\n        the list contains the line offset of the start of the prompt, and\n        the count of the lines in the prompt.\n        '
        self._request_content_tracking_refresh()
        self._mouse_hovering_over: int | None = None
        'Used to track what the mouse is hovering over.'
        self.highlighted = None

    def _request_content_tracking_refresh(self, rescroll_to_highlight: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Request that the content tracking information gets refreshed.\n\n        Args:\n            rescroll_to_highlight: Should the widget ensure the highlight is visible?\n\n        Calling this method sets a flag to say the refresh should happen,\n        and books the refresh call in for the next idle moment.\n        '
        self._needs_refresh_content_tracking = True
        self._needs_to_scroll_to_highlight = rescroll_to_highlight
        self.check_idle()

    async def _on_idle(self, _: Idle) -> None:
        """Perform content tracking data refresh when idle."""
        self._refresh_content_tracking()
        if self._needs_to_scroll_to_highlight:
            self._needs_to_scroll_to_highlight = False
            self.scroll_to_highlight()

    def watch_show_vertical_scrollbar(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handle the vertical scrollbar visibility status changing.\n\n        `show_vertical_scrollbar` is watched because it has an impact on the\n        available width in which to render the renderables that make up the\n        options in the list. If a vertical scrollbar appears or disappears\n        we need to recalculate all the lines that make up the list.\n        '
        self._request_content_tracking_refresh()

    def _on_resize(self) -> None:
        if False:
            while True:
                i = 10
        'Refresh the layout of the renderables in the list when resized.'
        self._request_content_tracking_refresh(rescroll_to_highlight=True)

    def _on_mouse_move(self, event: MouseMove) -> None:
        if False:
            while True:
                i = 10
        'React to the mouse moving.\n\n        Args:\n            event: The mouse movement event.\n        '
        self._mouse_hovering_over = event.style.meta.get('option')

    def _on_leave(self, _: Leave) -> None:
        if False:
            print('Hello World!')
        'React to the mouse leaving the widget.'
        self._mouse_hovering_over = None

    async def _on_click(self, event: Click) -> None:
        """React to the mouse being clicked on an item.

        Args:
            event: The click event.
        """
        clicked_option = event.style.meta.get('option')
        if clicked_option is not None:
            self.highlighted = clicked_option
            self.action_select()

    def _make_content(self, content: NewOptionListContent) -> OptionListContent:
        if False:
            i = 10
            return i + 15
        'Convert a single item of content for the list into a content type.\n\n        Args:\n            content: The content to turn into a full option list type.\n\n        Returns:\n            The content, usable in the option list.\n        '
        if isinstance(content, (Option, Separator)):
            return content
        if content is None:
            return Separator()
        return Option(content)

    def _clear_content_tracking(self) -> None:
        if False:
            while True:
                i = 10
        'Clear down the content tracking information.'
        self._lines.clear()
        self._spans.clear()
        self._option_ids.clear()

    def _left_gutter_width(self) -> int:
        if False:
            while True:
                i = 10
        'Returns the size of any left gutter that should be taken into account.\n\n        Returns:\n            The width of the left gutter.\n        '
        return 0

    def _refresh_content_tracking(self, force: bool=False) -> None:
        if False:
            return 10
        'Refresh the various forms of option list content tracking.\n\n        Args:\n            force: Optionally force the refresh.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n\n        Without a `force` the refresh will only take place if it has been\n        requested via `_refresh_content_tracking`.\n        '
        if not self._needs_refresh_content_tracking and (not force):
            return
        if not self.size.width:
            return
        self._clear_content_tracking()
        self._needs_refresh_content_tracking = False
        lines_from = self.app.console.render_lines
        add_span = self._spans.append
        option_ids = self._option_ids
        add_lines = self._lines.extend
        options = self.app.console.options.update_width(self.scrollable_content_region.width - self._left_gutter_width())
        options.no_wrap = not self._wrap
        if not self._wrap:
            options.overflow = 'ellipsis'
        separator = Strip(lines_from(Rule(style=''))[0])
        line = 0
        option = 0
        padding = self.get_component_styles('option-list--option').padding
        for content in self._contents:
            if isinstance(content, Option):
                new_lines = [Line(Strip(prompt_line).apply_style(Style(meta={'option': option})), option) for prompt_line in lines_from(Padding(content.prompt, padding) if padding else content.prompt, options)]
                add_span(OptionLineSpan(line, len(new_lines)))
                if content.id is not None:
                    option_ids[content.id] = option
                option += 1
            else:
                new_lines = [Line(separator)]
            add_lines(new_lines)
            line += len(new_lines)
        self.virtual_size = Size(self.scrollable_content_region.width, len(self._lines))

    def _duplicate_id_check(self, candidate_items: list[OptionListContent]) -> None:
        if False:
            return 10
        'Check the items to be added for any duplicates.\n\n        Args:\n            candidate_items: The items that are going be added.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n        '
        new_options = [item for item in candidate_items if isinstance(item, Option) and item.id is not None]
        new_option_ids = {option.id for option in new_options}
        if len(new_options) != len(new_option_ids) or not new_option_ids.isdisjoint(self._option_ids):
            raise DuplicateID('Attempt made to add options with duplicate IDs.')

    def add_options(self, items: Iterable[NewOptionListContent]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Add new options to the end of the option list.\n\n        Args:\n            items: The new items to add.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n\n        Note:\n            All options are checked for duplicate IDs *before* any option is\n            added. A duplicate ID will cause none of the passed items to be\n            added to the option list.\n        '
        if items:
            content = [self._make_content(item) for item in items]
            self._duplicate_id_check(content)
            self._contents.extend(content)
            self._options.extend([item for item in content if isinstance(item, Option)])
            self._refresh_content_tracking(force=True)
            self.refresh()
        return self

    def add_option(self, item: NewOptionListContent=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Add a new option to the end of the option list.\n\n        Args:\n            item: The new item to add.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            DuplicateID: If there is an attempt to use a duplicate ID.\n        '
        return self.add_options([item])

    def _remove_option(self, index: int) -> None:
        if False:
            while True:
                i = 10
        'Remove an option from the option list.\n\n        Args:\n            index: The index of the item to remove.\n\n        Raises:\n            IndexError: If there is no option of the given index.\n        '
        option = self._options[index]
        del self._options[index]
        del self._contents[self._contents.index(option)]
        self._refresh_content_tracking(force=True)
        self.highlighted = self.highlighted
        self._mouse_hovering_over = None
        self.refresh()

    def remove_option(self, option_id: str) -> Self:
        if False:
            i = 10
            return i + 15
        'Remove the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to remove.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        self._remove_option(self.get_option_index(option_id))
        return self

    def remove_option_at_index(self, index: int) -> Self:
        if False:
            return 10
        'Remove the option at the given index.\n\n        Args:\n            index: The index of the option to remove.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        try:
            self._remove_option(index)
        except IndexError:
            raise OptionDoesNotExist(f'There is no option with an index of {index}') from None
        return self

    def _replace_option_prompt(self, index: int, prompt: RenderableType) -> None:
        if False:
            print('Hello World!')
        'Replace the prompt of an option in the list.\n\n        Args:\n            index: The index of the option to replace the prompt of.\n            prompt: The new prompt for the option.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        self.get_option_at_index(index).set_prompt(prompt)
        self._refresh_content_tracking(force=True)
        self.refresh()

    def replace_option_prompt(self, option_id: str, prompt: RenderableType) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Replace the prompt of the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to replace the prompt of.\n            prompt: The new prompt for the option.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        self._replace_option_prompt(self.get_option_index(option_id), prompt)
        return self

    def replace_option_prompt_at_index(self, index: int, prompt: RenderableType) -> Self:
        if False:
            print('Hello World!')
        'Replace the prompt of the option at the given index.\n\n        Args:\n            index: The index of the option to replace the prompt of.\n            prompt: The new prompt for the option.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        self._replace_option_prompt(index, prompt)
        return self

    def clear_options(self) -> Self:
        if False:
            while True:
                i = 10
        'Clear the content of the option list.\n\n        Returns:\n            The `OptionList` instance.\n        '
        self._contents.clear()
        self._options.clear()
        self.highlighted = None
        self._mouse_hovering_over = None
        self.virtual_size = Size(self.scrollable_content_region.width, 0)
        self._refresh_content_tracking(force=True)
        return self

    def _set_option_disabled(self, index: int, disabled: bool) -> Self:
        if False:
            i = 10
            return i + 15
        'Set the disabled state of an option in the list.\n\n        Args:\n            index: The index of the option to set the disabled state of.\n            disabled: The disabled state to set.\n\n        Returns:\n            The `OptionList` instance.\n        '
        self._options[index].disabled = disabled
        self.refresh()
        return self

    def enable_option_at_index(self, index: int) -> Self:
        if False:
            i = 10
            return i + 15
        'Enable the option at the given index.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        try:
            return self._set_option_disabled(index, False)
        except IndexError:
            raise OptionDoesNotExist(f'There is no option with an index of {index}') from None

    def disable_option_at_index(self, index: int) -> Self:
        if False:
            while True:
                i = 10
        'Disable the option at the given index.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        try:
            return self._set_option_disabled(index, True)
        except IndexError:
            raise OptionDoesNotExist(f'There is no option with an index of {index}') from None

    def enable_option(self, option_id: str) -> Self:
        if False:
            return 10
        'Enable the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to enable.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        return self.enable_option_at_index(self.get_option_index(option_id))

    def disable_option(self, option_id: str) -> Self:
        if False:
            return 10
        'Disable the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to disable.\n\n        Returns:\n            The `OptionList` instance.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        return self.disable_option_at_index(self.get_option_index(option_id))

    @property
    def option_count(self) -> int:
        if False:
            i = 10
            return i + 15
        'The count of options.'
        return len(self._options)

    def get_option_at_index(self, index: int) -> Option:
        if False:
            for i in range(10):
                print('nop')
        'Get the option at the given index.\n\n        Args:\n            index: The index of the option to get.\n\n        Returns:\n            The option at that index.\n\n        Raises:\n            OptionDoesNotExist: If there is no option with the given index.\n        '
        try:
            return self._options[index]
        except IndexError:
            raise OptionDoesNotExist(f'There is no option with an index of {index}') from None

    def get_option(self, option_id: str) -> Option:
        if False:
            while True:
                i = 10
        'Get the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to get.\n\n        Returns:\n            The option with the ID.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        return self.get_option_at_index(self.get_option_index(option_id))

    def get_option_index(self, option_id: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get the index of the option with the given ID.\n\n        Args:\n            option_id: The ID of the option to get the index of.\n\n        Returns:\n            The index of the item with the given ID.\n\n        Raises:\n            OptionDoesNotExist: If no option has the given ID.\n        '
        try:
            return self._option_ids[option_id]
        except KeyError:
            raise OptionDoesNotExist(f"There is no option with an ID of '{option_id}'") from None

    def render_line(self, y: int) -> Strip:
        if False:
            print('Hello World!')
        'Render a single line in the option list.\n\n        Args:\n            y: The Y offset of the line to render.\n\n        Returns:\n            A `Strip` instance for the caller to render.\n        '
        (scroll_x, scroll_y) = self.scroll_offset
        line_number = scroll_y + y
        try:
            line = self._lines[line_number]
        except IndexError:
            return Strip([])
        option_index = line.option_index
        strip = line.segments
        if option_index is None:
            return strip.apply_style(self.get_component_rich_style('option-list--separator'))
        strip = strip.crop(scroll_x, scroll_x + self.scrollable_content_region.width)
        highlighted = self.highlighted
        mouse_over = self._mouse_hovering_over
        spans = self._spans
        if self._options[option_index].disabled:
            if option_index == highlighted:
                return strip.apply_style(self.get_component_rich_style('option-list--option-hover-highlighted-disabled' if option_index == mouse_over else 'option-list--option-highlighted-disabled'))
            if option_index == mouse_over:
                return strip.apply_style(self.get_component_rich_style('option-list--option-hover-disabled'))
            return strip.apply_style(self.get_component_rich_style('option-list--option-disabled'))
        if highlighted is not None and line_number in spans[highlighted]:
            if option_index == mouse_over:
                return strip.apply_style(self.get_component_rich_style('option-list--option-hover-highlighted'))
            return strip.apply_style(self.get_component_rich_style('option-list--option-highlighted'))
        if mouse_over is not None and line_number in spans[mouse_over]:
            return strip.apply_style(self.get_component_rich_style('option-list--option-hover'))
        return strip.apply_style(self.rich_style)

    def scroll_to_highlight(self, top: bool=False) -> None:
        if False:
            return 10
        'Ensure that the highlighted option is in view.\n\n        Args:\n            top: Scroll highlight to top of the list.\n\n        '
        highlighted = self.highlighted
        if highlighted is None:
            return
        try:
            span = self._spans[highlighted]
        except IndexError:
            return
        self.scroll_to_region(Region(0, span.first, self.scrollable_content_region.width, span.line_count), force=True, animate=False, top=top)

    def validate_highlighted(self, highlighted: int | None) -> int | None:
        if False:
            print('Hello World!')
        'Validate the `highlighted` property value on access.'
        if not self._options:
            return None
        if highlighted is None or highlighted < 0:
            return 0
        return min(highlighted, len(self._options) - 1)

    def watch_highlighted(self, highlighted: int | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'React to the highlighted option having changed.'
        if highlighted is not None:
            self.scroll_to_highlight()
            if not self._options[highlighted].disabled:
                self.post_message(self.OptionHighlighted(self, highlighted))

    def action_cursor_up(self) -> None:
        if False:
            print('Hello World!')
        'Move the highlight up by one option.'
        if self.highlighted is not None:
            if self.highlighted > 0:
                self.highlighted -= 1
            else:
                self.highlighted = len(self._options) - 1
        elif self._options:
            self.action_first()

    def action_cursor_down(self) -> None:
        if False:
            print('Hello World!')
        'Move the highlight down by one option.'
        if self.highlighted is not None:
            if self.highlighted < len(self._options) - 1:
                self.highlighted += 1
            else:
                self.highlighted = 0
        elif self._options:
            self.action_first()

    def action_first(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Move the highlight to the first option.'
        if self._options:
            self.highlighted = 0

    def action_last(self) -> None:
        if False:
            print('Hello World!')
        'Move the highlight to the last option.'
        if self._options:
            self.highlighted = len(self._options) - 1

    def _page(self, direction: Literal[-1, 1]) -> None:
        if False:
            i = 10
            return i + 15
        'Move the highlight by one page.\n\n        Args:\n            direction: The direction to head, -1 for up and 1 for down.\n        '
        fallback = self.action_first if direction == -1 else self.action_last
        highlighted = self.highlighted
        if highlighted is None:
            fallback()
        else:
            target_line = max(0, self._spans[highlighted].first + direction * self.scrollable_content_region.height)
            try:
                target_option = self._lines[target_line].option_index
            except IndexError:
                fallback()
            else:
                self.highlighted = target_option

    def action_page_up(self):
        if False:
            for i in range(10):
                print('nop')
        'Move the highlight up one page.'
        self._page(-1)

    def action_page_down(self):
        if False:
            print('Hello World!')
        'Move the highlight down one page.'
        self._page(1)

    def action_select(self) -> None:
        if False:
            print('Hello World!')
        'Select the currently-highlighted option.\n\n        If no option is selected, then nothing happens. If an option is\n        selected, a [OptionList.OptionSelected][textual.widgets.OptionList.OptionSelected]\n        message will be posted.\n        '
        highlighted = self.highlighted
        if highlighted is not None and (not self._options[highlighted].disabled):
            self.post_message(self.OptionSelected(self, highlighted))