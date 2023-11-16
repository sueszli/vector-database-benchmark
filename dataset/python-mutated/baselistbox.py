"""This is the baseclass for list box types"""
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod
from asciimatics.event import KeyboardEvent, MouseEvent
from asciimatics.screen import Screen
from asciimatics.widgets.widget import Widget
from asciimatics.widgets.scrollbar import _ScrollBar

class _BaseListBox(Widget, metaclass=ABCMeta):
    """
    An Internal class to contain common function between list box types.
    """
    __slots__ = ['_options', '_titles', '_line', '_start_line', '_required_height', '_on_change', '_on_select', '_validator', '_search', '_last_search', '_scroll_bar', '_parser']

    def __init__(self, height, options, titles=None, label=None, name=None, parser=None, on_change=None, on_select=None, validator=None):
        if False:
            while True:
                i = 10
        '\n        :param height: The required number of input lines for this widget.\n        :param options: The options for each row in the widget.\n        :param label: An optional label for the widget.\n        :param name: The name for the widget.\n        :param parser: Optional parser to colour text.\n        :param on_change: Optional function to call when selection changes.\n        :param on_select: Optional function to call when the user actually selects an entry from\n            this list - e.g. by double-clicking or pressing Enter.\n        :param validator: Optional function to validate selection for this widget.\n        '
        super().__init__(name)
        self._titles = titles
        self._label = label
        self._parser = parser
        self._options = self._parse_options(options)
        self._line = 0
        self._value = None
        self._start_line = 0
        self._required_height = height
        self._on_change = on_change
        self._on_select = on_select
        self._validator = validator
        self._search = ''
        self._last_search = datetime.now()
        self._scroll_bar = None

    def reset(self):
        if False:
            i = 10
            return i + 15
        pass

    def process_event(self, event):
        if False:
            i = 10
            return i + 15
        if isinstance(event, KeyboardEvent):
            if len(self._options) > 0 and event.key_code == Screen.KEY_UP:
                self._line = max(0, self._line - 1)
                self.value = self._options[self._line][1]
            elif len(self._options) > 0 and event.key_code == Screen.KEY_DOWN:
                self._line = min(len(self._options) - 1, self._line + 1)
                self.value = self._options[self._line][1]
            elif len(self._options) > 0 and event.key_code == Screen.KEY_PAGE_UP:
                self._line = max(0, self._line - self._h + (1 if self._titles else 0))
                self.value = self._options[self._line][1]
            elif len(self._options) > 0 and event.key_code == Screen.KEY_PAGE_DOWN:
                self._line = min(len(self._options) - 1, self._line + self._h - (1 if self._titles else 0))
                self.value = self._options[self._line][1]
            elif event.key_code in [Screen.ctrl('m'), Screen.ctrl('j')]:
                if self._on_select:
                    self._on_select()
            elif event.key_code > 0:
                now = datetime.now()
                if now - self._last_search >= timedelta(seconds=1):
                    self._search = ''
                self._search += chr(event.key_code)
                self._last_search = now
                new_value = self._find_option(self._search)
                if new_value is not None:
                    self.value = new_value
            else:
                return event
        elif isinstance(event, MouseEvent):
            if event.buttons != 0:
                if len(self._options) > 0 and self.is_mouse_over(event, include_label=False, width_modifier=1 if self._scroll_bar else 0):
                    new_line = event.y - self._y + self._start_line
                    if self._titles:
                        new_line -= 1
                    new_line = min(new_line, len(self._options) - 1)
                    if new_line >= 0:
                        self._line = new_line
                        self.value = self._options[self._line][1]
                        if event.buttons & MouseEvent.DOUBLE_CLICK != 0 and self._on_select:
                            self._on_select()
                    return None
                if self._scroll_bar:
                    if self._scroll_bar.process_event(event):
                        return None
            return event
        else:
            return event
        return None

    def _add_or_remove_scrollbar(self, width, height, dy):
        if False:
            print('Hello World!')
        '\n        Add or remove a scrollbar from this listbox based on height and available options.\n\n        :param width: Width of the Listbox\n        :param height: Height of the Listbox.\n        :param dy: Vertical offset from top of widget.\n        '
        if self._scroll_bar is None and len(self._options) > height:
            self._scroll_bar = _ScrollBar(self._frame.canvas, self._frame.palette, self._x + width - 1, self._y + dy, height, self._get_pos, self._set_pos)
        elif self._scroll_bar is not None and len(self._options) <= height:
            self._scroll_bar = None

    def _get_pos(self):
        if False:
            i = 10
            return i + 15
        '\n        Get current position for scroll bar.\n        '
        if self._h >= len(self._options):
            return 0
        return self._start_line / (len(self._options) - self._h)

    def _set_pos(self, pos):
        if False:
            i = 10
            return i + 15
        '\n        Set current position for scroll bar.\n        '
        if self._h < len(self._options):
            pos *= len(self._options) - self._h
            pos = int(round(max(0, pos), 0))
            self._start_line = pos

    @abstractmethod
    def _find_option(self, search_value):
        if False:
            i = 10
            return i + 15
        '\n        Internal function called by the BaseListBox to do a text search on user input.\n\n        :param search_value: The string value to search for in the list.\n        :return: The value of the matching option (or None if nothing matches).\n        '

    def required_height(self, offset, width):
        if False:
            print('Hello World!')
        return self._required_height

    @property
    def start_line(self):
        if False:
            while True:
                i = 10
        '\n        The line that will be drawn at the top of the visible section of this list.\n        '
        return self._start_line

    @start_line.setter
    def start_line(self, new_value):
        if False:
            for i in range(10):
                print('nop')
        if 0 <= new_value < len(self._options):
            self._start_line = new_value

    @property
    def value(self):
        if False:
            print('Hello World!')
        '\n        The current value for this list box.\n        '
        return self._value

    @value.setter
    def value(self, new_value):
        if False:
            print('Hello World!')
        old_value = self._value
        self._value = new_value
        for (i, [_, value]) in enumerate(self._options):
            if value == new_value:
                self._line = i
                break
        else:
            if len(self._options) > 0:
                self._line = 0
                self._value = self._options[self._line][1]
            else:
                self._line = -1
                self._value = None
        if self._validator:
            self._is_valid = self._validator(self._value)
        if old_value != self._value and self._on_change:
            self._on_change()
        self._start_line = max(0, self._line - self._h + 1, min(self._start_line, self._line))

    def _parse_options(self, options):
        if False:
            print('Hello World!')
        '\n        Parse a the options list for ColouredText.\n\n        :param options: the options list to parse\n        :returns: the options list parsed and converted to ColouredText as needed.\n        '
        if self._parser:
            parsed_value = []
            for option in options:
                parsed_value.append((self._parse_option(option[0]), option[1]))
            return parsed_value
        return options

    @abstractmethod
    def _parse_option(self, option):
        if False:
            return 10
        '\n        Parse a single option for ColouredText.\n\n        :param option: the option to parse\n        :returns: the option parsed and converted to ColouredText.\n        '

    @property
    @abstractmethod
    def options(self):
        if False:
            return 10
        '\n        The list of options available for user selection.\n        '