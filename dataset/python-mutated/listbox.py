"""This module implements the listbox widget"""
from asciimatics.strings import ColouredText
from asciimatics.widgets.utilities import _enforce_width
from asciimatics.widgets.baselistbox import _BaseListBox

class ListBox(_BaseListBox):
    """
    A ListBox is a widget that displays a list from which the user can select one option.
    """

    def __init__(self, height, options, centre=False, label=None, name=None, add_scroll_bar=False, parser=None, on_change=None, on_select=None, validator=None):
        if False:
            while True:
                i = 10
        '\n        :param height: The required number of input lines for this ListBox.\n        :param options: The options for each row in the widget.\n        :param centre: Whether to centre the selected line in the list.\n        :param label: An optional label for the widget.\n        :param name: The name for the ListBox.\n        :param parser: Optional parser to colour text.\n        :param on_change: Optional function to call when selection changes.\n        :param on_select: Optional function to call when the user actually selects an entry from\n        :param validator: Optional function to validate selection for this widget.\n\n        The `options` are a list of tuples, where the first value is the string to be displayed\n        to the user and the second is an interval value to identify the entry to the program.\n        For example:\n\n            options=[("First option", 1), ("Second option", 2)]\n        '
        super().__init__(height, options, label=label, name=name, parser=parser, on_change=on_change, on_select=on_select, validator=validator)
        self._centre = centre
        self._add_scroll_bar = add_scroll_bar

    def update(self, frame_no):
        if False:
            print('Hello World!')
        self._draw_label()
        height = self._h
        width = self._w - self._offset
        (colour, attr, background) = self._frame.palette['field']
        for i in range(height):
            self._frame.canvas.print_at(' ' * self.width, self._x + self._offset, self._y + i, colour, attr, background)
        if len(self._options) <= 0:
            return
        if self._add_scroll_bar:
            self._add_or_remove_scrollbar(width, height, 0)
        if self._scroll_bar:
            width -= 1
        y_offset = 0
        if self._centre:
            self._start_line = self._line - height // 2
        start_line = self._start_line
        if self._start_line < 0:
            y_offset = -self._start_line
            start_line = 0
        for (i, (text, _)) in enumerate(self._options):
            if start_line <= i < start_line + height - y_offset:
                (colour, attr, background) = self._pick_colours('field', i == self._line)
                if len(text) > width:
                    text = text[:width - 3] + '...'
                paint_text = _enforce_width(text, width, self._frame.canvas.unicode_aware)
                paint_text += ' ' * (width - self.string_len(str(paint_text)))
                self._frame.canvas.paint(str(paint_text), self._x + self._offset, self._y + y_offset + i - start_line, colour, attr, background, colour_map=paint_text.colour_map if hasattr(paint_text, 'colour_map') else None)
        if self._scroll_bar:
            self._scroll_bar.update()

    def _find_option(self, search_value):
        if False:
            return 10
        for (text, value) in self._options:
            if text.startswith(search_value):
                return value
        return None

    def _parse_option(self, option):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse a single option for ColouredText.\n\n        :param option: the option to parse\n        :returns: the option parsed and converted to ColouredText.\n        '
        try:
            return ColouredText(option.raw_text, self._parser)
        except AttributeError:
            return ColouredText(option, self._parser)

    @property
    def options(self):
        if False:
            i = 10
            return i + 15
        '\n        The list of options available for user selection\n\n        This is a list of tuples (<human readable string>, <internal value>).\n        '
        return self._options

    @options.setter
    def options(self, new_value):
        if False:
            i = 10
            return i + 15
        self._options = self._parse_options(new_value)
        self.value = self._value