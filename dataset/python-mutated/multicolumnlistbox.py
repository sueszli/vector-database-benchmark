"""This module implements the widget for a multiple column list box"""
from re import match as re_match
from itertools import zip_longest
from asciimatics.strings import ColouredText
from asciimatics.widgets.utilities import _enforce_width
from asciimatics.widgets.baselistbox import _BaseListBox

class MultiColumnListBox(_BaseListBox):
    """
    A MultiColumnListBox is a widget for displaying tabular data.

    It displays a list of related data in columns, from which the user can select a line.
    """

    def __init__(self, height, columns, options, titles=None, label=None, name=None, add_scroll_bar=False, parser=None, on_change=None, on_select=None, space_delimiter=' '):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param height: The required number of input lines for this ListBox.\n        :param columns: A list of widths and alignments for each column.\n        :param options: The options for each row in the widget.\n        :param titles: Optional list of titles for each column.  Must match the length of\n            `columns`.\n        :param label: An optional label for the widget.\n        :param name: The name for the ListBox.\n        :param add_scroll_bar: Whether to add optional scrollbar for large lists.\n        :param parser: Optional parser to colour options and titles text.\n        :param on_change: Optional function to call when selection changes.\n        :param on_select: Optional function to call when the user actually selects an entry from\n        :param space_delimiter: Optional parameter to define the delimiter between columns.\n            The default value is blank space.\n\n        The `columns` parameter is a list of integers or strings.  If it is an integer, this is\n        the absolute width of the column in characters.  If it is a string, it must be of the\n        format "[<align>]<width>[%]" where:\n\n        * <align> is the alignment string ("<" = left, ">" = right, "^" = centre)\n        * <width> is the width in characters\n        * % is an optional qualifier that says the number is a percentage of the width of the\n          widget.\n\n        Column widths need to encompass any space required between columns, so for example, if\n        your column is 5 characters, allow 6 for an extra space at the end.  It is not possible\n        to do this when you have a right-justified column next to a left-justified column, so\n        this widget will automatically space them for you.\n\n        An integer value of 0 is interpreted to be use whatever space is left available after the\n        rest of the columns have been calculated.  There must be only one of these columns.\n\n        The number of columns is for this widget is determined from the number of entries in the\n        `columns` parameter.  The `options` list is then a list of tuples of the form\n        ([val1, val2, ... , valn], index).  For example, this data provides 2 rows for a 3 column\n        widget:\n\n            options=[(["One", "row", "here"], 1), (["Second", "row", "here"], 2)]\n\n        The options list may be None and then can be set later using the `options` property on\n        this widget.\n        '
        if titles is not None and parser is not None:
            titles = [ColouredText(x, parser) for x in titles]
        super().__init__(height, options, titles=titles, label=label, name=name, parser=parser, on_change=on_change, on_select=on_select)
        self._columns = []
        self._align = []
        self._spacing = []
        self._add_scroll_bar = add_scroll_bar
        self._space_delimiter = space_delimiter
        for (i, column) in enumerate(columns):
            if isinstance(column, int):
                self._columns.append(column)
                self._align.append('<')
            else:
                match = re_match('([<>^]?)(\\d+)([%]?)', column)
                self._columns.append(float(match.group(2)) / 100 if match.group(3) else int(match.group(2)))
                self._align.append(match.group(1) if match.group(1) else '<')
            if space_delimiter == ' ':
                self._spacing.append(1 if i > 0 and self._align[i] == '<' and (self._align[i - 1] == '>') else 0)
            else:
                self._spacing.append(1 if i > 0 else 0)

    def _get_width(self, width, max_width):
        if False:
            while True:
                i = 10
        '\n        Helper function to figure out the actual column width from the various options.\n\n        :param width: The size of column requested\n        :param max_width: The maximum width allowed for this widget.\n        :return: the integer width of the column in characters\n        '
        if isinstance(width, float):
            return int(max_width * width)
        if width == 0:
            width = max_width - sum(self._spacing) - sum((self._get_width(x, max_width) for x in self._columns if x != 0))
        return width

    def _print_cell(self, space, text, align, width, x, y, foreground, attr, background):
        if False:
            for i in range(10):
                print('nop')
        if space:
            self._frame.canvas.print_at(self._space_delimiter * space, x, y, foreground, attr, background)
        paint_text = _enforce_width(text, width, self._frame.canvas.unicode_aware)
        text_size = self.string_len(str(paint_text))
        if text_size < width:
            buffer_1 = buffer_2 = ''
            if align == '<':
                buffer_2 = ' ' * (width - text_size)
            elif align == '>':
                buffer_1 = ' ' * (width - text_size)
            elif align == '^':
                start_len = int((width - text_size) / 2)
                buffer_1 = ' ' * start_len
                buffer_2 = ' ' * (width - text_size - start_len)
            paint_text = paint_text.join([buffer_1, buffer_2])
        self._frame.canvas.paint(str(paint_text), x + space, y, foreground, attr, background, colour_map=paint_text.colour_map if hasattr(paint_text, 'colour_map') else None)

    def update(self, frame_no):
        if False:
            i = 10
            return i + 15
        self._draw_label()
        height = self._h
        width = self._w
        delta_y = 0
        (colour, attr, background) = self._frame.palette['field']
        for i in range(height):
            self._frame.canvas.print_at(' ' * width, self._x + self._offset, self._y + i + delta_y, colour, attr, background)
        if self._titles:
            delta_y += 1
            height -= 1
        if self._add_scroll_bar:
            self._add_or_remove_scrollbar(width, height, delta_y)
        if self._scroll_bar:
            width -= 1
        if self._titles:
            row_dx = 0
            (colour, attr, background) = self._frame.palette['title']
            for (i, [title, align, space]) in enumerate(zip(self._titles, self._align, self._spacing)):
                cell_width = self._get_width(self._columns[i], width)
                self._print_cell(space, title, align, cell_width, self._x + self._offset + row_dx, self._y, colour, attr, background)
                row_dx += cell_width + space
        if len(self._options) <= 0:
            return
        self._start_line = max(0, self._line - height + 1, min(self._start_line, self._line))
        for (i, [row, _]) in enumerate(self._options):
            if self._start_line <= i < self._start_line + height:
                (colour, attr, background) = self._pick_colours('field', i == self._line)
                row_dx = 0
                for (text, cell_width, align, space) in zip_longest(row, self._columns, self._align, self._spacing, fillvalue=''):
                    if cell_width == '':
                        break
                    cell_width = self._get_width(cell_width, width)
                    if len(text) > cell_width:
                        text = text[:cell_width - 3] + '...'
                    self._print_cell(space, text, align, cell_width, self._x + self._offset + row_dx, self._y + i + delta_y - self._start_line, colour, attr, background)
                    row_dx += cell_width + space
        if self._scroll_bar:
            self._scroll_bar.update()

    def _find_option(self, search_value):
        if False:
            for i in range(10):
                print('nop')
        for (row, value) in self._options:
            if row[0].startswith(search_value):
                return value
        return None

    def _parse_option(self, option):
        if False:
            while True:
                i = 10
        '\n        Parse a single option for ColouredText.\n\n        :param option: the option to parse\n        :returns: the option parsed and converted to ColouredText.\n        '
        option_items = []
        for item in option:
            try:
                value = ColouredText(item.raw_text, self._parser)
            except AttributeError:
                value = ColouredText(item, self._parser)
            option_items.append(value)
        return option_items

    @property
    def options(self):
        if False:
            i = 10
            return i + 15
        '\n        The list of options available for user selection\n\n        This is a list of tuples ([<col 1 string>, ..., <col n string>], <internal value>).\n        '
        return self._options

    @options.setter
    def options(self, new_value):
        if False:
            for i in range(10):
                print('nop')
        self._options = self._parse_options(new_value)
        self.value = self._value