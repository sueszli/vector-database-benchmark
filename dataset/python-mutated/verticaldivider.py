"""This module implements a vertical division between widgets"""
from asciimatics.widgets.widget import Widget

class VerticalDivider(Widget):
    """
    A vertical divider for separating columns.

    This widget should be put into a column of its own in the Layout.
    """
    __slots__ = ['_required_height']

    def __init__(self, height=Widget.FILL_COLUMN):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param height: The required height for this divider.\n        '
        super().__init__(None, tab_stop=False)
        self._required_height = height

    def process_event(self, event):
        if False:
            return 10
        return event

    def update(self, frame_no):
        if False:
            return 10
        (color, attr, background) = self._frame.palette['borders']
        vert = '│' if self._frame.canvas.unicode_aware else '|'
        for i in range(self._h):
            self._frame.canvas.print_at(vert, self._x, self._y + i, color, attr, background)

    def reset(self):
        if False:
            return 10
        pass

    def required_height(self, offset, width):
        if False:
            i = 10
            return i + 15
        return self._required_height

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        '\n        The current value for this VerticalDivider.\n        '
        return self._value