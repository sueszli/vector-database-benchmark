"""This module implements the widget for radio buttons"""
from asciimatics.event import KeyboardEvent, MouseEvent
from asciimatics.screen import Screen
from asciimatics.widgets.widget import Widget

class RadioButtons(Widget):
    """
    A RadioButtons widget is used to ask for one of a list of values to be selected by the user.

    It consists of an optional label and then a list of selection bullets with field names.
    """
    __slots__ = ['_options', '_selection', '_start_column', '_on_change']

    def __init__(self, options, label=None, name=None, on_change=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param options: A list of (text, value) tuples for each radio button.\n        :param label: An optional label for the widget.\n        :param name: The internal name for the widget.\n        :param on_change: Optional function to call when text changes.\n\n        Also see the common keyword arguments in :py:obj:`.Widget`.\n        '
        super().__init__(name, **kwargs)
        self._options = options
        self._label = label
        self._selection = 0
        self._start_column = 0
        self._on_change = on_change

    def update(self, frame_no):
        if False:
            print('Hello World!')
        self._draw_label()
        check_char = '•' if self._frame.canvas.unicode_aware else 'X'
        for (i, (text, _)) in enumerate(self._options):
            (fg, attr, bg) = self._pick_colours('control', self._has_focus and i == self._selection)
            (fg2, attr2, bg2) = self._pick_colours('field', self._has_focus and i == self._selection)
            check = check_char if i == self._selection else ' '
            self._frame.canvas.print_at(f'({check}) ', self._x + self._offset, self._y + i, fg, attr, bg)
            self._frame.canvas.print_at(text, self._x + self._offset + 4, self._y + i, fg2, attr2, bg2)

    def reset(self):
        if False:
            print('Hello World!')
        pass

    def process_event(self, event):
        if False:
            print('Hello World!')
        if isinstance(event, KeyboardEvent):
            if event.key_code == Screen.KEY_UP:
                self._selection = max(0, self._selection - 1)
                self.value = self._options[self._selection][1]
            elif event.key_code == Screen.KEY_DOWN:
                self._selection = min(self._selection + 1, len(self._options) - 1)
                self.value = self._options[self._selection][1]
            else:
                return event
        elif isinstance(event, MouseEvent):
            if event.buttons != 0:
                if self.is_mouse_over(event, include_label=False):
                    self._selection = event.y - self._y
                    self.value = self._options[self._selection][1]
                    return None
            return event
        else:
            return event
        return None

    def required_height(self, offset, width):
        if False:
            print('Hello World!')
        return len(self._options)

    @property
    def value(self):
        if False:
            print('Hello World!')
        '\n        The current value for these RadioButtons.\n        '
        return self._options[self._selection][1]

    @value.setter
    def value(self, new_value):
        if False:
            i = 10
            return i + 15
        old_value = self._value
        for (i, (_, value)) in enumerate(self._options):
            if new_value == value:
                self._selection = i
                break
        else:
            self._selection = 0
        self._value = self._options[self._selection][1]
        if old_value != self._value and self._on_change:
            self._on_change()