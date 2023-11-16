"""This module defines a dropdown list widget"""
from asciimatics.event import KeyboardEvent, MouseEvent
from asciimatics.screen import Screen
from asciimatics.widgets.divider import Divider
from asciimatics.widgets.layout import Layout
from asciimatics.widgets.listbox import ListBox
from asciimatics.widgets.temppopup import _TempPopup
from asciimatics.widgets.text import Text
from asciimatics.widgets.utilities import _enforce_width
from asciimatics.widgets.widget import Widget

class _DropdownPopup(_TempPopup):
    """
    An internal Frame for selecting an item from a drop-down list..
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        '\n        :param parent: The widget that spawned this pop-up.\n        '
        location = parent.get_location()
        if parent.frame.screen.height - location[1] < 3:
            height = min(len(parent.options) + 4, location[1] + 2)
            start_line = location[1] - height + 2
            reverse = True
        else:
            start_line = location[1] - 1
            height = min(len(parent.options) + 4, parent.frame.screen.height - location[1] + 1)
            reverse = False
        if parent.fit:
            width = min(max(map(lambda x: len(x[0]), parent.options)) + 4, parent.width)
        else:
            width = parent.width
        super().__init__(parent.frame.screen, parent, location[0], start_line, width, height)
        layout = Layout([1], fill_frame=True)
        self.add_layout(layout)
        self._field = Text()
        self._field.disabled = True
        divider = Divider()
        divider.disabled = True
        self._list = ListBox(Widget.FILL_FRAME, [(f' {i[0]}', i[1]) for i in parent.options], add_scroll_bar=len(parent.options) > height - 4, on_select=self.close, on_change=self._link)
        layout.add_widget(self._list if reverse else self._field, 0)
        layout.add_widget(divider, 0)
        layout.add_widget(self._field if reverse else self._list, 0)
        self.fix()
        self._list.value = parent.value

    def _link(self):
        if False:
            i = 10
            return i + 15
        self._field.value = self._list.options[self._list._line][0]

    def _on_close(self, cancelled):
        if False:
            i = 10
            return i + 15
        if not cancelled:
            self._parent.value = self._list.value

class DropdownList(Widget):
    """
    This widget allows you to pick an item from a temporary pop-up list.
    """
    __slots__ = ['_on_change', '_child', '_options', '_line', '_fit']

    def __init__(self, options, label=None, name=None, on_change=None, fit=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        :param options: The options for each row in the widget.\n        :param label: An optional label for the widget.\n        :param name: The name for the widget.\n        :param on_change: Optional function to call when the selected time changes.\n        :param fit: Shrink width of dropdown to fit the width of options. Default False.\n\n        The `options` are a list of tuples, where the first value is the string to be displayed\n        to the user and the second is an interval value to identify the entry to the program.\n        For example:\n\n            options=[("First option", 1), ("Second option", 2)]\n\n        Also see the common keyword arguments in :py:obj:`.Widget`.\n        '
        super().__init__(name, **kwargs)
        self._label = label
        self._on_change = on_change
        self._child = None
        self._options = options
        self._line = 0 if len(options) > 0 else None
        self._value = options[self._line][1] if self._line is not None else None
        self._fit = fit

    @property
    def options(self):
        if False:
            return 10
        '\n        The set of allowed options for the drop-down list.\n        '
        return self._options

    @options.setter
    def options(self, new_value):
        if False:
            i = 10
            return i + 15
        self._options = new_value
        self.value = self._value

    @property
    def fit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to shrink to largest element width or not.\n        '
        return self._fit

    def update(self, frame_no):
        if False:
            for i in range(10):
                print('nop')
        self._draw_label()
        text = '' if self._line is None else self._options[self._line][0]
        (colour, attr, background) = self._pick_colours('field', selected=self._has_focus)
        if self._fit:
            width = min(max(map(lambda x: len(x[0]), self._options)) + 1, self.width - 3)
        else:
            width = self.width - 3
        self._frame.canvas.print_at(f'[ {_enforce_width(text, width, self._frame.canvas.unicode_aware):{width}}]', self._x + self._offset, self._y, colour, attr, background)

    def reset(self):
        if False:
            i = 10
            return i + 15
        pass

    def process_event(self, event):
        if False:
            return 10
        if event is not None:
            if isinstance(event, KeyboardEvent):
                if event.key_code in [Screen.ctrl('M'), Screen.ctrl('J'), ord(' ')]:
                    event = None
            elif isinstance(event, MouseEvent):
                if event.buttons != 0:
                    if self.is_mouse_over(event, include_label=False):
                        event = None
            if event is None:
                self._child = _DropdownPopup(self)
                self.frame.scene.add_effect(self._child)
        return event

    def required_height(self, offset, width):
        if False:
            i = 10
            return i + 15
        return 1

    @property
    def value(self):
        if False:
            return 10
        '\n        The current value for this DropdownList.\n        '
        return self._value

    @value.setter
    def value(self, new_value):
        if False:
            return 10
        old_value = self._value
        self._value = new_value
        for (i, [_, value]) in enumerate(self._options):
            if value == new_value:
                self._line = i
                break
        else:
            self._value = self._line = None
        if old_value != self._value and self._on_change:
            self._on_change()