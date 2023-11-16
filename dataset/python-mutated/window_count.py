from __future__ import annotations
from typing import Any
from libqtile import bar, hook
from libqtile.command.base import expose_command
from libqtile.widget import base

class WindowCount(base._TextBox):
    """
    A simple widget to display the number of windows in the
    current group of the screen on which the widget is.
    """
    defaults: list[tuple[str, Any, str]] = [('font', 'sans', 'Text font'), ('fontsize', None, 'Font pixel size. Calculated if None.'), ('fontshadow', None, 'font shadow color, default is None(no shadow)'), ('padding', None, 'Padding left and right. Calculated if None.'), ('foreground', '#ffffff', 'Foreground colour.'), ('text_format', '{num}', 'Format for message'), ('show_zero', False, 'Show window count when no windows')]

    def __init__(self, width=bar.CALCULATED, **config):
        if False:
            return 10
        base._TextBox.__init__(self, width=width, **config)
        self.add_defaults(WindowCount.defaults)
        self._count = 0

    def _configure(self, qtile, bar):
        if False:
            while True:
                i = 10
        base._TextBox._configure(self, qtile, bar)
        self._setup_hooks()
        self._wincount()

    def _setup_hooks(self):
        if False:
            while True:
                i = 10
        hook.subscribe.client_killed(self._win_killed)
        hook.subscribe.client_managed(self._wincount)
        hook.subscribe.current_screen_change(self._wincount)
        hook.subscribe.setgroup(self._wincount)

    def _wincount(self, *args):
        if False:
            print('Hello World!')
        try:
            self._count = len(self.bar.screen.group.windows)
        except AttributeError:
            self._count = 0
        self.update(self.text_format.format(num=self._count))

    def _win_killed(self, window):
        if False:
            while True:
                i = 10
        try:
            self._count = len(self.bar.screen.group.windows)
        except AttributeError:
            self._count = 0
        self.update(self.text_format.format(num=self._count))

    def calculate_length(self):
        if False:
            while True:
                i = 10
        if self.text and (self._count or self.show_zero):
            return min(self.layout.width, self.bar.width) + self.actual_padding * 2
        else:
            return 0

    @expose_command()
    def get(self):
        if False:
            i = 10
            return i + 15
        'Retrieve the current text.'
        return self.text