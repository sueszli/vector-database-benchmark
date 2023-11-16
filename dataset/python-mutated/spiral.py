from __future__ import annotations
from typing import TYPE_CHECKING
from libqtile.command.base import expose_command
from libqtile.layout.base import _SimpleLayoutBase
from libqtile.log_utils import logger
if TYPE_CHECKING:
    from typing import Any, Self
    from libqtile.backend.base import Window
    from libqtile.group import _Group
    Rect = tuple[int, int, int, int]
GOLDEN_RATIO = 1.618

class Spiral(_SimpleLayoutBase):
    """
    A mathematical layout.

    Renders windows in a spiral form by splitting the screen based on a selected ratio.
    The direction of the split is changed every time in a defined order resulting in a
    spiral formation.

    The main window can be sized with ``lazy.layout.grow_main()`` and
    ``lazy.layout.shrink_main()``. All other windows are sized by
    ``lazy.layout.increase_ratio()`` and ``lazy.layout.decrease_ratio()``.

    NB if ``main_pane_ratio`` is not set then it will also be adjusted according to ``ratio``.
    However, as soon ``shrink_main()`` or ``grow_main()`` have been called once then the
    master pane will only change size following further calls to those methods.

    Users are able to choose the location of the main (i.e. largest) pane and the direction
    of the rotation.

    Some examples:

    ``main_pane="left", clockwise=True``

    ::

        ----------------------
        |1        |2         |
        |         |          |
        |         |          |
        |         |----------|
        |         |5 |6 |3   |
        |         |-----|    |
        |         |4    |    |
        ----------------------

    ``main_pane="top", clockwise=False``

    ::

        ----------------------
        |1                   |
        |                    |
        |                    |
        |--------------------|
        |2        |5    |4   |
        |         |----------|
        |         |3         |
        ----------------------

    """
    split_ratio: float
    defaults = [('border_focus', '#0000ff', 'Border colour(s) for the focused window.'), ('border_normal', '#000000', 'Border colour(s) for un-focused windows.'), ('border_width', 1, 'Border width.'), ('margin', 0, 'Margin of the layout (int or list of ints [N E S W])'), ('ratio', 1 / GOLDEN_RATIO, 'Ratio of the tiles'), ('main_pane_ratio', None, "Ratio for biggest window or 'None' to use same ratio for all windows."), ('ratio_increment', 0.1, 'Amount to increment per ratio increment'), ('main_pane', 'left', "Location of biggest window 'top', 'bottom', 'left', 'right'"), ('clockwise', True, 'Direction of spiral'), ('new_client_position', 'top', "Place new windows:  'after_current' - after the active window, 'before_current' - before the active window, 'top' - in the main pane, 'bottom '- at the bottom of the stack. NB windows that are added too low in the stack may be hidden if there is no remaining space in the spiral.")]

    def __init__(self, **config):
        if False:
            print('Hello World!')
        _SimpleLayoutBase.__init__(self, **config)
        self.add_defaults(Spiral.defaults)
        self.dirty = True
        self.layout_info = []
        self.last_size = None
        self.last_screen = None
        self.initial_ratio = self.ratio
        self.initial_main_pane_ratio = self.main_pane_ratio
        self.main_pane = self.main_pane.lower()
        if self.main_pane not in ['top', 'left', 'bottom', 'right']:
            logger.warning("Unknown main_pane location: %s. Defaulting to 'left'.", self.main_pane)
            self.main_pane = 'left'
        if self.clockwise:
            order = ['left', 'top', 'right', 'bottom', 'left', 'top', 'right']
        else:
            order = ['left', 'bottom', 'right', 'top', 'left', 'bottom', 'right']
        idx = order.index(self.main_pane)
        self.splits = order[idx:idx + 4]

    def clone(self, group: _Group) -> Self:
        if False:
            return 10
        return _SimpleLayoutBase.clone(self, group)

    def add_client(self, client: Window) -> None:
        if False:
            return 10
        self.dirty = True
        self.clients.add_client(client, client_position=self.new_client_position)

    def remove(self, w: Window) -> Window | None:
        if False:
            while True:
                i = 10
        self.dirty = True
        return _SimpleLayoutBase.remove(self, w)

    def configure(self, win, screen):
        if False:
            print('Hello World!')
        if not self.last_screen or self.last_screen != screen:
            self.last_screen = screen
            self.dirty = True
        if self.last_size and (not self.dirty):
            if screen.width != self.last_size[0] or screen.height != self.last_size[1]:
                self.dirty = True
        if self.dirty:
            self.layout_info = self.get_spiral(screen.x, screen.y, screen.width, screen.height)
            self.dirty = False
        try:
            idx = self.clients.index(win)
        except ValueError:
            win.hide()
            return
        try:
            (x, y, w, h) = self.layout_info[idx]
        except IndexError:
            win.hide()
            return
        if win.has_focus:
            bc = self.border_focus
        else:
            bc = self.border_normal
        ((x, y, w, h), margins) = self._fix_double_margins(x, y, w, h)
        win.place(x, y, w - self.border_width * 2, h - self.border_width * 2, self.border_width, bc, margin=margins)
        win.unhide()

    def split_left(self, rect: Rect) -> tuple[Rect, Rect]:
        if False:
            for i in range(10):
                print('nop')
        (rect_x, rect_y, rect_w, rect_h) = rect
        win_w = int(rect_w * self.split_ratio)
        win_h = rect_h
        win_x = rect_x
        win_y = rect_y
        rect_x = win_x + win_w
        rect_y = win_y
        rect_w = rect_w - win_w
        return ((win_x, win_y, win_w, win_h), (rect_x, rect_y, rect_w, rect_h))

    def split_right(self, rect: Rect) -> tuple[Rect, Rect]:
        if False:
            i = 10
            return i + 15
        (rect_x, rect_y, rect_w, rect_h) = rect
        win_w = int(rect_w * self.split_ratio)
        win_h = rect_h
        win_x = rect_x + (rect_w - win_w)
        win_y = rect_y
        rect_x = win_x - (rect_w - win_w)
        rect_y = win_y
        rect_w = rect_w - win_w
        return ((win_x, win_y, win_w, win_h), (rect_x, rect_y, rect_w, rect_h))

    def split_top(self, rect: Rect) -> tuple[Rect, Rect]:
        if False:
            print('Hello World!')
        (rect_x, rect_y, rect_w, rect_h) = rect
        win_w = rect_w
        win_h = int(rect_h * self.split_ratio)
        win_x = rect_x
        win_y = rect_y
        rect_x = win_x
        rect_y = win_y + win_h
        rect_h = rect_h - win_h
        return ((win_x, win_y, win_w, win_h), (rect_x, rect_y, rect_w, rect_h))

    def split_bottom(self, rect: Rect) -> tuple[Rect, Rect]:
        if False:
            return 10
        (rect_x, rect_y, rect_w, rect_h) = rect
        win_w = rect_w
        win_h = int(rect_h * self.split_ratio)
        win_x = rect_x
        win_y = rect_y + (rect_h - win_h)
        rect_x = win_x
        rect_y = win_y - (rect_h - win_h)
        rect_h = rect_h - win_h
        return ((win_x, win_y, win_w, win_h), (rect_x, rect_y, rect_w, rect_h))

    def _fix_double_margins(self, win_x: int, win_y: int, win_w: int, win_h: int) -> tuple[Rect, list[int]]:
        if False:
            for i in range(10):
                print('nop')
        'Prevent doubling up of margins by halving margins for internal margins.'
        if isinstance(self.margin, int):
            margins = [self.margin] * 4
        else:
            margins = self.margin
        if win_y - margins[0] > self.last_screen.y:
            win_y -= margins[0] // 2
            win_h += margins[0] // 2
        if win_x + win_w + margins[1] < self.last_screen.x + self.last_screen.width:
            win_w += margins[1] // 2
        if win_y + win_h + margins[2] < self.last_screen.y + self.last_screen.height:
            win_h += margins[2] // 2
        if win_x - margins[3] > self.last_screen.x:
            win_x -= margins[3] // 2
            win_w += margins[3] // 2
        return ((win_x, win_y, win_w, win_h), margins)

    def has_invalid_size(self, win: Rect) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if window would have an invalid size.\n\n        A window that would have negative height or width (after adjusting for margins and borders)\n        will return True.\n        '
        if isinstance(self.margin, int):
            margin = [self.margin] * 4
        else:
            margin = self.margin
        return any([win[2] <= margin[1] + margin[3] + 2 * self.border_width, win[3] <= margin[0] + margin[2] + 2 * self.border_width])

    def get_spiral(self, x, y, width, height) -> list[Rect]:
        if False:
            return 10
        '\n        Calculates positions of windows in the spiral.\n\n        Returns a list of tuples (x, y, w, h) for positioning windows.\n        '
        num_windows = len(self.clients)
        direction = 0
        spiral = []
        rect = (x, y, width, height)
        for c in range(num_windows):
            if c == 0 and self.main_pane_ratio is not None:
                self.split_ratio = self.main_pane_ratio
            else:
                self.split_ratio = self.ratio
            split = c < num_windows - 1
            if not split:
                spiral.append(rect)
                continue
            (win, new_rect) = getattr(self, f'split_{self.splits[direction]}')(rect)
            if self.has_invalid_size(win):
                spiral.append(rect)
                break
            spiral.append(win)
            direction = (direction + 1) % 4
            rect = new_rect
        return spiral

    def info(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        d = _SimpleLayoutBase.info(self)
        focused = self.clients.current_client
        d['ratio'] = self.ratio
        d['focused'] = focused.name if focused else None
        d['layout_info'] = self.layout_info
        d['main_pane'] = self.main_pane
        d['clockwise'] = self.clockwise
        return d

    @expose_command('up')
    def previous(self) -> None:
        if False:
            return 10
        _SimpleLayoutBase.previous(self)

    @expose_command('down')
    def next(self) -> None:
        if False:
            while True:
                i = 10
        _SimpleLayoutBase.next(self)

    def _set_ratio(self, prop: str, value: float | str):
        if False:
            print('Hello World!')
        if not isinstance(value, (float, int)):
            try:
                value = float(value)
            except ValueError:
                logger.error('Invalid ratio value: %s', value)
                return
        if not 0 <= value <= 1:
            logger.warning('Invalid value for %s: %s. Value must be between 0 and 1.', prop, value)
            return
        setattr(self, prop, value)
        self.group.layout_all()

    @expose_command()
    def shuffle_down(self):
        if False:
            return 10
        if self.clients:
            self.clients.rotate_down()
            self.group.layout_all()

    @expose_command()
    def shuffle_up(self):
        if False:
            while True:
                i = 10
        if self.clients:
            self.clients.rotate_up()
            self.group.layout_all()

    @expose_command()
    def decrease_ratio(self):
        if False:
            return 10
        'Decrease spiral ratio.'
        self._set_ratio('ratio', self.ratio - self.ratio_increment)

    @expose_command()
    def increase_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        'Increase spiral ratio.'
        self._set_ratio('ratio', self.ratio + self.ratio_increment)

    @expose_command()
    def shrink_main(self):
        if False:
            print('Hello World!')
        'Shrink the main window.'
        if self.main_pane_ratio is None:
            self.main_pane_ratio = self.ratio
        self._set_ratio('main_pane_ratio', self.main_pane_ratio - self.ratio_increment)

    @expose_command()
    def grow_main(self):
        if False:
            while True:
                i = 10
        'Grow the main window.'
        if self.main_pane_ratio is None:
            self.main_pane_ratio = self.ratio
        self._set_ratio('main_pane_ratio', self.main_pane_ratio + self.ratio_increment)

    @expose_command()
    def set_ratio(self, ratio: float | str):
        if False:
            i = 10
            return i + 15
        'Set the ratio for all windows.'
        self._set_ratio('ratio', ratio)

    @expose_command()
    def set_master_ratio(self, ratio: float | str):
        if False:
            while True:
                i = 10
        'Set the ratio for the main window.'
        self._set_ratio('main_pane_ratio', ratio)

    @expose_command()
    def reset(self):
        if False:
            while True:
                i = 10
        'Reset ratios to values set in config.'
        self.ratio = self.initial_ratio
        self.main_pane_ratio = self.initial_main_pane_ratio
        self.group.layout_all()