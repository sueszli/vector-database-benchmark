from enum import IntFlag
from typing import Iterable, NamedTuple, Sequence
from .fast_data_types import BORDERS_PROGRAM, add_borders_rect, get_options, init_borders_program, os_window_has_background_image
from .shaders import program_for
from .typing import LayoutType
from .window_list import WindowGroup, WindowList

class BorderColor(IntFlag):
    (default_bg, active, inactive, window_bg, bell, tab_bar_bg, tab_bar_margin_color, tab_bar_left_edge_color, tab_bar_right_edge_color) = range(9)

class Border(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int
    color: BorderColor

def vertical_edge(os_window_id: int, tab_id: int, color: int, width: int, top: int, bottom: int, left: int) -> None:
    if False:
        return 10
    if width > 0:
        add_borders_rect(os_window_id, tab_id, left, top, left + width, bottom, color)

def horizontal_edge(os_window_id: int, tab_id: int, color: int, height: int, left: int, right: int, top: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    if height > 0:
        add_borders_rect(os_window_id, tab_id, left, top, right, top + height, color)

def draw_edges(os_window_id: int, tab_id: int, colors: Sequence[int], wg: WindowGroup, borders: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    geometry = wg.geometry
    if geometry is None:
        return
    (pl, pt) = (wg.effective_padding('left'), wg.effective_padding('top'))
    (pr, pb) = (wg.effective_padding('right'), wg.effective_padding('bottom'))
    left = geometry.left - pl
    top = geometry.top - pt
    lr = geometry.right
    right = lr + pr
    bt = geometry.bottom
    bottom = bt + pb
    if borders:
        width = wg.effective_border()
        bt = bottom
        lr = right
        left -= width
        top -= width
        right += width
        bottom += width
        pl = pr = pb = pt = width
    horizontal_edge(os_window_id, tab_id, colors[1], pt, left, right, top)
    horizontal_edge(os_window_id, tab_id, colors[3], pb, left, right, bt)
    vertical_edge(os_window_id, tab_id, colors[0], pl, top, bottom, left)
    vertical_edge(os_window_id, tab_id, colors[2], pr, top, bottom, lr)

def load_borders_program() -> None:
    if False:
        i = 10
        return i + 15
    program_for('border').compile(BORDERS_PROGRAM)
    init_borders_program()

class Borders:

    def __init__(self, os_window_id: int, tab_id: int):
        if False:
            return 10
        self.os_window_id = os_window_id
        self.tab_id = tab_id

    def __call__(self, all_windows: WindowList, current_layout: LayoutType, tab_bar_rects: Iterable[Border], draw_window_borders: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        opts = get_options()
        draw_active_borders = opts.active_border_color is not None
        draw_minimal_borders = opts.draw_minimal_borders and max(opts.window_margin_width) < 1
        add_borders_rect(self.os_window_id, self.tab_id, 0, 0, 0, 0, BorderColor.default_bg)
        has_background_image = os_window_has_background_image(self.os_window_id)
        if not has_background_image or opts.background_tint > 0.0:
            for br in current_layout.blank_rects:
                add_borders_rect(self.os_window_id, self.tab_id, *br, BorderColor.default_bg)
            for tbr in tab_bar_rects:
                add_borders_rect(self.os_window_id, self.tab_id, *tbr)
        bw = 0
        groups = tuple(all_windows.iter_all_layoutable_groups(only_visible=True))
        if groups:
            bw = groups[0].effective_border()
        draw_borders = bw > 0 and draw_window_borders
        active_group = all_windows.active_group
        for (i, wg) in enumerate(groups):
            window_bg = wg.default_bg
            window_bg = window_bg << 8 | BorderColor.window_bg
            if draw_borders and (not draw_minimal_borders):
                if wg is active_group and draw_active_borders:
                    color = BorderColor.active
                else:
                    color = BorderColor.bell if wg.needs_attention else BorderColor.inactive
                draw_edges(self.os_window_id, self.tab_id, (color, color, color, color), wg, borders=True)
            if not has_background_image:
                colors = (window_bg, window_bg, window_bg, window_bg)
                draw_edges(self.os_window_id, self.tab_id, colors, wg)
        if draw_minimal_borders:
            for border_line in current_layout.get_minimal_borders(all_windows):
                (left, top, right, bottom) = border_line.edges
                add_borders_rect(self.os_window_id, self.tab_id, left, top, right, bottom, border_line.color)