"""
`ScrollView` is a base class for [line api](/guide/widgets#line-api) widgets.
"""
from __future__ import annotations
from rich.console import RenderableType
from ._animator import EasingFunction
from ._types import CallbackType
from .containers import ScrollableContainer
from .geometry import Region, Size

class ScrollView(ScrollableContainer):
    """
    A base class for a Widget that handles its own scrolling (i.e. doesn't rely
    on the compositor to render children).
    """
    DEFAULT_CSS = '\n    ScrollView {\n        overflow-y: auto;\n        overflow-x: auto;\n    }\n    '

    @property
    def is_scrollable(self) -> bool:
        if False:
            while True:
                i = 10
        'Always scrollable.'
        return True

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        if False:
            while True:
                i = 10
        if self.show_horizontal_scrollbar and round(old_value) != round(new_value):
            self.horizontal_scrollbar.position = round(new_value)
            self.refresh()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        if False:
            return 10
        if self.show_vertical_scrollbar and round(old_value) != round(new_value):
            self.vertical_scrollbar.position = round(new_value)
            self.refresh()

    def on_mount(self):
        if False:
            while True:
                i = 10
        self._refresh_scrollbars()

    def get_content_width(self, container: Size, viewport: Size) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Gets the width of the content area.\n\n        Args:\n            container: Size of the container (immediate parent) widget.\n            viewport: Size of the viewport.\n\n        Returns:\n            The optimal width of the content.\n        '
        return self.virtual_size.width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        if False:
            return 10
        'Gets the height (number of lines) in the content area.\n\n        Args:\n            container: Size of the container (immediate parent) widget.\n            viewport: Size of the viewport.\n            width: Width of renderable.\n\n        Returns:\n            The height of the content.\n        '
        return self.virtual_size.height

    def _size_updated(self, size: Size, virtual_size: Size, container_size: Size, layout: bool=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Called when size is updated.\n\n        Args:\n            size: New size.\n            virtual_size: New virtual size.\n            container_size: New container size.\n            layout: Perform layout if required.\n\n        Returns:\n            True if anything changed, or False if nothing changed.\n        '
        if self._size != size or self._container_size != container_size:
            self.refresh()
        if self._size != size or virtual_size != self.virtual_size or container_size != self.container_size:
            self._size = size
            virtual_size = self.virtual_size
            self._container_size = size - self.styles.gutter.totals
            self._scroll_update(virtual_size)
            return True
        else:
            return False

    def render(self) -> RenderableType:
        if False:
            for i in range(10):
                print('nop')
        'Render the scrollable region (if `render_lines` is not implemented).\n\n        Returns:\n            Renderable object.\n        '
        from rich.panel import Panel
        return Panel(f'{self.scroll_offset} {self.show_vertical_scrollbar}')

    def scroll_to(self, x: float | None=None, y: float | None=None, *, animate: bool=True, speed: float | None=None, duration: float | None=None, easing: EasingFunction | str | None=None, force: bool=False, on_complete: CallbackType | None=None) -> None:
        if False:
            return 10
        'Scroll to a given (absolute) coordinate, optionally animating.\n\n        Args:\n            x: X coordinate (column) to scroll to, or `None` for no change.\n            y: Y coordinate (row) to scroll to, or `None` for no change.\n            animate: Animate to new scroll position.\n            speed: Speed of scroll if `animate` is `True`; or `None` to use `duration`.\n            duration: Duration of animation, if `animate` is `True` and `speed` is `None`.\n            easing: An easing method for the scrolling animation.\n            force: Force scrolling even when prohibited by overflow styling.\n            on_complete: A callable to invoke when the animation is finished.\n        '
        self._scroll_to(x, y, animate=animate, speed=speed, duration=duration, easing=easing, force=force, on_complete=on_complete)

    def refresh_lines(self, y_start: int, line_count: int=1) -> None:
        if False:
            while True:
                i = 10
        'Refresh one or more lines.\n\n        Args:\n            y_start: First line to refresh.\n            line_count: Total number of lines to refresh.\n        '
        width = self.size.width
        (scroll_x, scroll_y) = self.scroll_offset
        refresh_region = Region(scroll_x, y_start - scroll_y, width, line_count)
        self.refresh(refresh_region)