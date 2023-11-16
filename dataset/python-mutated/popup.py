from __future__ import annotations
from typing import TYPE_CHECKING
from libqtile import configurable, pangocffi
if TYPE_CHECKING:
    from typing import Any
    from cairocffi import ImageSurface
    from libqtile.backend.base import Drawer
    from libqtile.core.manager import Qtile
    from libqtile.utils import ColorType

class Popup(configurable.Configurable):
    """
    This class can be used to create popup windows that display images and/or text.
    """
    defaults = [('opacity', 1.0, 'Opacity of notifications.'), ('foreground', '#ffffff', 'Colour of text.'), ('background', '#111111', 'Background colour.'), ('border', '#111111', 'Border colour.'), ('border_width', 0, 'Line width of drawn borders.'), ('font', 'sans', 'Font used in notifications.'), ('font_size', 14, 'Size of font.'), ('fontshadow', None, 'Colour for text shadows, or None for no shadows.'), ('horizontal_padding', 0, 'Padding at sides of text.'), ('vertical_padding', 0, 'Padding at top and bottom of text.'), ('text_alignment', 'left', 'Text alignment: left, center or right.'), ('wrap', True, 'Whether to wrap text.')]

    def __init__(self, qtile: Qtile, x: int=50, y: int=50, width: int=256, height: int=64, **config):
        if False:
            i = 10
            return i + 15
        configurable.Configurable.__init__(self, **config)
        self.add_defaults(Popup.defaults)
        self.qtile = qtile
        self.win: Any = qtile.core.create_internal(x, y, width, height)
        self.win.opacity = self.opacity
        self.win.process_button_click = self.process_button_click
        self.win.process_window_expose = self.draw
        self.drawer: Drawer = self.win.create_drawer(width, height)
        self.clear()
        self.layout = self.drawer.textlayout(text='', colour=self.foreground, font_family=self.font, font_size=self.font_size, font_shadow=self.fontshadow, wrap=self.wrap, markup=True)
        self.layout.layout.set_alignment(pangocffi.ALIGNMENTS[self.text_alignment])
        if self.border_width and self.border:
            self.win.paint_borders(self.border, self.border_width)
        self.x = self.win.x
        self.y = self.win.y

    def process_button_click(self, x, y, button) -> None:
        if False:
            print('Hello World!')
        if button == 1:
            self.hide()

    @property
    def width(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.win.width

    @width.setter
    def width(self, value: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.win.width = value
        self.drawer.width = value

    @property
    def height(self) -> int:
        if False:
            return 10
        return self.win.height

    @height.setter
    def height(self, value: int) -> None:
        if False:
            print('Hello World!')
        self.win.height = value
        self.drawer.height = value

    @property
    def text(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.layout.text

    @text.setter
    def text(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        self.layout.text = value

    @property
    def foreground(self) -> ColorType:
        if False:
            print('Hello World!')
        return self._foreground

    @foreground.setter
    def foreground(self, value: ColorType) -> None:
        if False:
            return 10
        self._foreground = value
        if hasattr(self, 'layout'):
            self.layout.colour = value

    def set_border(self, color: ColorType) -> None:
        if False:
            return 10
        self.win.paint_borders(color, self.border_width)

    def clear(self) -> None:
        if False:
            print('Hello World!')
        self.drawer.clear(self.background)

    def draw_text(self, x: int | None=None, y: int | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.layout.draw(x or self.horizontal_padding, y or self.vertical_padding)

    def draw(self) -> None:
        if False:
            i = 10
            return i + 15
        self.drawer.draw()

    def place(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.win.place(self.x, self.y, self.width, self.height, self.border_width, self.border, above=True)

    def unhide(self) -> None:
        if False:
            return 10
        self.win.unhide()

    def draw_image(self, image: ImageSurface, x: int, y: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Paint an image onto the window at point x, y. The image should be a surface e.g.\n        loaded from libqtile.images.Img.from_path.\n        '
        self.drawer.ctx.set_source_surface(image, x, y)
        self.drawer.ctx.paint()

    def hide(self) -> None:
        if False:
            i = 10
            return i + 15
        self.win.hide()

    def kill(self) -> None:
        if False:
            return 10
        self.win.kill()
        self.layout.finalize()
        self.drawer.finalize()