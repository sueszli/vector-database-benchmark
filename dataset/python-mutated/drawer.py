from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING
import cairocffi
import xcffib.xproto
from libqtile import utils
from libqtile.backend.base import drawer
if TYPE_CHECKING:
    from libqtile.backend.base import Internal
    from libqtile.core.manager import Qtile

class Drawer(drawer.Drawer):
    """A helper class for drawing to Internal windows.

    The underlying surface here is an XCBSurface backed by a pixmap. We draw to the
    pixmap starting at offset 0, 0, and when the time comes to display to the window (on
    draw()), we copy the appropriate portion of the pixmap onto the window. In the event
    that our drawing area is resized, we invalidate the underlying surface and pixmap
    and recreate them when we need them again with the new geometry.
    """

    def __init__(self, qtile: Qtile, win: Internal, width: int, height: int):
        if False:
            return 10
        drawer.Drawer.__init__(self, qtile, win, width, height)
        self._xcb_surface = None
        self._gc = None
        (self._depth, self._visual) = qtile.core.conn.default_screen._get_depth_and_visual(win._depth)
        self._check_xcb()

    def finalize(self):
        if False:
            return 10
        self._free_xcb_surface()
        self._free_pixmap()
        self._free_gc()
        drawer.Drawer.finalize(self)

    @property
    def width(self):
        if False:
            return 10
        return self._width

    @width.setter
    def width(self, width):
        if False:
            while True:
                i = 10
        if width > self._width:
            self._free_xcb_surface()
            self._free_pixmap()
        self._width = width

    @property
    def height(self):
        if False:
            while True:
                i = 10
        return self._height

    @height.setter
    def height(self, height):
        if False:
            return 10
        if height > self._height:
            self._free_xcb_surface()
            self._free_pixmap()
        self._height = height

    @property
    def pixmap(self):
        if False:
            for i in range(10):
                print('nop')
        if self._pixmap is None:
            self.draw()
        return self._pixmap

    def _create_gc(self):
        if False:
            return 10
        gc = self.qtile.core.conn.conn.generate_id()
        self.qtile.core.conn.conn.core.CreateGC(gc, self._win.wid, xcffib.xproto.GC.Foreground | xcffib.xproto.GC.Background, [self.qtile.core.conn.default_screen.black_pixel, self.qtile.core.conn.default_screen.white_pixel])
        return gc

    def _free_gc(self):
        if False:
            return 10
        if self._gc is not None:
            with contextlib.suppress(xcffib.ConnectionException):
                self.qtile.core.conn.conn.core.FreeGC(self._gc)
            self._gc = None

    def _create_xcb_surface(self):
        if False:
            for i in range(10):
                print('nop')
        surface = cairocffi.XCBSurface(self.qtile.core.conn.conn, self._pixmap, self._visual, self.width, self.height)
        return surface

    def _free_xcb_surface(self):
        if False:
            for i in range(10):
                print('nop')
        if self._xcb_surface is not None:
            self._xcb_surface.finish()
            self._xcb_surface = None

    def _create_pixmap(self):
        if False:
            return 10
        pixmap = self.qtile.core.conn.conn.generate_id()
        self.qtile.core.conn.conn.core.CreatePixmap(self._depth, pixmap, self._win.wid, self.width, self.height)
        return pixmap

    def _free_pixmap(self):
        if False:
            return 10
        if self._pixmap is not None:
            with contextlib.suppress(xcffib.ConnectionException):
                self.qtile.core.conn.conn.core.FreePixmap(self._pixmap)
            self._pixmap = None

    def _check_xcb(self):
        if False:
            i = 10
            return i + 15
        if self._xcb_surface is None:
            self._pixmap = self._create_pixmap()
            self._xcb_surface = self._create_xcb_surface()

    def _paint(self):
        if False:
            print('Hello World!')
        if self.needs_update:
            ctx = cairocffi.Context(self._xcb_surface)
            ctx.set_source_surface(self.surface, 0, 0)
            ctx.paint()
            self.previous_rect = self.current_rect

    def _draw(self, offsetx: int=0, offsety: int=0, width: int | None=None, height: int | None=None, src_x: int=0, src_y: int=0):
        if False:
            return 10
        self.current_rect = (offsetx, offsety, width, height)
        if self._gc is None:
            self._gc = self._create_gc()
        self._check_xcb()
        self._paint()
        self.qtile.core.conn.conn.core.CopyArea(self._pixmap, self._win.wid, self._gc, src_x, src_y, offsetx, offsety, self.width if width is None else width, self.height if height is None else height)

    def _find_root_visual(self):
        if False:
            return 10
        for i in self.qtile.core.conn.default_screen.allowed_depths:
            for v in i.visuals:
                if v.visual_id == self.qtile.core.conn.default_screen.root_visual:
                    return v

    def set_source_rgb(self, colour, ctx=None):
        if False:
            while True:
                i = 10
        if utils.has_transparency(colour) and self._depth != 32:
            colour = utils.remove_transparency(colour)
        drawer.Drawer.set_source_rgb(self, colour, ctx)

    def clear_rect(self, x=0, y=0, width=0, height=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Erases the background area specified by parameters. By default,\n        the whole Drawer is cleared.\n\n        The ability to clear a smaller area may be useful when you want to\n        erase a smaller area of the drawer (e.g. drawing widget decorations).\n        '
        if width <= 0:
            width = self.width
        if height <= 0:
            height = self.height
        with cairocffi.Context(self._xcb_surface) as ctx:
            ctx.set_operator(cairocffi.OPERATOR_CLEAR)
            ctx.rectangle(x, y, width, height)
            ctx.fill()