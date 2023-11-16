from __future__ import annotations
import typing
from wlroots import xwayland
from wlroots.wlr_types import SceneTree
from libqtile import hook
from libqtile.backend import base
from libqtile.backend.base import FloatStates
from libqtile.backend.wayland.window import Static, Window
from libqtile.command.base import expose_command
from libqtile.log_utils import logger
if typing.TYPE_CHECKING:
    from typing import Any
    import wlroots.wlr_types.foreign_toplevel_management_v1 as ftm
    from pywayland.server import Listener
    from wlroots.xwayland import SurfaceConfigureEvent
    from libqtile.backend.wayland.core import Core
    from libqtile.core.manager import Qtile
    from libqtile.utils import ColorsType

class XWindow(Window[xwayland.Surface]):
    """An X11 client connecting via XWayland."""

    def __init__(self, core: Core, qtile: Qtile, surface: xwayland.Surface):
        if False:
            for i in range(10):
                print('nop')
        Window.__init__(self, core, qtile, surface)
        self._wm_class = self.surface.wm_class
        self.tree: SceneTree | None = None
        if (title := surface.title):
            self.name = title
        self.add_listener(surface.map_event, self._on_map)
        self.add_listener(surface.unmap_event, self._on_unmap)
        self.add_listener(surface.request_activate_event, self._on_request_activate)
        self.add_listener(surface.request_configure_event, self._on_request_configure)
        self.add_listener(surface.destroy_event, self._on_destroy)

    def _on_commit(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        if self.floating:
            state = self.surface.surface.current
            if state.width != self._width or state.height != self._height:
                self.place(self.x, self.y, state.width, state.height, self.borderwidth, self.bordercolor)

    def _on_request_activate(self, _listener: Listener, event: SurfaceConfigureEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Signal: xwindow request_activate')
        self.surface.activate(True)

    def _on_request_configure(self, _listener: Listener, event: SurfaceConfigureEvent) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xwindow request_configure')
        if self.floating:
            self.place(event.x, event.y, event.width, event.height, self.borderwidth, self.bordercolor)
        else:
            self.surface.configure(event.x, event.y, event.width, event.height)
            self.place(self.x, self.y, self.width, self.height, self.borderwidth, self.bordercolor)

    def _on_unmap(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xwindow unmap')
        self.hide()
        if self not in self.core.pending_windows:
            self.finalize_listeners()
            if self.group and self not in self.group.windows:
                self.group = None
            self.qtile.unmanage(self.wid)
            self.core.pending_windows.add(self)
            self._wid = -1
            self.add_listener(self.surface.map_event, self._on_map)
            self.add_listener(self.surface.unmap_event, self._on_unmap)
            self.add_listener(self.surface.request_configure_event, self._on_request_configure)
            self.add_listener(self.surface.destroy_event, self._on_destroy)
        if self.ftm_handle:
            self.ftm_handle.destroy()
            self.ftm_handle = None
        self.core.remove_pointer_constraints(self)

    def _on_request_fullscreen(self, _listener: Listener, _data: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Signal: xwindow request_fullscreen')
        if self.qtile.config.auto_fullscreen:
            self.fullscreen = not self.fullscreen

    def _on_set_title(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xwindow set_title')
        title = self.surface.title
        if title and title != self.name:
            self.name = title
            if self.ftm_handle:
                self.ftm_handle.set_title(title)
            hook.fire('client_name_updated', self)

    def _on_set_class(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xwindow set_class')
        self._wm_class = self.surface.wm_class
        if self.ftm_handle:
            self.ftm_handle.set_app_id(self._wm_class or '')

    def hide(self) -> None:
        if False:
            while True:
                i = 10
        super().hide()
        if self.tree:
            self.tree.node.destroy()
            self.tree = None
            self.finalize_listener(self.surface.surface.commit_event)

    def unhide(self) -> None:
        if False:
            print('Hello World!')
        if self not in self.core.pending_windows:
            if self.group and self.group.screen:
                self.add_listener(self.surface.surface.commit_event, self._on_commit)
                if not self.tree:
                    self.tree = SceneTree.subsurface_tree_create(self.container, self.surface.surface)
                    self.tree.node.set_position(self.borderwidth, self.borderwidth)
                self.container.node.set_enabled(enabled=True)
                self.surface.restack(None, 0)
                return
        self.core.pending_windows.remove(self)
        self._wid = self.core.new_wid()
        logger.debug('Managing new XWayland window with window ID: %s', self._wid)
        surface = self.surface
        self.tree = SceneTree.subsurface_tree_create(self.container, surface.surface)
        if surface.override_redirect:
            self.static(None, surface.x, surface.y, surface.width, surface.height)
            win = self.qtile.windows_map[self._wid]
            assert isinstance(win, XStatic)
            self.core.focus_window(win)
            win.bring_to_front()
            return
        surface.data = self.data_handle
        self.add_listener(surface.surface.commit_event, self._on_commit)
        self.add_listener(surface.request_fullscreen_event, self._on_request_fullscreen)
        self.add_listener(surface.set_title_event, self._on_set_title)
        self.add_listener(surface.set_class_event, self._on_set_class)
        if surface.width > 1:
            self._width = self._float_width = surface.width
        if surface.height > 1:
            self._height = self._float_height = surface.height
        handle = self.ftm_handle = self.core.foreign_toplevel_manager_v1.create_handle()
        self.add_listener(handle.request_maximize_event, self._on_foreign_request_maximize)
        self.add_listener(handle.request_minimize_event, self._on_foreign_request_minimize)
        self.add_listener(handle.request_activate_event, self._on_foreign_request_activate)
        self.add_listener(handle.request_fullscreen_event, self._on_foreign_request_fullscreen)
        self.add_listener(handle.request_close_event, self._on_foreign_request_close)
        if (title := surface.title):
            self.name = title
            handle.set_title(title)
        self._wm_class = surface.wm_class
        handle.set_app_id(self._wm_class or '')
        self.qtile.manage(self)
        if self.group and self.group.screen:
            self.core.focus_window(self)

    @expose_command()
    def kill(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.surface.close()

    def has_fixed_size(self) -> bool:
        if False:
            while True:
                i = 10
        hints = self.surface.size_hints
        return bool(hints and 0 < hints.min_width == hints.max_width and (0 < hints.min_height == hints.max_height))

    def is_transient_for(self) -> base.WindowType | None:
        if False:
            return 10
        'What window is this window a transient window for?'
        parent = self.surface.parent
        if parent:
            for win in self.qtile.windows_map.values():
                if isinstance(win, XWindow) and win.surface == parent:
                    return win
        return None

    def get_pid(self) -> int:
        if False:
            return 10
        return self.surface.pid

    def get_wm_type(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        wm_type = self.surface.window_type
        if wm_type:
            return self.core.xwayland_atoms[wm_type[0]]
        return None

    def get_wm_role(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return self.surface.role

    def _update_fullscreen(self, do_full: bool) -> None:
        if False:
            while True:
                i = 10
        if do_full != (self._float_state == FloatStates.FULLSCREEN):
            self.surface.set_fullscreen(do_full)
            if self.ftm_handle:
                self.ftm_handle.set_fullscreen(do_full)

    def place(self, x: int, y: int, width: int, height: int, borderwidth: int, bordercolor: ColorsType | None, above: bool=False, margin: int | list[int] | None=None, respect_hints: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        if margin is not None:
            if isinstance(margin, int):
                margin = [margin] * 4
            x += margin[3]
            y += margin[0]
            width -= margin[1] + margin[3]
            height -= margin[0] + margin[2]
        if respect_hints:
            hints = self.surface.size_hints
            if hints:
                width = max(width, hints.min_width)
                height = max(height, hints.min_height)
                if hints.max_width > 0:
                    width = min(width, hints.max_width)
                if hints.max_height > 0:
                    height = min(height, hints.max_height)
        if self.group is not None and self.group.screen is not None:
            self.float_x = x - self.group.screen.x
            self.float_y = y - self.group.screen.y
        self.x = x
        self.y = y
        self._width = width
        self._height = height
        self.container.node.set_position(x, y)
        self.surface.configure(x, y, width, height)
        self.paint_borders(bordercolor, borderwidth)
        if above:
            self.bring_to_front()

    @expose_command()
    def bring_to_front(self) -> None:
        if False:
            while True:
                i = 10
        self.surface.restack(None, 0)
        self.container.node.raise_to_top()

    @expose_command()
    def static(self, screen: int | None=None, x: int | None=None, y: int | None=None, width: int | None=None, height: int | None=None) -> None:
        if False:
            while True:
                i = 10
        Window.static(self, screen, x, y, width, height)
        hook.fire('client_managed', self.qtile.windows_map[self._wid])

    def _to_static(self, x: int | None, y: int | None, width: int | None, height: int | None) -> XStatic:
        if False:
            i = 10
            return i + 15
        return XStatic(self.core, self.qtile, self, self._idle_inhibitors_count, x, y, width, height)

class ConfigWindow:
    """The XCB_CONFIG_WINDOW_* constants.

    Reproduced here to remove a dependency on xcffib.
    """
    X = 1
    Y = 2
    Width = 4
    Height = 8

class XStatic(Static[xwayland.Surface]):
    """A static window belonging to the XWayland shell."""
    surface: xwayland.Surface

    def __init__(self, core: Core, qtile: Qtile, win: XWindow, idle_inhibitor_count: int, x: int | None, y: int | None, width: int | None, height: int | None):
        if False:
            print('Hello World!')
        surface = win.surface
        Static.__init__(self, core, qtile, surface, win.wid, idle_inhibitor_count=idle_inhibitor_count)
        self._wm_class = surface.wm_class
        self._conf_x = x
        self._conf_y = y
        self._conf_width = width
        self._conf_height = height
        self.add_listener(surface.map_event, self._on_map)
        self.add_listener(surface.unmap_event, self._on_unmap)
        self.add_listener(surface.destroy_event, self._on_destroy)
        self.add_listener(surface.request_configure_event, self._on_request_configure)
        self.add_listener(surface.set_title_event, self._on_set_title)
        self.add_listener(surface.set_class_event, self._on_set_class)
        if surface.override_redirect:
            self.add_listener(surface.set_geometry_event, self._on_set_geometry)
        self.ftm_handle: ftm.ForeignToplevelHandleV1 | None = None
        self.container = win.container
        self.container.node.data = self.data_handle
        self.tree = win.tree

    def _on_unmap(self, _listener: Listener, _data: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Signal: xstatic unmap')
        self._on_destroy(None, None)
        win = XWindow(self.core, self.qtile, self.surface)
        self.core.pending_windows.add(win)

    def _on_request_configure(self, _listener: Listener, event: SurfaceConfigureEvent) -> None:
        if False:
            i = 10
            return i + 15
        logger.debug('Signal: xstatic request_configure')
        cw = ConfigWindow
        if self._conf_x is None and event.mask & cw.X:
            self.x = event.x
        if self._conf_y is None and event.mask & cw.Y:
            self.y = event.y
        if self._conf_width is None and event.mask & cw.Width:
            self.width = event.width
        if self._conf_height is None and event.mask & cw.Height:
            self.height = event.height
        self.place(self.x, self.y, self.width, self.height, self.borderwidth, self.bordercolor)

    @expose_command()
    def kill(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.surface.close()

    def hide(self) -> None:
        if False:
            i = 10
            return i + 15
        super().hide()
        self.container.node.set_enabled(enabled=False)

    def unhide(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self not in self.core.pending_windows:
            if not self.tree:
                self.tree = SceneTree.subsurface_tree_create(self.container, self.surface.surface)
                self.tree.node.set_position(self.borderwidth, self.borderwidth)
            self.container.node.set_enabled(enabled=True)
            self.bring_to_front()
            return

    def place(self, x: int, y: int, width: int, height: int, borderwidth: int, bordercolor: ColorsType | None, above: bool=False, margin: int | list[int] | None=None, respect_hints: bool=False) -> None:
        if False:
            return 10
        self.x = x
        self.y = y
        self._width = width
        self._height = height
        self.surface.configure(x, y, self._width, self._height)
        self.container.node.set_position(x, y)

    def _on_set_title(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xstatic set_title')
        title = self.surface.title
        if title and title != self.name:
            self.name = title
            if self.ftm_handle:
                self.ftm_handle.set_title(title)
            hook.fire('client_name_updated', self)

    def _on_set_class(self, _listener: Listener, _data: Any) -> None:
        if False:
            print('Hello World!')
        logger.debug('Signal: xstatic set_class')
        self._wm_class = self.surface.wm_class
        if self.ftm_handle:
            self.ftm_handle.set_app_id(self._wm_class or '')

    def _on_set_geometry(self, _listener: Listener, _data: Any) -> None:
        if False:
            i = 10
            return i + 15
        logger.debug('Signal: xstatic set_geometry')
        if self.surface.x != self.x or self.surface.y != self.y:
            self.place(self.surface.x, self.surface.y, self.surface.width, self.surface.height, 0, None)

    @expose_command()
    def bring_to_front(self) -> None:
        if False:
            print('Hello World!')
        self.surface.restack(None, 0)
        self.container.node.raise_to_top()