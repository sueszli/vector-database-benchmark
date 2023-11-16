"""
Abstract base classes define the primitives for Tools.
These tools are used by `matplotlib.backend_managers.ToolManager`

:class:`ToolBase`
    Simple stateless tool

:class:`ToolToggleBase`
    Tool that has two states, only one Toggle tool can be
    active at any given time for the same
    `matplotlib.backend_managers.ToolManager`
"""
import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook

class Cursors(enum.IntEnum):
    """Backend-independent cursor types."""
    POINTER = enum.auto()
    HAND = enum.auto()
    SELECT_REGION = enum.auto()
    MOVE = enum.auto()
    WAIT = enum.auto()
    RESIZE_HORIZONTAL = enum.auto()
    RESIZE_VERTICAL = enum.auto()
cursors = Cursors
_tool_registry = set()

def _register_tool_class(canvas_cls, tool_cls=None):
    if False:
        print('Hello World!')
    'Decorator registering *tool_cls* as a tool class for *canvas_cls*.'
    if tool_cls is None:
        return functools.partial(_register_tool_class, canvas_cls)
    _tool_registry.add((canvas_cls, tool_cls))
    return tool_cls

def _find_tool_class(canvas_cls, tool_cls):
    if False:
        while True:
            i = 10
    'Find a subclass of *tool_cls* registered for *canvas_cls*.'
    for canvas_parent in canvas_cls.__mro__:
        for tool_child in _api.recursive_subclasses(tool_cls):
            if (canvas_parent, tool_child) in _tool_registry:
                return tool_child
    return tool_cls
_views_positions = 'viewpos'

class ToolBase:
    """
    Base tool class.

    A base tool, only implements `trigger` method or no method at all.
    The tool is instantiated by `matplotlib.backend_managers.ToolManager`.
    """
    default_keymap = None
    '\n    Keymap to associate with this tool.\n\n    ``list[str]``: List of keys that will trigger this tool when a keypress\n    event is emitted on ``self.figure.canvas``.  Note that this attribute is\n    looked up on the instance, and can therefore be a property (this is used\n    e.g. by the built-in tools to load the rcParams at instantiation time).\n    '
    description = None
    '\n    Description of the Tool.\n\n    `str`: Tooltip used if the Tool is included in a Toolbar.\n    '
    image = None
    '\n    Filename of the image.\n\n    `str`: Filename of the image to use in a Toolbar.  If None, the *name* is\n    used as a label in the toolbar button.\n    '

    def __init__(self, toolmanager, name):
        if False:
            print('Hello World!')
        self._name = name
        self._toolmanager = toolmanager
        self._figure = None
    name = property(lambda self: self._name, doc='The tool id (str, must be unique among tools of a tool manager).')
    toolmanager = property(lambda self: self._toolmanager, doc='The `.ToolManager` that controls this tool.')
    canvas = property(lambda self: self._figure.canvas if self._figure is not None else None, doc='The canvas of the figure affected by this tool, or None.')

    def set_figure(self, figure):
        if False:
            while True:
                i = 10
        self._figure = figure
    figure = property(lambda self: self._figure, lambda self, figure: self.set_figure(figure), doc='The Figure affected by this tool, or None.')

    def _make_classic_style_pseudo_toolbar(self):
        if False:
            while True:
                i = 10
        '\n        Return a placeholder object with a single `canvas` attribute.\n\n        This is useful to reuse the implementations of tools already provided\n        by the classic Toolbars.\n        '
        return SimpleNamespace(canvas=self.canvas)

    def trigger(self, sender, event, data=None):
        if False:
            print('Hello World!')
        '\n        Called when this tool gets used.\n\n        This method is called by `.ToolManager.trigger_tool`.\n\n        Parameters\n        ----------\n        event : `.Event`\n            The canvas event that caused this tool to be called.\n        sender : object\n            Object that requested the tool to be triggered.\n        data : object\n            Extra data.\n        '
        pass

class ToolToggleBase(ToolBase):
    """
    Toggleable tool.

    Every time it is triggered, it switches between enable and disable.

    Parameters
    ----------
    ``*args``
        Variable length argument to be used by the Tool.
    ``**kwargs``
        `toggled` if present and True, sets the initial state of the Tool
        Arbitrary keyword arguments to be consumed by the Tool
    """
    radio_group = None
    "\n    Attribute to group 'radio' like tools (mutually exclusive).\n\n    `str` that identifies the group or **None** if not belonging to a group.\n    "
    cursor = None
    'Cursor to use when the tool is active.'
    default_toggled = False
    'Default of toggled state.'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._toggled = kwargs.pop('toggled', self.default_toggled)
        super().__init__(*args, **kwargs)

    def trigger(self, sender, event, data=None):
        if False:
            i = 10
            return i + 15
        'Calls `enable` or `disable` based on `toggled` value.'
        if self._toggled:
            self.disable(event)
        else:
            self.enable(event)
        self._toggled = not self._toggled

    def enable(self, event=None):
        if False:
            print('Hello World!')
        '\n        Enable the toggle tool.\n\n        `trigger` calls this method when `toggled` is False.\n        '
        pass

    def disable(self, event=None):
        if False:
            i = 10
            return i + 15
        '\n        Disable the toggle tool.\n\n        `trigger` call this method when `toggled` is True.\n\n        This can happen in different circumstances.\n\n        * Click on the toolbar tool button.\n        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`.\n        * Another `ToolToggleBase` derived tool is triggered\n          (from the same `.ToolManager`).\n        '
        pass

    @property
    def toggled(self):
        if False:
            i = 10
            return i + 15
        'State of the toggled tool.'
        return self._toggled

    def set_figure(self, figure):
        if False:
            i = 10
            return i + 15
        toggled = self.toggled
        if toggled:
            if self.figure:
                self.trigger(self, None)
            else:
                self._toggled = False
        super().set_figure(figure)
        if toggled:
            if figure:
                self.trigger(self, None)
            else:
                self._toggled = True

class ToolSetCursor(ToolBase):
    """
    Change to the current cursor while inaxes.

    This tool, keeps track of all `ToolToggleBase` derived tools, and updates
    the cursor when a tool gets triggered.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._id_drag = None
        self._current_tool = None
        self._default_cursor = cursors.POINTER
        self._last_cursor = self._default_cursor
        self.toolmanager.toolmanager_connect('tool_added_event', self._add_tool_cbk)
        for tool in self.toolmanager.tools.values():
            self._add_tool(tool)

    def set_figure(self, figure):
        if False:
            for i in range(10):
                print('nop')
        if self._id_drag:
            self.canvas.mpl_disconnect(self._id_drag)
        super().set_figure(figure)
        if figure:
            self._id_drag = self.canvas.mpl_connect('motion_notify_event', self._set_cursor_cbk)

    def _tool_trigger_cbk(self, event):
        if False:
            for i in range(10):
                print('nop')
        if event.tool.toggled:
            self._current_tool = event.tool
        else:
            self._current_tool = None
        self._set_cursor_cbk(event.canvasevent)

    def _add_tool(self, tool):
        if False:
            print('Hello World!')
        'Set the cursor when the tool is triggered.'
        if getattr(tool, 'cursor', None) is not None:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_trigger_cbk)

    def _add_tool_cbk(self, event):
        if False:
            return 10
        'Process every newly added tool.'
        if event.tool is self:
            return
        self._add_tool(event.tool)

    def _set_cursor_cbk(self, event):
        if False:
            print('Hello World!')
        if not event or not self.canvas:
            return
        if self._current_tool and getattr(event, 'inaxes', None) and event.inaxes.get_navigate():
            if self._last_cursor != self._current_tool.cursor:
                self.canvas.set_cursor(self._current_tool.cursor)
                self._last_cursor = self._current_tool.cursor
        elif self._last_cursor != self._default_cursor:
            self.canvas.set_cursor(self._default_cursor)
            self._last_cursor = self._default_cursor

class ToolCursorPosition(ToolBase):
    """
    Send message with the current pointer position.

    This tool runs in the background reporting the position of the cursor.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._id_drag = None
        super().__init__(*args, **kwargs)

    def set_figure(self, figure):
        if False:
            for i in range(10):
                print('nop')
        if self._id_drag:
            self.canvas.mpl_disconnect(self._id_drag)
        super().set_figure(figure)
        if figure:
            self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.send_message)

    def send_message(self, event):
        if False:
            while True:
                i = 10
        'Call `matplotlib.backend_managers.ToolManager.message_event`.'
        if self.toolmanager.messagelock.locked():
            return
        from matplotlib.backend_bases import NavigationToolbar2
        message = NavigationToolbar2._mouse_event_to_message(event)
        self.toolmanager.message_event(message, self)

class RubberbandBase(ToolBase):
    """Draw and remove a rubberband."""

    def trigger(self, sender, event, data=None):
        if False:
            while True:
                i = 10
        'Call `draw_rubberband` or `remove_rubberband` based on data.'
        if not self.figure.canvas.widgetlock.available(sender):
            return
        if data is not None:
            self.draw_rubberband(*data)
        else:
            self.remove_rubberband()

    def draw_rubberband(self, *data):
        if False:
            while True:
                i = 10
        '\n        Draw rubberband.\n\n        This method must get implemented per backend.\n        '
        raise NotImplementedError

    def remove_rubberband(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove rubberband.\n\n        This method should get implemented per backend.\n        '
        pass

class ToolQuit(ToolBase):
    """Tool to call the figure manager destroy method."""
    description = 'Quit the figure'
    default_keymap = property(lambda self: mpl.rcParams['keymap.quit'])

    def trigger(self, sender, event, data=None):
        if False:
            while True:
                i = 10
        Gcf.destroy_fig(self.figure)

class ToolQuitAll(ToolBase):
    """Tool to call the figure manager destroy method."""
    description = 'Quit all figures'
    default_keymap = property(lambda self: mpl.rcParams['keymap.quit_all'])

    def trigger(self, sender, event, data=None):
        if False:
            return 10
        Gcf.destroy_all()

class ToolGrid(ToolBase):
    """Tool to toggle the major grids of the figure."""
    description = 'Toggle major grids'
    default_keymap = property(lambda self: mpl.rcParams['keymap.grid'])

    def trigger(self, sender, event, data=None):
        if False:
            for i in range(10):
                print('nop')
        sentinel = str(uuid.uuid4())
        with cbook._setattr_cm(event, key=sentinel), mpl.rc_context({'keymap.grid': sentinel}):
            mpl.backend_bases.key_press_handler(event, self.figure.canvas)

class ToolMinorGrid(ToolBase):
    """Tool to toggle the major and minor grids of the figure."""
    description = 'Toggle major and minor grids'
    default_keymap = property(lambda self: mpl.rcParams['keymap.grid_minor'])

    def trigger(self, sender, event, data=None):
        if False:
            i = 10
            return i + 15
        sentinel = str(uuid.uuid4())
        with cbook._setattr_cm(event, key=sentinel), mpl.rc_context({'keymap.grid_minor': sentinel}):
            mpl.backend_bases.key_press_handler(event, self.figure.canvas)

class ToolFullScreen(ToolBase):
    """Tool to toggle full screen."""
    description = 'Toggle fullscreen mode'
    default_keymap = property(lambda self: mpl.rcParams['keymap.fullscreen'])

    def trigger(self, sender, event, data=None):
        if False:
            for i in range(10):
                print('nop')
        self.figure.canvas.manager.full_screen_toggle()

class AxisScaleBase(ToolToggleBase):
    """Base Tool to toggle between linear and logarithmic."""

    def trigger(self, sender, event, data=None):
        if False:
            i = 10
            return i + 15
        if event.inaxes is None:
            return
        super().trigger(sender, event, data)

    def enable(self, event=None):
        if False:
            while True:
                i = 10
        self.set_scale(event.inaxes, 'log')
        self.figure.canvas.draw_idle()

    def disable(self, event=None):
        if False:
            i = 10
            return i + 15
        self.set_scale(event.inaxes, 'linear')
        self.figure.canvas.draw_idle()

class ToolYScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the Y axis."""
    description = 'Toggle scale Y axis'
    default_keymap = property(lambda self: mpl.rcParams['keymap.yscale'])

    def set_scale(self, ax, scale):
        if False:
            for i in range(10):
                print('nop')
        ax.set_yscale(scale)

class ToolXScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the X axis."""
    description = 'Toggle scale X axis'
    default_keymap = property(lambda self: mpl.rcParams['keymap.xscale'])

    def set_scale(self, ax, scale):
        if False:
            for i in range(10):
                print('nop')
        ax.set_xscale(scale)

class ToolViewsPositions(ToolBase):
    """
    Auxiliary Tool to handle changes in views and positions.

    Runs in the background and should get used by all the tools that
    need to access the figure's history of views and positions, e.g.

    * `ToolZoom`
    * `ToolPan`
    * `ToolHome`
    * `ToolBack`
    * `ToolForward`
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.views = WeakKeyDictionary()
        self.positions = WeakKeyDictionary()
        self.home_views = WeakKeyDictionary()
        super().__init__(*args, **kwargs)

    def add_figure(self, figure):
        if False:
            while True:
                i = 10
        'Add the current figure to the stack of views and positions.'
        if figure not in self.views:
            self.views[figure] = cbook._Stack()
            self.positions[figure] = cbook._Stack()
            self.home_views[figure] = WeakKeyDictionary()
            self.push_current(figure)
            figure.add_axobserver(lambda fig: self.update_home_views(fig))

    def clear(self, figure):
        if False:
            for i in range(10):
                print('nop')
        'Reset the axes stack.'
        if figure in self.views:
            self.views[figure].clear()
            self.positions[figure].clear()
            self.home_views[figure].clear()
            self.update_home_views()

    def update_view(self):
        if False:
            return 10
        "\n        Update the view limits and position for each axes from the current\n        stack position. If any axes are present in the figure that aren't in\n        the current stack position, use the home view limits for those axes and\n        don't update *any* positions.\n        "
        views = self.views[self.figure]()
        if views is None:
            return
        pos = self.positions[self.figure]()
        if pos is None:
            return
        home_views = self.home_views[self.figure]
        all_axes = self.figure.get_axes()
        for a in all_axes:
            if a in views:
                cur_view = views[a]
            else:
                cur_view = home_views[a]
            a._set_view(cur_view)
        if set(all_axes).issubset(pos):
            for a in all_axes:
                a._set_position(pos[a][0], 'original')
                a._set_position(pos[a][1], 'active')
        self.figure.canvas.draw_idle()

    def push_current(self, figure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Push the current view limits and position onto their respective stacks.\n        '
        if not figure:
            figure = self.figure
        views = WeakKeyDictionary()
        pos = WeakKeyDictionary()
        for a in figure.get_axes():
            views[a] = a._get_view()
            pos[a] = self._axes_pos(a)
        self.views[figure].push(views)
        self.positions[figure].push(pos)

    def _axes_pos(self, ax):
        if False:
            print('Hello World!')
        '\n        Return the original and modified positions for the specified axes.\n\n        Parameters\n        ----------\n        ax : matplotlib.axes.Axes\n            The `.Axes` to get the positions for.\n\n        Returns\n        -------\n        original_position, modified_position\n            A tuple of the original and modified positions.\n        '
        return (ax.get_position(True).frozen(), ax.get_position().frozen())

    def update_home_views(self, figure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure that ``self.home_views`` has an entry for all axes present\n        in the figure.\n        '
        if not figure:
            figure = self.figure
        for a in figure.get_axes():
            if a not in self.home_views[figure]:
                self.home_views[figure][a] = a._get_view()

    def home(self):
        if False:
            print('Hello World!')
        'Recall the first view and position from the stack.'
        self.views[self.figure].home()
        self.positions[self.figure].home()

    def back(self):
        if False:
            return 10
        'Back one step in the stack of views and positions.'
        self.views[self.figure].back()
        self.positions[self.figure].back()

    def forward(self):
        if False:
            while True:
                i = 10
        'Forward one step in the stack of views and positions.'
        self.views[self.figure].forward()
        self.positions[self.figure].forward()

class ViewsPositionsBase(ToolBase):
    """Base class for `ToolHome`, `ToolBack` and `ToolForward`."""
    _on_trigger = None

    def trigger(self, sender, event, data=None):
        if False:
            print('Hello World!')
        self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
        getattr(self.toolmanager.get_tool(_views_positions), self._on_trigger)()
        self.toolmanager.get_tool(_views_positions).update_view()

class ToolHome(ViewsPositionsBase):
    """Restore the original view limits."""
    description = 'Reset original view'
    image = 'home'
    default_keymap = property(lambda self: mpl.rcParams['keymap.home'])
    _on_trigger = 'home'

class ToolBack(ViewsPositionsBase):
    """Move back up the view limits stack."""
    description = 'Back to previous view'
    image = 'back'
    default_keymap = property(lambda self: mpl.rcParams['keymap.back'])
    _on_trigger = 'back'

class ToolForward(ViewsPositionsBase):
    """Move forward in the view lim stack."""
    description = 'Forward to next view'
    image = 'forward'
    default_keymap = property(lambda self: mpl.rcParams['keymap.forward'])
    _on_trigger = 'forward'

class ConfigureSubplotsBase(ToolBase):
    """Base tool for the configuration of subplots."""
    description = 'Configure subplots'
    image = 'subplots'

class SaveFigureBase(ToolBase):
    """Base tool for figure saving."""
    description = 'Save the figure'
    image = 'filesave'
    default_keymap = property(lambda self: mpl.rcParams['keymap.save'])

class ZoomPanBase(ToolToggleBase):
    """Base class for `ToolZoom` and `ToolPan`."""

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self._button_pressed = None
        self._xypress = None
        self._idPress = None
        self._idRelease = None
        self._idScroll = None
        self.base_scale = 2.0
        self.scrollthresh = 0.5
        self.lastscroll = time.time() - self.scrollthresh

    def enable(self, event=None):
        if False:
            return 10
        'Connect press/release events and lock the canvas.'
        self.figure.canvas.widgetlock(self)
        self._idPress = self.figure.canvas.mpl_connect('button_press_event', self._press)
        self._idRelease = self.figure.canvas.mpl_connect('button_release_event', self._release)
        self._idScroll = self.figure.canvas.mpl_connect('scroll_event', self.scroll_zoom)

    def disable(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        'Release the canvas and disconnect press/release events.'
        self._cancel_action()
        self.figure.canvas.widgetlock.release(self)
        self.figure.canvas.mpl_disconnect(self._idPress)
        self.figure.canvas.mpl_disconnect(self._idRelease)
        self.figure.canvas.mpl_disconnect(self._idScroll)

    def trigger(self, sender, event, data=None):
        if False:
            while True:
                i = 10
        self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
        super().trigger(sender, event, data)
        new_navigate_mode = self.name.upper() if self.toggled else None
        for ax in self.figure.axes:
            ax.set_navigate_mode(new_navigate_mode)

    def scroll_zoom(self, event):
        if False:
            while True:
                i = 10
        if event.inaxes is None:
            return
        if event.button == 'up':
            scl = self.base_scale
        elif event.button == 'down':
            scl = 1 / self.base_scale
        else:
            scl = 1
        ax = event.inaxes
        ax._set_view_from_bbox([event.x, event.y, scl])
        if time.time() - self.lastscroll < self.scrollthresh:
            self.toolmanager.get_tool(_views_positions).back()
        self.figure.canvas.draw_idle()
        self.lastscroll = time.time()
        self.toolmanager.get_tool(_views_positions).push_current()

class ToolZoom(ZoomPanBase):
    """A Tool for zooming using a rectangle selector."""
    description = 'Zoom to rectangle'
    image = 'zoom_to_rect'
    default_keymap = property(lambda self: mpl.rcParams['keymap.zoom'])
    cursor = cursors.SELECT_REGION
    radio_group = 'default'

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self._ids_zoom = []

    def _cancel_action(self):
        if False:
            i = 10
            return i + 15
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self.toolmanager.trigger_tool('rubberband', self)
        self.figure.canvas.draw_idle()
        self._xypress = None
        self._button_pressed = None
        self._ids_zoom = []
        return

    def _press(self, event):
        if False:
            while True:
                i = 10
        'Callback for mouse button presses in zoom-to-rectangle mode.'
        if self._ids_zoom:
            self._cancel_action()
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return
        (x, y) = (event.x, event.y)
        self._xypress = []
        for (i, a) in enumerate(self.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(event) and a.get_navigate() and a.can_zoom():
                self._xypress.append((x, y, a, i, a._get_view()))
        id1 = self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_move)
        id2 = self.figure.canvas.mpl_connect('key_press_event', self._switch_on_zoom_mode)
        id3 = self.figure.canvas.mpl_connect('key_release_event', self._switch_off_zoom_mode)
        self._ids_zoom = (id1, id2, id3)
        self._zoom_mode = event.key

    def _switch_on_zoom_mode(self, event):
        if False:
            return 10
        self._zoom_mode = event.key
        self._mouse_move(event)

    def _switch_off_zoom_mode(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._zoom_mode = None
        self._mouse_move(event)

    def _mouse_move(self, event):
        if False:
            i = 10
            return i + 15
        'Callback for mouse moves in zoom-to-rectangle mode.'
        if self._xypress:
            (x, y) = (event.x, event.y)
            (lastx, lasty, a, ind, view) = self._xypress[0]
            ((x1, y1), (x2, y2)) = np.clip([[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
            if self._zoom_mode == 'x':
                (y1, y2) = a.bbox.intervaly
            elif self._zoom_mode == 'y':
                (x1, x2) = a.bbox.intervalx
            self.toolmanager.trigger_tool('rubberband', self, data=(x1, y1, x2, y2))

    def _release(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Callback for mouse button releases in zoom-to-rectangle mode.'
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []
        if not self._xypress:
            self._cancel_action()
            return
        done_ax = []
        for cur_xypress in self._xypress:
            (x, y) = (event.x, event.y)
            (lastx, lasty, a, _ind, view) = cur_xypress
            if abs(x - lastx) < 5 or abs(y - lasty) < 5:
                self._cancel_action()
                return
            twinx = any((a.get_shared_x_axes().joined(a, a1) for a1 in done_ax))
            twiny = any((a.get_shared_y_axes().joined(a, a1) for a1 in done_ax))
            done_ax.append(a)
            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue
            a._set_view_from_bbox((lastx, lasty, x, y), direction, self._zoom_mode, twinx, twiny)
        self._zoom_mode = None
        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()

class ToolPan(ZoomPanBase):
    """Pan axes with left mouse, zoom with right."""
    default_keymap = property(lambda self: mpl.rcParams['keymap.pan'])
    description = 'Pan axes with left mouse, zoom with right'
    image = 'move'
    cursor = cursors.MOVE
    radio_group = 'default'

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self._id_drag = None

    def _cancel_action(self):
        if False:
            i = 10
            return i + 15
        self._button_pressed = None
        self._xypress = []
        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)
        self.figure.canvas.draw_idle()

    def _press(self, event):
        if False:
            print('Hello World!')
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return
        (x, y) = (event.x, event.y)
        self._xypress = []
        for (i, a) in enumerate(self.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(event) and a.get_navigate() and a.can_pan():
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                self.toolmanager.messagelock(self)
                self._id_drag = self.figure.canvas.mpl_connect('motion_notify_event', self._mouse_move)

    def _release(self, event):
        if False:
            return 10
        if self._button_pressed is None:
            self._cancel_action()
            return
        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)
        for (a, _ind) in self._xypress:
            a.end_pan()
        if not self._xypress:
            self._cancel_action()
            return
        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()

    def _mouse_move(self, event):
        if False:
            return 10
        for (a, _ind) in self._xypress:
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        self.toolmanager.canvas.draw_idle()

class ToolHelpBase(ToolBase):
    description = 'Print tool list, shortcuts and description'
    default_keymap = property(lambda self: mpl.rcParams['keymap.help'])
    image = 'help'

    @staticmethod
    def format_shortcut(key_sequence):
        if False:
            i = 10
            return i + 15
        "\n        Convert a shortcut string from the notation used in rc config to the\n        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.\n        "
        return key_sequence if len(key_sequence) == 1 else re.sub('\\+[A-Z]', '+Shift\\g<0>', key_sequence).title()

    def _format_tool_keymap(self, name):
        if False:
            while True:
                i = 10
        keymaps = self.toolmanager.get_tool_keymap(name)
        return ', '.join((self.format_shortcut(keymap) for keymap in keymaps))

    def _get_help_entries(self):
        if False:
            for i in range(10):
                print('nop')
        return [(name, self._format_tool_keymap(name), tool.description) for (name, tool) in sorted(self.toolmanager.tools.items()) if tool.description]

    def _get_help_text(self):
        if False:
            i = 10
            return i + 15
        entries = self._get_help_entries()
        entries = ['{}: {}\n\t{}'.format(*entry) for entry in entries]
        return '\n'.join(entries)

    def _get_help_html(self):
        if False:
            while True:
                i = 10
        fmt = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'
        rows = [fmt.format('<b>Action</b>', '<b>Shortcuts</b>', '<b>Description</b>')]
        rows += [fmt.format(*row) for row in self._get_help_entries()]
        return '<style>td {padding: 0px 4px}</style><table><thead>' + rows[0] + '</thead><tbody>'.join(rows[1:]) + '</tbody></table>'

class ToolCopyToClipboardBase(ToolBase):
    """Tool to copy the figure to the clipboard."""
    description = 'Copy the canvas figure to clipboard'
    default_keymap = property(lambda self: mpl.rcParams['keymap.copy'])

    def trigger(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        message = 'Copy tool is not available'
        self.toolmanager.message_event(message, self)
default_tools = {'home': ToolHome, 'back': ToolBack, 'forward': ToolForward, 'zoom': ToolZoom, 'pan': ToolPan, 'subplots': ConfigureSubplotsBase, 'save': SaveFigureBase, 'grid': ToolGrid, 'grid_minor': ToolMinorGrid, 'fullscreen': ToolFullScreen, 'quit': ToolQuit, 'quit_all': ToolQuitAll, 'xscale': ToolXScale, 'yscale': ToolYScale, 'position': ToolCursorPosition, _views_positions: ToolViewsPositions, 'cursor': ToolSetCursor, 'rubberband': RubberbandBase, 'help': ToolHelpBase, 'copy': ToolCopyToClipboardBase}
default_toolbar_tools = [['navigation', ['home', 'back', 'forward']], ['zoompan', ['pan', 'zoom', 'subplots']], ['io', ['save', 'help']]]

def add_tools_to_manager(toolmanager, tools=default_tools):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add multiple tools to a `.ToolManager`.\n\n    Parameters\n    ----------\n    toolmanager : `.backend_managers.ToolManager`\n        Manager to which the tools are added.\n    tools : {str: class_like}, optional\n        The tools to add in a {name: tool} dict, see\n        `.backend_managers.ToolManager.add_tool` for more info.\n    '
    for (name, tool) in tools.items():
        toolmanager.add_tool(name, tool)

def add_tools_to_container(container, tools=default_toolbar_tools):
    if False:
        while True:
            i = 10
    '\n    Add multiple tools to the container.\n\n    Parameters\n    ----------\n    container : Container\n        `.backend_bases.ToolContainerBase` object that will get the tools\n        added.\n    tools : list, optional\n        List in the form ``[[group1, [tool1, tool2 ...]], [group2, [...]]]``\n        where the tools ``[tool1, tool2, ...]`` will display in group1.\n        See `.backend_bases.ToolContainerBase.add_tool` for details.\n    '
    for (group, grouptools) in tools:
        for (position, tool) in enumerate(grouptools):
            container.add_tool(tool, group, position)