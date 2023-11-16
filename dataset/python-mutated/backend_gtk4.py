import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import ToolContainerBase, KeyEvent, LocationEvent, MouseEvent, ResizeEvent, CloseEvent
try:
    import gi
except ImportError as err:
    raise ImportError('The GTK4 backends require PyGObject') from err
try:
    gi.require_version('Gtk', '4.0')
except ValueError as e:
    raise ImportError(e) from e
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import _BackendGTK, _FigureCanvasGTK, _FigureManagerGTK, _NavigationToolbar2GTK, TimerGTK as TimerGTK4

class FigureCanvasGTK4(_FigureCanvasGTK, Gtk.DrawingArea):
    required_interactive_framework = 'gtk4'
    supports_blit = False
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK4)
    _context_is_scaled = False

    def __init__(self, figure=None):
        if False:
            while True:
                i = 10
        super().__init__(figure=figure)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self._idle_draw_id = 0
        self._rubberband_rect = None
        self.set_draw_func(self._draw_func)
        self.connect('resize', self.resize_event)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)
        click = Gtk.GestureClick()
        click.set_button(0)
        click.connect('pressed', self.button_press_event)
        click.connect('released', self.button_release_event)
        self.add_controller(click)
        key = Gtk.EventControllerKey()
        key.connect('key-pressed', self.key_press_event)
        key.connect('key-released', self.key_release_event)
        self.add_controller(key)
        motion = Gtk.EventControllerMotion()
        motion.connect('motion', self.motion_notify_event)
        motion.connect('enter', self.enter_notify_event)
        motion.connect('leave', self.leave_notify_event)
        self.add_controller(motion)
        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect('scroll', self.scroll_event)
        self.add_controller(scroll)
        self.set_focusable(True)
        css = Gtk.CssProvider()
        style = '.matplotlib-canvas { background-color: white; }'
        if Gtk.check_version(4, 9, 3) is None:
            css.load_from_data(style, -1)
        else:
            css.load_from_data(style.encode('utf-8'))
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        style_ctx.add_class('matplotlib-canvas')

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        CloseEvent('close_event', self)._process()

    def set_cursor(self, cursor):
        if False:
            i = 10
            return i + 15
        self.set_cursor_from_name(_backend_gtk.mpl_to_gtk_cursor_name(cursor))

    def _mpl_coords(self, xy=None):
        if False:
            return 10
        '\n        Convert the *xy* position of a GTK event, or of the current cursor\n        position if *xy* is None, to Matplotlib coordinates.\n\n        GTK use logical pixels, but the figure is scaled to physical pixels for\n        rendering.  Transform to physical pixels so that all of the down-stream\n        transforms work as expected.\n\n        Also, the origin is different and needs to be corrected.\n        '
        if xy is None:
            surface = self.get_native().get_surface()
            (is_over, x, y, mask) = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
        else:
            (x, y) = xy
        x = x * self.device_pixel_ratio
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        return (x, y)

    def scroll_event(self, controller, dx, dy):
        if False:
            while True:
                i = 10
        MouseEvent('scroll_event', self, *self._mpl_coords(), step=dy, modifiers=self._mpl_modifiers(controller))._process()
        return True

    def button_press_event(self, controller, n_press, x, y):
        if False:
            while True:
                i = 10
        MouseEvent('button_press_event', self, *self._mpl_coords((x, y)), controller.get_current_button(), modifiers=self._mpl_modifiers(controller))._process()
        self.grab_focus()

    def button_release_event(self, controller, n_press, x, y):
        if False:
            for i in range(10):
                print('nop')
        MouseEvent('button_release_event', self, *self._mpl_coords((x, y)), controller.get_current_button(), modifiers=self._mpl_modifiers(controller))._process()

    def key_press_event(self, controller, keyval, keycode, state):
        if False:
            for i in range(10):
                print('nop')
        KeyEvent('key_press_event', self, self._get_key(keyval, keycode, state), *self._mpl_coords())._process()
        return True

    def key_release_event(self, controller, keyval, keycode, state):
        if False:
            while True:
                i = 10
        KeyEvent('key_release_event', self, self._get_key(keyval, keycode, state), *self._mpl_coords())._process()
        return True

    def motion_notify_event(self, controller, x, y):
        if False:
            return 10
        MouseEvent('motion_notify_event', self, *self._mpl_coords((x, y)), modifiers=self._mpl_modifiers(controller))._process()

    def enter_notify_event(self, controller, x, y):
        if False:
            print('Hello World!')
        LocationEvent('figure_enter_event', self, *self._mpl_coords((x, y)), modifiers=self._mpl_modifiers())._process()

    def leave_notify_event(self, controller):
        if False:
            while True:
                i = 10
        LocationEvent('figure_leave_event', self, *self._mpl_coords(), modifiers=self._mpl_modifiers())._process()

    def resize_event(self, area, width, height):
        if False:
            while True:
                i = 10
        self._update_device_pixel_ratio()
        dpi = self.figure.dpi
        winch = width * self.device_pixel_ratio / dpi
        hinch = height * self.device_pixel_ratio / dpi
        self.figure.set_size_inches(winch, hinch, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def _mpl_modifiers(self, controller=None):
        if False:
            print('Hello World!')
        if controller is None:
            surface = self.get_native().get_surface()
            (is_over, x, y, event_state) = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
        else:
            event_state = controller.get_current_event_state()
        mod_table = [('ctrl', Gdk.ModifierType.CONTROL_MASK), ('alt', Gdk.ModifierType.ALT_MASK), ('shift', Gdk.ModifierType.SHIFT_MASK), ('super', Gdk.ModifierType.SUPER_MASK)]
        return [name for (name, mask) in mod_table if event_state & mask]

    def _get_key(self, keyval, keycode, state):
        if False:
            print('Hello World!')
        unikey = chr(Gdk.keyval_to_unicode(keyval))
        key = cbook._unikey_or_keysym_to_mplkey(unikey, Gdk.keyval_name(keyval))
        modifiers = [('ctrl', Gdk.ModifierType.CONTROL_MASK, 'control'), ('alt', Gdk.ModifierType.ALT_MASK, 'alt'), ('shift', Gdk.ModifierType.SHIFT_MASK, 'shift'), ('super', Gdk.ModifierType.SUPER_MASK, 'super')]
        mods = [mod for (mod, mask, mod_key) in modifiers if mod_key != key and state & mask and (not (mod == 'shift' and unikey.isprintable()))]
        return '+'.join([*mods, key])

    def _update_device_pixel_ratio(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            self.draw()

    def _draw_rubberband(self, rect):
        if False:
            for i in range(10):
                print('nop')
        self._rubberband_rect = rect
        self.queue_draw()

    def _draw_func(self, drawing_area, ctx, width, height):
        if False:
            for i in range(10):
                print('nop')
        self.on_draw_event(self, ctx)
        self._post_draw(self, ctx)

    def _post_draw(self, widget, ctx):
        if False:
            while True:
                i = 10
        if self._rubberband_rect is None:
            return
        lw = 1
        dash = 3
        if not self._context_is_scaled:
            (x0, y0, w, h) = (dim / self.device_pixel_ratio for dim in self._rubberband_rect)
        else:
            (x0, y0, w, h) = self._rubberband_rect
            lw *= self.device_pixel_ratio
            dash *= self.device_pixel_ratio
        x1 = x0 + w
        y1 = y0 + h
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)
        ctx.set_antialias(1)
        ctx.set_line_width(lw)
        ctx.set_dash((dash, dash), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()
        ctx.set_dash((dash, dash), dash)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

    def on_draw_event(self, widget, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def draw(self):
        if False:
            i = 10
            return i + 15
        if self.is_drawable():
            self.queue_draw()

    def draw_idle(self):
        if False:
            print('Hello World!')
        if self._idle_draw_id != 0:
            return

        def idle_draw(*args):
            if False:
                i = 10
                return i + 15
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        self._idle_draw_id = GLib.idle_add(idle_draw)

    def flush_events(self):
        if False:
            i = 10
            return i + 15
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(True)

class NavigationToolbar2GTK4(_NavigationToolbar2GTK, Gtk.Box):

    def __init__(self, canvas):
        if False:
            print('Hello World!')
        Gtk.Box.__init__(self)
        self.add_css_class('toolbar')
        self._gtk_ids = {}
        for (text, tooltip_text, image_file, callback) in self.toolitems:
            if text is None:
                self.append(Gtk.Separator())
                continue
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(str(cbook._get_data_path('images', f'{image_file}-symbolic.svg'))))
            self._gtk_ids[text] = button = Gtk.ToggleButton() if callback in ['zoom', 'pan'] else Gtk.Button()
            button.set_child(image)
            button.add_css_class('flat')
            button.add_css_class('image-button')
            button._signal_handler = button.connect('clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.append(button)
        label = Gtk.Label()
        label.set_markup('<small>\xa0\n\xa0</small>')
        label.set_hexpand(True)
        self.append(label)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        self.append(self.message)
        _NavigationToolbar2GTK.__init__(self, canvas)

    def save_figure(self, *args):
        if False:
            print('Hello World!')
        dialog = Gtk.FileChooserNative(title='Save the figure', transient_for=self.canvas.get_root(), action=Gtk.FileChooserAction.SAVE, modal=True)
        self._save_dialog = dialog
        ff = Gtk.FileFilter()
        ff.set_name('All files')
        ff.add_pattern('*')
        dialog.add_filter(ff)
        dialog.set_filter(ff)
        formats = []
        default_format = None
        for (i, (name, fmts)) in enumerate(self.canvas.get_supported_filetypes_grouped().items()):
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            formats.append(name)
            if self.canvas.get_default_filetype() in fmts:
                default_format = i
        formats = [formats[default_format], *formats[:default_format], *formats[default_format + 1:]]
        dialog.add_choice('format', 'File format', formats, formats)
        dialog.set_choice('format', formats[default_format])
        dialog.set_current_folder(Gio.File.new_for_path(os.path.expanduser(mpl.rcParams['savefig.directory'])))
        dialog.set_current_name(self.canvas.get_default_filename())

        @functools.partial(dialog.connect, 'response')
        def on_response(dialog, response):
            if False:
                while True:
                    i = 10
            file = dialog.get_file()
            fmt = dialog.get_choice('format')
            fmt = self.canvas.get_supported_filetypes_grouped()[fmt][0]
            dialog.destroy()
            self._save_dialog = None
            if response != Gtk.ResponseType.ACCEPT:
                return
            if mpl.rcParams['savefig.directory']:
                parent = file.get_parent()
                mpl.rcParams['savefig.directory'] = parent.get_path()
            try:
                self.canvas.figure.savefig(file.get_path(), format=fmt)
            except Exception as e:
                msg = Gtk.MessageDialog(transient_for=self.canvas.get_root(), message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK, modal=True, text=str(e))
                msg.show()
        dialog.show()

class ToolbarGTK4(ToolContainerBase, Gtk.Box):
    _icon_extension = '-symbolic.svg'

    def __init__(self, toolmanager):
        if False:
            return 10
        ToolContainerBase.__init__(self, toolmanager)
        Gtk.Box.__init__(self)
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        self._tool_box = Gtk.Box()
        self.append(self._tool_box)
        self._groups = {}
        self._toolitems = {}
        label = Gtk.Label()
        label.set_markup('<small>\xa0\n\xa0</small>')
        label.set_hexpand(True)
        self.append(label)
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)
        self.append(self._message)

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        if False:
            return 10
        if toggle:
            button = Gtk.ToggleButton()
        else:
            button = Gtk.Button()
        button.set_label(name)
        button.add_css_class('flat')
        if image_file is not None:
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(image_file))
            button.set_child(image)
            button.add_css_class('image-button')
        if position is None:
            position = -1
        self._add_button(button, group, position)
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    def _find_child_at_position(self, group, position):
        if False:
            i = 10
            return i + 15
        children = [None]
        child = self._groups[group].get_first_child()
        while child is not None:
            children.append(child)
            child = child.get_next_sibling()
        return children[position]

    def _add_button(self, button, group, position):
        if False:
            while True:
                i = 10
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            group_box = Gtk.Box()
            self._tool_box.append(group_box)
            self._groups[group] = group_box
        self._groups[group].insert_child_after(button, self._find_child_at_position(group, position))

    def _call_tool(self, btn, name):
        if False:
            return 10
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if False:
            print('Hello World!')
        if name not in self._toolitems:
            return
        for (toolitem, signal) in self._toolitems[name]:
            toolitem.handler_block(signal)
            toolitem.set_active(toggled)
            toolitem.handler_unblock(signal)

    def remove_toolitem(self, name):
        if False:
            return 10
        if name not in self._toolitems:
            self.toolmanager.message_event(f'{name} not in toolbar', self)
            return
        for group in self._groups:
            for (toolitem, _signal) in self._toolitems[name]:
                if toolitem in self._groups[group]:
                    self._groups[group].remove(toolitem)
        del self._toolitems[name]

    def _add_separator(self):
        if False:
            print('Hello World!')
        sep = Gtk.Separator()
        sep.set_property('orientation', Gtk.Orientation.VERTICAL)
        self._tool_box.append(sep)

    def set_message(self, s):
        if False:
            i = 10
            return i + 15
        self._message.set_label(s)

@backend_tools._register_tool_class(FigureCanvasGTK4)
class SaveFigureGTK4(backend_tools.SaveFigureBase):

    def trigger(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        NavigationToolbar2GTK4.save_figure(self._make_classic_style_pseudo_toolbar())

@backend_tools._register_tool_class(FigureCanvasGTK4)
class HelpGTK4(backend_tools.ToolHelpBase):

    def _normalize_shortcut(self, key):
        if False:
            return 10
        '\n        Convert Matplotlib key presses to GTK+ accelerator identifiers.\n\n        Related to `FigureCanvasGTK4._get_key`.\n        '
        special = {'backspace': 'BackSpace', 'pagedown': 'Page_Down', 'pageup': 'Page_Up', 'scroll_lock': 'Scroll_Lock'}
        parts = key.split('+')
        mods = ['<' + mod + '>' for mod in parts[:-1]]
        key = parts[-1]
        if key in special:
            key = special[key]
        elif len(key) > 1:
            key = key.capitalize()
        elif key.isupper():
            mods += ['<shift>']
        return ''.join(mods) + key

    def _is_valid_shortcut(self, key):
        if False:
            while True:
                i = 10
        "\n        Check for a valid shortcut to be displayed.\n\n        - GTK will never send 'cmd+' (see `FigureCanvasGTK4._get_key`).\n        - The shortcut window only shows keyboard shortcuts, not mouse buttons.\n        "
        return 'cmd+' not in key and (not key.startswith('MouseButton.'))

    def trigger(self, *args):
        if False:
            while True:
                i = 10
        section = Gtk.ShortcutsSection()
        for (name, tool) in sorted(self.toolmanager.tools.items()):
            if not tool.description:
                continue
            group = Gtk.ShortcutsGroup()
            section.append(group)
            child = group.get_first_child()
            while child is not None:
                child.set_visible(False)
                child = child.get_next_sibling()
            shortcut = Gtk.ShortcutsShortcut(accelerator=' '.join((self._normalize_shortcut(key) for key in self.toolmanager.get_tool_keymap(name) if self._is_valid_shortcut(key))), title=tool.name, subtitle=tool.description)
            group.append(shortcut)
        window = Gtk.ShortcutsWindow(title='Help', modal=True, transient_for=self._figure.canvas.get_root())
        window.set_child(section)
        window.show()

@backend_tools._register_tool_class(FigureCanvasGTK4)
class ToolCopyToClipboardGTK4(backend_tools.ToolCopyToClipboardBase):

    def trigger(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        with io.BytesIO() as f:
            self.canvas.print_rgba(f)
            (w, h) = self.canvas.get_width_height()
            pb = GdkPixbuf.Pixbuf.new_from_data(f.getbuffer(), GdkPixbuf.Colorspace.RGB, True, 8, w, h, w * 4)
        clipboard = self.canvas.get_clipboard()
        clipboard.set(pb)
backend_tools._register_tool_class(FigureCanvasGTK4, _backend_gtk.ConfigureSubplotsGTK)
backend_tools._register_tool_class(FigureCanvasGTK4, _backend_gtk.RubberbandGTK)
Toolbar = ToolbarGTK4

class FigureManagerGTK4(_FigureManagerGTK):
    _toolbar2_class = NavigationToolbar2GTK4
    _toolmanager_toolbar_class = ToolbarGTK4

@_BackendGTK.export
class _BackendGTK4(_BackendGTK):
    FigureCanvas = FigureCanvasGTK4
    FigureManager = FigureManagerGTK4