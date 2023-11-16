import re
import cairocffi
try:
    from xdg.IconTheme import getIconPath
    has_xdg = True
except ImportError:
    has_xdg = False
from libqtile import bar, hook, pangocffi
from libqtile.images import Img
from libqtile.log_utils import logger
from libqtile.widget import base

class TaskList(base._Widget, base.PaddingMixin, base.MarginMixin):
    """Displays the icon and name of each window in the current group

    Contrary to WindowTabs this is an interactive widget.  The window that
    currently has focus is highlighted.

    Optional requirements: `pyxdg <https://pypi.org/project/pyxdg/>`__ is needed
    to use theme icons and to display icons on Wayland.
    """
    orientations = base.ORIENTATION_HORIZONTAL
    defaults = [('font', 'sans', 'Default font'), ('fontsize', None, 'Font size. Calculated if None.'), ('foreground', 'ffffff', 'Foreground colour'), ('fontshadow', None, 'font shadow color, default is None(no shadow)'), ('borderwidth', 2, 'Current group border width'), ('border', '215578', 'Border colour'), ('rounded', True, 'To round or not to round borders'), ('highlight_method', 'border', "Method of highlighting (one of 'border' or 'block') Uses `*_border` color settings"), ('urgent_border', 'FF0000', 'Urgent border color'), ('urgent_alert_method', 'border', "Method for alerting you of WM urgent hints (one of 'border' or 'text')"), ('unfocused_border', None, "Border color for unfocused windows. Affects only hightlight_method 'border' and 'block'. Defaults to None, which means no special color."), ('max_title_width', None, 'Max size in pixels of task title.(if set to None, as much as available.)'), ('title_width_method', None, "Method to compute the width of task title. (None, 'uniform'.)Defaults to None, the normal behaviour."), ('parse_text', None, 'Function to parse and modify window names. e.g. function in config that removes excess strings from window name: def my_func(text)    for string in [" - Chromium", " - Firefox"]:        text = text.replace(string, "")   return textthen set option parse_text=my_func'), ('spacing', None, 'Spacing between tasks.(if set to None, will be equal to margin_x)'), ('txt_minimized', '_ ', 'Text representation of the minimized window state. e.g., "_ " or "🗕 "'), ('txt_maximized', '[] ', 'Text representation of the maximized window state. e.g., "[] " or "🗖 "'), ('txt_floating', 'V ', 'Text representation of the floating window state. e.g., "V " or "🗗 "'), ('markup_normal', None, 'Text markup of the normal window state. Supports pangomarkup with markup=True.e.g., "{}" or "<span underline="low">{}</span>"'), ('markup_minimized', None, 'Text markup of the minimized window state. Supports pangomarkup with markup=True.e.g., "{}" or "<span underline="low">{}</span>"'), ('markup_maximized', None, 'Text markup of the maximized window state. Supports pangomarkup with markup=True.e.g., "{}" or "<span underline="low">{}</span>"'), ('markup_floating', None, 'Text markup of the floating window state. Supports pangomarkup with markup=True.e.g., "{}" or "<span underline="low">{}</span>"'), ('markup_focused', None, 'Text markup of the focused window state. Supports pangomarkup with markup=True.e.g., "{}" or "<span underline="low">{}</span>"'), ('icon_size', None, 'Icon size. (Calculated if set to None. Icons are hidden if set to 0.)'), ('theme_mode', None, 'When to use theme icons. `None` = never, `preferred` = use if available, `fallback` = use if app does not provide icon directly. `preferred` and `fallback` have identical behaviour on Wayland.'), ('theme_path', None, 'Path to icon theme to be used by pyxdg for icons. ``None`` will use default icon theme.'), ('window_name_location', False, 'Whether to show the location of the window in the title.'), ('window_name_location_offset', 0, 'The offset given to the window location')]

    def __init__(self, **config):
        if False:
            for i in range(10):
                print('nop')
        base._Widget.__init__(self, bar.STRETCH, **config)
        self.add_defaults(TaskList.defaults)
        self.add_defaults(base.PaddingMixin.defaults)
        self.add_defaults(base.MarginMixin.defaults)
        self._icons_cache = {}
        self._box_end_positions = []
        self.markup = False
        self.clicked = None
        if self.spacing is None:
            self.spacing = self.margin_x
        self.add_callbacks({'Button1': self.select_window})

    def box_width(self, text):
        if False:
            while True:
                i = 10
        '\n        calculate box width for given text.\n        If max_title_width is given, the returned width is limited to it.\n        '
        if self.markup:
            text = re.sub('<[^<]+?>', '', text)
        (width, _) = self.drawer.max_layout_size([text], self.font, self.fontsize)
        width = width + 2 * (self.padding_x + self.borderwidth)
        return width

    def get_taskname(self, window):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get display name for given window.\n        Depending on its state minimized, maximized and floating\n        appropriate characters are prepended.\n        '
        state = ''
        markup_str = self.markup_normal
        if self.markup_minimized or self.markup_maximized or self.markup_floating or self.markup_focused:
            enforce_markup = True
        else:
            enforce_markup = False
        if window is None:
            pass
        elif window.minimized:
            state = self.txt_minimized
            markup_str = self.markup_minimized
        elif window.maximized:
            state = self.txt_maximized
            markup_str = self.markup_maximized
        elif window.floating:
            state = self.txt_floating
            markup_str = self.markup_floating
        elif window is window.group.current_window:
            markup_str = self.markup_focused
        window_location = f'[{window.group.windows.index(window) + self.window_name_location_offset}] ' if self.window_name_location else ''
        window_name = window_location + window.name if window and window.name else '?'
        if callable(self.parse_text):
            try:
                window_name = self.parse_text(window_name)
            except:
                logger.exception('parse_text function failed:')
        if enforce_markup and markup_str is None:
            markup_str = '%s{}' % state
        if markup_str is not None:
            self.markup = True
            window_name = pangocffi.markup_escape_text(window_name)
            return markup_str.format(window_name)
        return '%s%s' % (state, window_name)

    @property
    def windows(self):
        if False:
            for i in range(10):
                print('nop')
        if self.qtile.core.name == 'x11':
            return [w for w in self.bar.screen.group.windows if w.window.get_wm_type() in ('normal', None)]
        return self.bar.screen.group.windows

    def calc_box_widths(self):
        if False:
            while True:
                i = 10
        '\n        Calculate box width for each window in current group.\n        If the available space is less than overall size of boxes,\n        the boxes are shrunk by percentage if greater than average.\n        '
        windows = self.windows
        window_count = len(windows)
        if not window_count:
            return []
        width_total = self.width - 2 * self.margin_x - (window_count - 1) * self.spacing
        width_avg = width_total / window_count
        names = [self.get_taskname(w) for w in windows]
        if self.icon_size == 0:
            icons = len(windows) * [None]
        else:
            icons = [self.get_window_icon(w) for w in windows]
        if self.title_width_method == 'uniform':
            width_uniform = width_total // window_count
            width_boxes = [width_uniform for w in range(window_count)]
        else:
            width_boxes = [self.box_width(names[idx]) + (self.icon_size + self.padding_x if icons[idx] else 0) for idx in range(window_count)]
        if self.max_title_width:
            width_boxes = [min(w, self.max_title_width) for w in width_boxes]
        width_sum = sum(width_boxes)
        if width_sum > width_total:
            width_shorter_sum = sum([w for w in width_boxes if w < width_avg])
            ratio = (width_total - width_shorter_sum) / (width_sum - width_shorter_sum)
            width_boxes = [w if w < width_avg else w * ratio for w in width_boxes]
        return zip(windows, icons, names, width_boxes)

    def _configure(self, qtile, bar):
        if False:
            while True:
                i = 10
        base._Widget._configure(self, qtile, bar)
        if not has_xdg and self.theme_mode is not None:
            logger.warning('You must install pyxdg to use theme icons.')
            self.theme_mode = None
        if self.theme_mode and self.theme_mode not in ['preferred', 'fallback']:
            logger.warning('Unexpected theme_mode (%s). Theme icons will be disabled.', self.theme_mode)
            self.theme_mode = None
        if qtile.core.name == 'wayland' and self.theme_mode is None and (self.icon_size != 0):
            self.icon_size = 0
        if self.icon_size is None:
            self.icon_size = self.bar.height - 2 * (self.borderwidth + self.margin_y)
        if self.fontsize is None:
            calc = self.bar.height - self.margin_y * 2 - self.borderwidth * 2 - self.padding_y * 2
            self.fontsize = max(calc, 1)
        self.layout = self.drawer.textlayout('', 'ffffff', self.font, self.fontsize, self.fontshadow, wrap=False)
        self.setup_hooks()

    def update(self, window=None):
        if False:
            while True:
                i = 10
        if not window or window in self.windows:
            self.bar.draw()

    def remove_icon_cache(self, window):
        if False:
            for i in range(10):
                print('nop')
        wid = window.wid
        if wid in self._icons_cache:
            self._icons_cache.pop(wid)

    def invalidate_cache(self, window):
        if False:
            return 10
        self.remove_icon_cache(window)
        self.update(window)

    def setup_hooks(self):
        if False:
            print('Hello World!')
        hook.subscribe.client_name_updated(self.update)
        hook.subscribe.focus_change(self.update)
        hook.subscribe.float_change(self.update)
        hook.subscribe.client_urgent_hint_changed(self.update)
        hook.subscribe.net_wm_icon_change(self.invalidate_cache)
        hook.subscribe.client_killed(self.remove_icon_cache)

    def drawtext(self, text, textcolor, width):
        if False:
            return 10
        if self.markup:
            self.layout.markup = self.markup
        self.layout.text = text
        self.layout.font_family = self.font
        self.layout.font_size = self.fontsize
        self.layout.colour = textcolor
        if width is not None:
            self.layout.width = width

    def drawbox(self, offset, text, bordercolor, textcolor, width=None, rounded=False, block=False, icon=None):
        if False:
            print('Hello World!')
        self.drawtext(text, textcolor, width)
        icon_padding = self.icon_size + self.padding_x if icon else 0
        padding_x = [self.padding_x + icon_padding, self.padding_x]
        if bordercolor is None:
            border_width = 0
            framecolor = self.background or self.bar.background
        else:
            border_width = self.borderwidth
            framecolor = bordercolor
        framed = self.layout.framed(border_width, framecolor, padding_x, self.padding_y)
        if block and bordercolor is not None:
            framed.draw_fill(offset, self.margin_y, rounded)
        else:
            framed.draw(offset, self.margin_y, rounded)
        if icon:
            self.draw_icon(icon, offset)

    def get_clicked(self, x, y):
        if False:
            i = 10
            return i + 15
        box_start = self.margin_x
        for (box_end, win) in zip(self._box_end_positions, self.windows):
            if box_start <= x <= box_end:
                return win
            else:
                box_start = box_end + self.spacing
        return None

    def button_press(self, x, y, button):
        if False:
            return 10
        self.clicked = self.get_clicked(x, y)
        base._Widget.button_press(self, x, y, button)

    def select_window(self):
        if False:
            for i in range(10):
                print('nop')
        if self.clicked:
            current_win = self.bar.screen.group.current_window
            window = self.clicked
            if window is not current_win:
                window.group.focus(window, False)
                if window.floating:
                    window.bring_to_front()
            else:
                window.toggle_minimize()

    def _get_class_icon(self, window):
        if False:
            while True:
                i = 10
        if not getattr(window, 'icons', False):
            return None
        icons = sorted(iter(window.icons.items()), key=lambda x: abs(self.icon_size - int(x[0].split('x')[0])))
        icon = icons[0]
        (width, height) = map(int, icon[0].split('x'))
        img = cairocffi.ImageSurface.create_for_data(icon[1], cairocffi.FORMAT_ARGB32, width, height)
        return img

    def _get_theme_icon(self, window):
        if False:
            print('Hello World!')
        classes = window.get_wm_class()
        if not classes:
            return None
        icon = None
        for cl in classes:
            for app in set([cl, cl.lower()]):
                icon = getIconPath(app, theme=self.theme_path)
                if icon is not None:
                    break
            else:
                continue
            break
        if not icon:
            return None
        img = Img.from_path(icon)
        return img.surface

    def get_window_icon(self, window):
        if False:
            print('Hello World!')
        if not getattr(window, 'icons', False) and self.theme_mode is None:
            return None
        cache = self._icons_cache.get(window.wid)
        if cache:
            return cache
        surface = None
        img = None
        if self.qtile.core.name == 'x11':
            img = self._get_class_icon(window)
        if self.theme_mode == 'preferred' or (self.theme_mode == 'fallback' and img is None):
            xdg_img = self._get_theme_icon(window)
            if xdg_img:
                img = xdg_img
        if img is not None:
            surface = cairocffi.SurfacePattern(img)
            height = img.get_height()
            width = img.get_width()
            scaler = cairocffi.Matrix()
            if height != self.icon_size:
                sp = height / self.icon_size
                height = self.icon_size
                width /= sp
                scaler.scale(sp, sp)
            surface.set_matrix(scaler)
        self._icons_cache[window.wid] = surface
        return surface

    def draw_icon(self, surface, offset):
        if False:
            i = 10
            return i + 15
        if not surface:
            return
        x = offset + self.borderwidth + self.padding_x
        y = (self.height - self.icon_size) // 2
        self.drawer.ctx.save()
        self.drawer.ctx.translate(x, y)
        self.drawer.ctx.set_source(surface)
        self.drawer.ctx.paint()
        self.drawer.ctx.restore()

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        self.drawer.clear(self.background or self.bar.background)
        offset = self.margin_x
        self._box_end_positions = []
        for (w, icon, task, bw) in self.calc_box_widths():
            self._box_end_positions.append(offset + bw)
            if w.urgent:
                border = self.urgent_border
                text_color = border
            elif w is w.group.current_window:
                border = self.border
                text_color = border
            else:
                border = self.unfocused_border or None
                text_color = self.foreground
            if self.highlight_method == 'text':
                border = None
            else:
                text_color = self.foreground
            textwidth = bw - 2 * self.padding_x - (self.icon_size + self.padding_x if icon else 0)
            self.drawbox(offset, task, border, text_color, rounded=self.rounded, block=self.highlight_method == 'block', width=textwidth, icon=icon)
            offset += bw + self.spacing
        self.drawer.draw(offsetx=self.offset, offsety=self.offsety, width=self.width)