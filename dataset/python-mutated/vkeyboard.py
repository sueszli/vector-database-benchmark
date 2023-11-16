"""
VKeyboard
=========

.. image:: images/vkeyboard.jpg
    :align: right

.. versionadded:: 1.0.8


VKeyboard is an onscreen keyboard for Kivy. Its operation is intended to be
transparent to the user. Using the widget directly is NOT recommended. Read the
section `Request keyboard`_ first.

Modes
-----

This virtual keyboard has a docked and free mode:

* docked mode (:attr:`VKeyboard.docked` = True)
  Generally used when only one person is using the computer, like a tablet or
  personal computer etc.
* free mode: (:attr:`VKeyboard.docked` = False)
  Mostly for multitouch surfaces. This mode allows multiple virtual
  keyboards to be used on the screen.

If the docked mode changes, you need to manually call
:meth:`VKeyboard.setup_mode` otherwise the change will have no impact.
During that call, the VKeyboard, implemented on top of a
:class:`~kivy.uix.scatter.Scatter`, will change the
behavior of the scatter and position the keyboard near the target (if target
and docked mode is set).


Layouts
-------

The virtual keyboard is able to load a custom layout. If you create a new
layout and put the JSON in :file:`<kivy_data_dir>/keyboards/<layoutid>.json`,
you can load it by setting :attr:`VKeyboard.layout` to your layoutid.

The JSON must be structured like this::

    {
        "title": "Title of your layout",
        "description": "Description of your layout",
        "cols": 15,
        "rows": 5,

        ...
    }

Then, you need to describe the keys in each row, for either a "normal",
"shift" or a "special" (added in version 1.9.0) mode. Keys for this row
data must be named `normal_<row>`, `shift_<row>` and `special_<row>`.
Replace `row` with the row number.
Inside each row, you will describe the key. A key is a 4 element list in
the format::

    [ <text displayed on the keyboard>, <text to put when the key is pressed>,
      <text that represents the keycode>, <size of cols> ]

Here are example keys::

    # f key
    ["f", "f", "f", 1]
    # capslock
    ["↹", "	", "tab", 1.5]

Finally, complete the JSON::

    {
        ...
        "normal_1": [
            ["`", "`", "`", 1],    ["1", "1", "1", 1],    ["2", "2", "2", 1],
            ["3", "3", "3", 1],    ["4", "4", "4", 1],    ["5", "5", "5", 1],
            ["6", "6", "6", 1],    ["7", "7", "7", 1],    ["8", "8", "8", 1],
            ["9", "9", "9", 1],    ["0", "0", "0", 1],    ["+", "+", "+", 1],
            ["=", "=", "=", 1],    ["⌫", null, "backspace", 2]
        ],

        "shift_1": [ ... ],
        "normal_2": [ ... ],
        "special_2": [ ... ],
        ...
    }


Request Keyboard
----------------

The instantiation of the virtual keyboard is controlled by the configuration.
Check `keyboard_mode` and `keyboard_layout` in the :doc:`api-kivy.config`.

If you intend to create a widget that requires a keyboard, do not use the
virtual keyboard directly, but prefer to use the best method available on
the platform. Check the :meth:`~kivy.core.window.WindowBase.request_keyboard`
method in the :doc:`api-kivy.core.window`.

If you want a specific layout when you request the keyboard, you should write
something like this (from 1.8.0, numeric.json can be in the same directory as
your main.py)::

    keyboard = Window.request_keyboard(
        self._keyboard_close, self)
    if keyboard.widget:
        vkeyboard = self._keyboard.widget
        vkeyboard.layout = 'numeric.json'

"""
__all__ = ('VKeyboard',)
from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, BooleanProperty, DictProperty, OptionProperty, ListProperty, ColorProperty
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
default_layout_path = join(kivy_data_dir, 'keyboards')

class VKeyboard(Scatter):
    """
    VKeyboard is an onscreen keyboard with multitouch support.
    Its layout is entirely customizable and you can switch between available
    layouts using a button in the bottom right of the widget.

    :Events:
        `on_key_down`: keycode, internal, modifiers
            Fired when the keyboard received a key down event (key press).
        `on_key_up`: keycode, internal, modifiers
            Fired when the keyboard received a key up event (key release).
    """
    target = ObjectProperty(None, allownone=True)
    'Target widget associated with the VKeyboard. If set, it will be used to\n    send keyboard events. If the VKeyboard mode is "free", it will also be used\n    to set the initial position.\n\n    :attr:`target` is an :class:`~kivy.properties.ObjectProperty` instance and\n    defaults to None.\n    '
    callback = ObjectProperty(None, allownone=True)
    'Callback can be set to a function that will be called if the\n    VKeyboard is closed by the user.\n\n    :attr:`target` is an :class:`~kivy.properties.ObjectProperty` instance and\n    defaults to None.\n    '
    layout = StringProperty(None)
    'Layout to use for the VKeyboard. By default, it will be the\n    layout set in the configuration, according to the `keyboard_layout`\n    in `[kivy]` section.\n\n    .. versionchanged:: 1.8.0\n        If layout is a .json filename, it will loaded and added to the\n        available_layouts.\n\n    :attr:`layout` is a :class:`~kivy.properties.StringProperty` and defaults\n    to None.\n    '
    layout_path = StringProperty(default_layout_path)
    'Path from which layouts are read.\n\n    :attr:`layout` is a :class:`~kivy.properties.StringProperty` and\n    defaults to :file:`<kivy_data_dir>/keyboards/`\n    '
    available_layouts = DictProperty({})
    'Dictionary of all available layouts. Keys are the layout ID, and the\n    value is the JSON (translated into a Python object).\n\n    :attr:`available_layouts` is a :class:`~kivy.properties.DictProperty` and\n    defaults to {}.\n    '
    docked = BooleanProperty(False)
    'Indicate whether the VKeyboard is docked on the screen or not. If you\n    change it, you must manually call :meth:`setup_mode` otherwise it will have\n    no impact. If the VKeyboard is created by the Window, the docked mode will\n    be automatically set by the configuration, using the `keyboard_mode` token\n    in `[kivy]` section.\n\n    :attr:`docked` is a :class:`~kivy.properties.BooleanProperty` and defaults\n    to False.\n    '
    margin_hint = ListProperty([0.05, 0.06, 0.05, 0.06])
    'Margin hint, used as spacing between keyboard background and keys\n    content. The margin is composed of four values, between 0 and 1::\n\n        margin_hint = [top, right, bottom, left]\n\n    The margin hints will be multiplied by width and height, according to their\n    position.\n\n    :attr:`margin_hint` is a :class:`~kivy.properties.ListProperty` and\n    defaults to [.05, .06, .05, .06]\n    '
    key_margin = ListProperty([2, 2, 2, 2])
    'Key margin, used to create space between keys. The margin is composed of\n    four values, in pixels::\n\n        key_margin = [top, right, bottom, left]\n\n    :attr:`key_margin` is a :class:`~kivy.properties.ListProperty` and defaults\n    to [2, 2, 2, 2]\n    '
    font_size = NumericProperty(20.0)
    'font_size, specifies the size of the text on the virtual keyboard keys.\n    It should be kept within limits to ensure the text does not extend beyond\n    the bounds of the key or become too small to read.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`font_size` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 20.\n    '
    background_color = ColorProperty([1, 1, 1, 1])
    'Background color, in the format (r, g, b, a). If a background is\n    set, the color will be combined with the background texture.\n\n    :attr:`background_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    background = StringProperty('atlas://data/images/defaulttheme/vkeyboard_background')
    'Filename of the background image.\n\n    :attr:`background` is a :class:`~kivy.properties.StringProperty` and\n    defaults to :file:`atlas://data/images/defaulttheme/vkeyboard_background`.\n    '
    background_disabled = StringProperty('atlas://data/images/defaulttheme/vkeyboard_disabled_background')
    'Filename of the background image when the vkeyboard is disabled.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`background_disabled` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    :file:`atlas://data/images/defaulttheme/vkeyboard__disabled_background`.\n\n    '
    key_background_color = ColorProperty([1, 1, 1, 1])
    'Key background color, in the format (r, g, b, a). If a key background is\n    set, the color will be combined with the key background texture.\n\n    :attr:`key_background_color` is a :class:`~kivy.properties.ColorProperty`\n    and defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    key_background_normal = StringProperty('atlas://data/images/defaulttheme/vkeyboard_key_normal')
    'Filename of the key background image for use when no touches are active\n    on the widget.\n\n    :attr:`key_background_normal` is a :class:`~kivy.properties.StringProperty`\n    and defaults to\n    :file:`atlas://data/images/defaulttheme/vkeyboard_key_normal`.\n    '
    key_disabled_background_normal = StringProperty('atlas://data/images/defaulttheme/vkeyboard_key_normal')
    'Filename of the key background image for use when no touches are active\n    on the widget and vkeyboard is disabled.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`key_disabled_background_normal` is a\n    :class:`~kivy.properties.StringProperty` and defaults to\n    :file:`atlas://data/images/defaulttheme/vkeyboard_disabled_key_normal`.\n\n    '
    key_background_down = StringProperty('atlas://data/images/defaulttheme/vkeyboard_key_down')
    'Filename of the key background image for use when a touch is active\n    on the widget.\n\n    :attr:`key_background_down` is a :class:`~kivy.properties.StringProperty`\n    and defaults to\n    :file:`atlas://data/images/defaulttheme/vkeyboard_key_down`.\n    '
    background_border = ListProperty([16, 16, 16, 16])
    'Background image border. Used for controlling the\n    :attr:`~kivy.graphics.vertex_instructions.BorderImage.border` property of\n    the background.\n\n    :attr:`background_border` is a :class:`~kivy.properties.ListProperty` and\n    defaults to [16, 16, 16, 16]\n    '
    key_border = ListProperty([8, 8, 8, 8])
    'Key image border. Used for controlling the\n    :attr:`~kivy.graphics.vertex_instructions.BorderImage.border` property of\n    the key.\n\n    :attr:`key_border` is a :class:`~kivy.properties.ListProperty` and\n    defaults to [16, 16, 16, 16]\n    '
    layout_mode = OptionProperty('normal', options=('normal', 'shift', 'special'))
    layout_geometry = DictProperty({})
    have_capslock = BooleanProperty(False)
    have_shift = BooleanProperty(False)
    have_special = BooleanProperty(False)
    active_keys = DictProperty({})
    font_name = StringProperty('data/fonts/DejaVuSans.ttf')
    repeat_touch = ObjectProperty(allownone=True)
    _start_repeat_key_ev = None
    _repeat_key_ev = None
    __events__ = ('on_key_down', 'on_key_up', 'on_textinput')

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'size_hint' not in kwargs:
            if 'size_hint_x' not in kwargs:
                self.size_hint_x = None
            if 'size_hint_y' not in kwargs:
                self.size_hint_y = None
        if 'size' not in kwargs:
            if 'width' not in kwargs:
                self.width = 700
            if 'height' not in kwargs:
                self.height = 200
        if 'scale_min' not in kwargs:
            self.scale_min = 0.4
        if 'scale_max' not in kwargs:
            self.scale_max = 1.6
        if 'docked' not in kwargs:
            self.docked = False
        layout_mode = self._trigger_update_layout_mode = Clock.create_trigger(self._update_layout_mode)
        layouts = self._trigger_load_layouts = Clock.create_trigger(self._load_layouts)
        layout = self._trigger_load_layout = Clock.create_trigger(self._load_layout)
        fbind = self.fbind
        fbind('docked', self.setup_mode)
        fbind('have_shift', layout_mode)
        fbind('have_capslock', layout_mode)
        fbind('have_special', layout_mode)
        fbind('layout_path', layouts)
        fbind('layout', layout)
        super(VKeyboard, self).__init__(**kwargs)
        self._load_layouts()
        available_layouts = self.available_layouts
        if not available_layouts:
            Logger.critical('VKeyboard: unable to load default layouts')
        if self.layout is None:
            self.layout = Config.get('kivy', 'keyboard_layout')
        else:
            self._trigger_load_layout()
        self._trigger_update_layout_mode()
        with self.canvas:
            self.background_key_layer = Canvas()
            self.active_keys_layer = Canvas()

    def on_disabled(self, instance, value):
        if False:
            print('Hello World!')
        self.refresh_keys()

    def _update_layout_mode(self, *l):
        if False:
            for i in range(10):
                print('nop')
        mode = self.have_capslock != self.have_shift
        mode = 'shift' if mode else 'normal'
        if self.have_special:
            mode = 'special'
        if mode != self.layout_mode:
            self.layout_mode = mode
            self.refresh(False)

    def _load_layout(self, *largs):
        if False:
            i = 10
            return i + 15
        if self._trigger_load_layouts.is_triggered:
            self._load_layouts()
            self._trigger_load_layouts.cancel()
        value = self.layout
        available_layouts = self.available_layouts
        if self.layout[-5:] == '.json':
            if value not in available_layouts:
                fn = resource_find(self.layout)
                self._load_layout_fn(fn, self.layout)
        if not available_layouts:
            return
        if value not in available_layouts and value != 'qwerty':
            Logger.error('Vkeyboard: <%s> keyboard layout mentioned in conf file was not found, fallback on qwerty' % value)
            self.layout = 'qwerty'
        self.refresh(True)

    def _load_layouts(self, *largs):
        if False:
            print('Hello World!')
        value = self.layout_path
        for fn in listdir(value):
            self._load_layout_fn(join(value, fn), basename(splitext(fn)[0]))

    def _load_layout_fn(self, fn, name):
        if False:
            while True:
                i = 10
        available_layouts = self.available_layouts
        if fn[-5:] != '.json':
            return
        with open(fn, 'r', encoding='utf-8') as fd:
            json_content = fd.read()
            layout = loads(json_content)
        available_layouts[name] = layout

    def setup_mode(self, *largs):
        if False:
            print('Hello World!')
        'Call this method when you want to readjust the keyboard according to\n        options: :attr:`docked` or not, with attached :attr:`target` or not:\n\n        * If :attr:`docked` is True, it will call :meth:`setup_mode_dock`\n        * If :attr:`docked` is False, it will call :meth:`setup_mode_free`\n\n        Feel free to overload these methods to create new\n        positioning behavior.\n        '
        if self.docked:
            self.setup_mode_dock()
        else:
            self.setup_mode_free()

    def setup_mode_dock(self, *largs):
        if False:
            return 10
        "Setup the keyboard in docked mode.\n\n        Dock mode will reset the rotation, disable translation, rotation and\n        scale. Scale and position will be automatically adjusted to attach the\n        keyboard to the bottom of the screen.\n\n        .. note::\n            Don't call this method directly, use :meth:`setup_mode` instead.\n        "
        self.do_translation = False
        self.do_rotation = False
        self.do_scale = False
        self.rotation = 0
        win = self.get_parent_window()
        scale = win.width / float(self.width)
        self.scale = scale
        self.pos = (0, 0)
        win.bind(on_resize=self._update_dock_mode)

    def _update_dock_mode(self, win, *largs):
        if False:
            return 10
        scale = win.width / float(self.width)
        self.scale = scale
        self.pos = (0, 0)

    def setup_mode_free(self):
        if False:
            i = 10
            return i + 15
        "Setup the keyboard in free mode.\n\n        Free mode is designed to let the user control the position and\n        orientation of the keyboard. The only real usage is for a multiuser\n        environment, but you might found other ways to use it.\n        If a :attr:`target` is set, it will place the vkeyboard under the\n        target.\n\n        .. note::\n            Don't call this method directly, use :meth:`setup_mode` instead.\n        "
        self.do_translation = True
        self.do_rotation = True
        self.do_scale = True
        target = self.target
        if not target:
            return
        a = Vector(1, 0)
        b = Vector(target.to_window(0, 0))
        c = Vector(target.to_window(1, 0)) - b
        self.rotation = -a.angle(c)
        dpos = Vector(self.to_window(self.width / 2.0, self.height))
        cpos = Vector(target.to_window(target.center_x, target.y))
        diff = dpos - cpos
        diff2 = Vector(self.x + self.width / 2.0, self.y + self.height) - Vector(self.to_parent(self.width / 2.0, self.height))
        diff -= diff2
        self.pos = -diff

    def change_layout(self):
        if False:
            return 10
        pass

    def refresh(self, force=False):
        if False:
            i = 10
            return i + 15
        '(internal) Recreate the entire widget and graphics according to the\n        selected layout.\n        '
        self.clear_widgets()
        if force:
            self.refresh_keys_hint()
        self.refresh_keys()
        self.refresh_active_keys_layer()

    def refresh_active_keys_layer(self):
        if False:
            print('Hello World!')
        self.active_keys_layer.clear()
        active_keys = self.active_keys
        layout_geometry = self.layout_geometry
        background = resource_find(self.key_background_down)
        texture = Image(background, mipmap=True).texture
        with self.active_keys_layer:
            Color(*self.key_background_color)
            for (line_nb, index) in active_keys.values():
                (pos, size) = layout_geometry['LINE_%d' % line_nb][index]
                BorderImage(texture=texture, pos=pos, size=size, border=self.key_border)

    def refresh_keys_hint(self):
        if False:
            return 10
        layout = self.available_layouts[self.layout]
        layout_cols = layout['cols']
        layout_rows = layout['rows']
        layout_geometry = self.layout_geometry
        (mtop, mright, mbottom, mleft) = self.margin_hint
        el_hint = 1.0 - mleft - mright
        eh_hint = 1.0 - mtop - mbottom
        ex_hint = 0 + mleft
        ey_hint = 0 + mbottom
        uw_hint = 1.0 / layout_cols * el_hint
        uh_hint = 1.0 / layout_rows * eh_hint
        layout_geometry['U_HINT'] = (uw_hint, uh_hint)
        current_y_hint = ey_hint + eh_hint
        for line_nb in range(1, layout_rows + 1):
            current_y_hint -= uh_hint
            line_name = '%s_%d' % (self.layout_mode, line_nb)
            line_hint = 'LINE_HINT_%d' % line_nb
            layout_geometry[line_hint] = []
            current_x_hint = ex_hint
            for key in layout[line_name]:
                layout_geometry[line_hint].append([(current_x_hint, current_y_hint), (key[3] * uw_hint, uh_hint)])
                current_x_hint += key[3] * uw_hint
        self.layout_geometry = layout_geometry

    def refresh_keys(self):
        if False:
            for i in range(10):
                print('nop')
        layout = self.available_layouts[self.layout]
        layout_rows = layout['rows']
        layout_geometry = self.layout_geometry
        (w, h) = self.size
        (kmtop, kmright, kmbottom, kmleft) = self.key_margin
        (uw_hint, uh_hint) = layout_geometry['U_HINT']
        for line_nb in range(1, layout_rows + 1):
            llg = layout_geometry['LINE_%d' % line_nb] = []
            llg_append = llg.append
            for key in layout_geometry['LINE_HINT_%d' % line_nb]:
                (x_hint, y_hint) = key[0]
                (w_hint, h_hint) = key[1]
                kx = x_hint * w
                ky = y_hint * h
                kw = w_hint * w
                kh = h_hint * h
                kx = int(kx + kmleft)
                ky = int(ky + kmbottom)
                kw = int(kw - kmleft - kmright)
                kh = int(kh - kmbottom - kmtop)
                pos = (kx, ky)
                size = (kw, kh)
                llg_append((pos, size))
        self.layout_geometry = layout_geometry
        self.draw_keys()

    def draw_keys(self):
        if False:
            while True:
                i = 10
        layout = self.available_layouts[self.layout]
        layout_rows = layout['rows']
        layout_geometry = self.layout_geometry
        layout_mode = self.layout_mode
        background = resource_find(self.background_disabled if self.disabled else self.background)
        texture = Image(background, mipmap=True).texture
        self.background_key_layer.clear()
        with self.background_key_layer:
            Color(*self.background_color)
            BorderImage(texture=texture, size=self.size, border=self.background_border)
        key_normal = resource_find(self.key_background_disabled_normal if self.disabled else self.key_background_normal)
        texture = Image(key_normal, mipmap=True).texture
        with self.background_key_layer:
            Color(*self.key_background_color)
            for line_nb in range(1, layout_rows + 1):
                for (pos, size) in layout_geometry['LINE_%d' % line_nb]:
                    BorderImage(texture=texture, pos=pos, size=size, border=self.key_border)
        for line_nb in range(1, layout_rows + 1):
            key_nb = 0
            for (pos, size) in layout_geometry['LINE_%d' % line_nb]:
                text = layout[layout_mode + '_' + str(line_nb)][key_nb][0]
                z = Label(text=text, font_size=self.font_size, pos=pos, size=size, font_name=self.font_name)
                self.add_widget(z)
                key_nb += 1

    def on_key_down(self, *largs):
        if False:
            i = 10
            return i + 15
        pass

    def on_key_up(self, *largs):
        if False:
            while True:
                i = 10
        pass

    def on_textinput(self, *largs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_key_at_pos(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (w, h) = self.size
        x_hint = x / w
        layout_geometry = self.layout_geometry
        layout = self.available_layouts[self.layout]
        layout_rows = layout['rows']
        (mtop, mright, mbottom, mleft) = self.margin_hint
        e_height = h - (mbottom + mtop) * h
        line_height = e_height / layout_rows
        y = y - mbottom * h
        line_nb = layout_rows - int(y / line_height)
        if line_nb > layout_rows:
            line_nb = layout_rows
        if line_nb < 1:
            line_nb = 1
        key_index = ''
        current_key_index = 0
        for key in layout_geometry['LINE_HINT_%d' % line_nb]:
            if x_hint >= key[0][0] and x_hint < key[0][0] + key[1][0]:
                key_index = current_key_index
                break
            else:
                current_key_index += 1
        if key_index == '':
            return None
        key = layout['%s_%d' % (self.layout_mode, line_nb)][key_index]
        return [key, (line_nb, key_index)]

    def collide_margin(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        'Do a collision test, and return True if the (x, y) is inside the\n        vkeyboard margin.\n        '
        (mtop, mright, mbottom, mleft) = self.margin_hint
        x_hint = x / self.width
        y_hint = y / self.height
        if x_hint > mleft and x_hint < 1.0 - mright and (y_hint > mbottom) and (y_hint < 1.0 - mtop):
            return False
        return True

    def process_key_on(self, touch):
        if False:
            print('Hello World!')
        if not touch:
            return
        (x, y) = self.to_local(*touch.pos)
        key = self.get_key_at_pos(x, y)
        if not key:
            return
        key_data = key[0]
        (displayed_char, internal, special_char, size) = key_data
        (line_nb, key_index) = key[1]
        ud = touch.ud[self.uid] = {}
        ud['key'] = key
        uid = touch.uid
        if special_char is not None:
            if special_char in ('capslock', 'shift', 'layout', 'special'):
                if self._start_repeat_key_ev is not None:
                    self._start_repeat_key_ev.cancel()
                    self._start_repeat_key_ev = None
                self.repeat_touch = None
            if special_char == 'capslock':
                self.have_capslock = not self.have_capslock
                uid = -1
            elif special_char == 'shift':
                self.have_shift = True
            elif special_char == 'special':
                self.have_special = True
            elif special_char == 'layout':
                self.change_layout()
        b_keycode = special_char
        b_modifiers = self._get_modifiers()
        if self.get_parent_window().__class__.__module__ == 'kivy.core.window.window_sdl2' and internal:
            self.dispatch('on_textinput', internal)
        else:
            self.dispatch('on_key_down', b_keycode, internal, b_modifiers)
        self.active_keys[uid] = key[1]
        self.refresh_active_keys_layer()

    def process_key_up(self, touch):
        if False:
            print('Hello World!')
        uid = touch.uid
        if self.uid not in touch.ud:
            return
        (key_data, key) = touch.ud[self.uid]['key']
        (displayed_char, internal, special_char, size) = key_data
        b_keycode = special_char
        b_modifiers = self._get_modifiers()
        self.dispatch('on_key_up', b_keycode, internal, b_modifiers)
        if special_char == 'capslock':
            uid = -1
        if uid in self.active_keys:
            self.active_keys.pop(uid, None)
            if special_char == 'shift':
                self.have_shift = False
            elif special_char == 'special':
                self.have_special = False
            if special_char == 'capslock' and self.have_capslock:
                self.active_keys[-1] = key
            self.refresh_active_keys_layer()

    def _get_modifiers(self):
        if False:
            print('Hello World!')
        ret = []
        if self.have_shift:
            ret.append('shift')
        if self.have_capslock:
            ret.append('capslock')
        return ret

    def _start_repeat_key(self, *kwargs):
        if False:
            while True:
                i = 10
        self._repeat_key_ev = Clock.schedule_interval(self._repeat_key, 0.05)

    def _repeat_key(self, *kwargs):
        if False:
            print('Hello World!')
        self.process_key_on(self.repeat_touch)

    def on_touch_down(self, touch):
        if False:
            while True:
                i = 10
        (x, y) = touch.pos
        if not self.collide_point(x, y):
            return
        if self.disabled:
            return True
        (x, y) = self.to_local(x, y)
        if not self.collide_margin(x, y):
            if self.repeat_touch is None:
                self._start_repeat_key_ev = Clock.schedule_once(self._start_repeat_key, 0.5)
            self.repeat_touch = touch
            self.process_key_on(touch)
            touch.grab(self, exclusive=True)
        else:
            super(VKeyboard, self).on_touch_down(touch)
        return True

    def on_touch_up(self, touch):
        if False:
            i = 10
            return i + 15
        if touch.grab_current is self:
            self.process_key_up(touch)
            if self._start_repeat_key_ev is not None:
                self._start_repeat_key_ev.cancel()
                self._start_repeat_key_ev = None
            if touch == self.repeat_touch:
                if self._repeat_key_ev is not None:
                    self._repeat_key_ev.cancel()
                    self._repeat_key_ev = None
                self.repeat_touch = None
        return super(VKeyboard, self).on_touch_up(touch)
if __name__ == '__main__':
    from kivy.base import runTouchApp
    vk = VKeyboard(layout='azerty')
    runTouchApp(vk)