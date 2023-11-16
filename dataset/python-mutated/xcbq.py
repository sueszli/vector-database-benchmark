"""
    A minimal EWMH-aware OO layer over xcffib. This is NOT intended to be
    complete - it only implements the subset of functionalty needed by qtile.
"""
from __future__ import annotations
import contextlib
import functools
import operator
from itertools import chain, repeat
import cairocffi
import cairocffi.pixbuf
import cairocffi.xcb
import xcffib
import xcffib.randr
import xcffib.xinerama
import xcffib.xproto
from xcffib.xfixes import SelectionEventMask
from xcffib.xproto import CW, EventMask, WindowClass
from libqtile.backend.x11 import window
from libqtile.backend.x11.xcursors import Cursors
from libqtile.backend.x11.xkeysyms import keysyms
from libqtile.log_utils import logger
from libqtile.utils import QtileError, hex

class XCBQError(QtileError):
    pass
ModMasks = {'shift': 1 << 0, 'lock': 1 << 1, 'control': 1 << 2, 'mod1': 1 << 3, 'mod2': 1 << 4, 'mod3': 1 << 5, 'mod4': 1 << 6, 'mod5': 1 << 7}
AllButtonsMask = 31 << 8
ButtonMotionMask = 1 << 13
ButtonReleaseMask = 1 << 3
PointerMotionHintMask = 1 << 7
NormalHintsFlags = {'USPosition': 1, 'USSize': 2, 'PPosition': 4, 'PSize': 8, 'PMinSize': 16, 'PMaxSize': 32, 'PResizeInc': 64, 'PAspect': 128, 'PBaseSize': 256, 'PWinGravity': 512}
HintsFlags = {'InputHint': 1, 'StateHint': 2, 'IconPixmapHint': 4, 'IconWindowHint': 8, 'IconPositionHint': 16, 'IconMaskHint': 32, 'WindowGroupHint': 64, 'MessageHint': 128, 'UrgencyHint': 256}
WindowTypes = {'_NET_WM_WINDOW_TYPE_DESKTOP': 'desktop', '_NET_WM_WINDOW_TYPE_DOCK': 'dock', '_NET_WM_WINDOW_TYPE_TOOLBAR': 'toolbar', '_NET_WM_WINDOW_TYPE_MENU': 'menu', '_NET_WM_WINDOW_TYPE_UTILITY': 'utility', '_NET_WM_WINDOW_TYPE_SPLASH': 'splash', '_NET_WM_WINDOW_TYPE_DIALOG': 'dialog', '_NET_WM_WINDOW_TYPE_DROPDOWN_MENU': 'dropdown', '_NET_WM_WINDOW_TYPE_POPUP_MENU': 'menu', '_NET_WM_WINDOW_TYPE_TOOLTIP': 'tooltip', '_NET_WM_WINDOW_TYPE_NOTIFICATION': 'notification', '_NET_WM_WINDOW_TYPE_COMBO': 'combo', '_NET_WM_WINDOW_TYPE_DND': 'dnd', '_NET_WM_WINDOW_TYPE_NORMAL': 'normal'}
net_wm_states = ('_NET_WM_STATE_MODAL', '_NET_WM_STATE_STICKY', '_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ', '_NET_WM_STATE_SHADED', '_NET_WM_STATE_SKIP_TASKBAR', '_NET_WM_STATE_SKIP_PAGER', '_NET_WM_STATE_HIDDEN', '_NET_WM_STATE_FULLSCREEN', '_NET_WM_STATE_ABOVE', '_NET_WM_STATE_BELOW', '_NET_WM_STATE_DEMANDS_ATTENTION', '_NET_WM_STATE_FOCUSED')
WindowStates = {None: 'normal', '_NET_WM_STATE_FULLSCREEN': 'fullscreen', '_NET_WM_STATE_DEMANDS_ATTENTION': 'urgent'}
PropertyMap = {'_NET_DESKTOP_GEOMETRY': ('CARDINAL', 32), '_NET_SUPPORTED': ('ATOM', 32), '_NET_SUPPORTING_WM_CHECK': ('WINDOW', 32), '_NET_WM_NAME': ('UTF8_STRING', 8), '_NET_WM_PID': ('CARDINAL', 32), '_NET_CLIENT_LIST': ('WINDOW', 32), '_NET_CLIENT_LIST_STACKING': ('WINDOW', 32), '_NET_NUMBER_OF_DESKTOPS': ('CARDINAL', 32), '_NET_CURRENT_DESKTOP': ('CARDINAL', 32), '_NET_DESKTOP_NAMES': ('UTF8_STRING', 8), '_NET_DESKTOP_VIEWPORT': ('CARDINAL', 32), '_NET_WORKAREA': ('CARDINAL', 32), '_NET_ACTIVE_WINDOW': ('WINDOW', 32), '_NET_WM_DESKTOP': ('CARDINAL', 32), '_NET_WM_STRUT': ('CARDINAL', 32), '_NET_WM_STRUT_PARTIAL': ('CARDINAL', 32), '_NET_WM_WINDOW_OPACITY': ('CARDINAL', 32), '_NET_WM_WINDOW_TYPE': ('ATOM', 32), '_NET_FRAME_EXTENTS': ('CARDINAL', 32), '_NET_WM_STATE': ('ATOM', 32), '_XEMBED_INFO': ('_XEMBED_INFO', 32), 'WM_STATE': ('WM_STATE', 32), 'QTILE_INTERNAL': ('CARDINAL', 32)}
for _name in net_wm_states:
    PropertyMap[_name] = ('ATOM', 32)
SUPPORTED_ATOMS = ['_NET_SUPPORTED', '_NET_CLIENT_LIST', '_NET_CLIENT_LIST_STACKING', '_NET_CURRENT_DESKTOP', '_NET_DESKTOP_VIEWPORT', '_NET_ACTIVE_WINDOW', '_NET_SUPPORTING_WM_CHECK', '_NET_WM_NAME', '_NET_WM_VISIBLE_NAME', '_NET_WM_ICON_NAME', '_NET_WM_DESKTOP', '_NET_WM_WINDOW_TYPE', '_NET_WM_STATE', '_NET_WM_STRUT_PARTIAL', '_NET_WM_PID']
SUPPORTED_ATOMS.extend(WindowTypes.keys())
SUPPORTED_ATOMS.extend(net_wm_states)
XCB_CONN_ERRORS = {1: 'XCB_CONN_ERROR', 2: 'XCB_CONN_CLOSED_EXT_NOTSUPPORTED', 3: 'XCB_CONN_CLOSED_MEM_INSUFFICIENT', 4: 'XCB_CONN_CLOSED_REQ_LEN_EXCEED', 5: 'XCB_CONN_CLOSED_PARSE_ERR', 6: 'XCB_CONN_CLOSED_INVALID_SCREEN', 7: 'XCB_CONN_CLOSED_FDPASSING_FAILED'}
XCB_KEY_PRESS = 2
XCB_KEY_RELEASE = 3
XCB_BUTTON_PRESS = 4
XCB_BUTTON_RELEASE = 5
XCB_MOTION_NOTIFY = 6

class MaskMap:
    """
    A general utility class that encapsulates the way the bitmask/listofvalue idiom
    works in X protocol. It understands a special attribute _maskvalue on
    objects, which will be used instead of the object value if present.
    This lets us pass in a Font object, rather than Font.fid, for example.
    """

    def __init__(self, obj):
        if False:
            print('Hello World!')
        self.mmap = []
        for i in dir(obj):
            if not i.startswith('_'):
                self.mmap.append((getattr(obj, i), i.lower()))
        self.mmap.sort()

    def __call__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        kwargs: keys should be in the mmap name set\n\n        Returns a (mask, values) tuple.\n        '
        mask = 0
        values = []
        for (m, s) in self.mmap:
            if s in kwargs:
                val = kwargs.get(s)
                if val is not None:
                    mask |= m
                    values.append(getattr(val, '_maskvalue', val))
                del kwargs[s]
        if kwargs:
            raise ValueError('Unknown mask names: %s' % list(kwargs.keys()))
        return (mask, values)
ConfigureMasks = MaskMap(xcffib.xproto.ConfigWindow)
AttributeMasks = MaskMap(CW)

class AtomCache:

    def __init__(self, conn):
        if False:
            while True:
                i = 10
        self.conn = conn
        self.atoms = {}
        self.reverse = {}
        for name in WindowTypes.keys():
            self.insert(name=name)
        for i in dir(xcffib.xproto.Atom):
            if not i.startswith('_'):
                self.insert(name=i, atom=getattr(xcffib.xproto.Atom, i))

    def insert(self, name=None, atom=None):
        if False:
            i = 10
            return i + 15
        assert name or atom
        if atom is None:
            c = self.conn.conn.core.InternAtom(False, len(name), name)
            atom = c.reply().atom
        if name is None:
            c = self.conn.conn.core.GetAtomName(atom)
            name = c.reply().name.to_string()
        self.atoms[name] = atom
        self.reverse[atom] = name

    def get_name(self, atom):
        if False:
            i = 10
            return i + 15
        if atom not in self.reverse:
            self.insert(atom=atom)
        return self.reverse[atom]

    def __getitem__(self, key):
        if False:
            return 10
        if key not in self.atoms:
            self.insert(name=key)
        return self.atoms[key]

class _Wrapper:

    def __init__(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        self.wrapped = wrapped

    def __getattr__(self, x):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.wrapped, x)

class Screen(_Wrapper):
    """
    This represents an actual X screen.
    """

    def __init__(self, conn, screen):
        if False:
            i = 10
            return i + 15
        _Wrapper.__init__(self, screen)
        self.default_colormap = Colormap(conn, screen.default_colormap)
        self.root = window.XWindow(conn, self.root)
        self._visuals = {}
        for d in [32, 24, self.root_depth]:
            if d not in self._visuals:
                visual = self.get_visual_for_depth(self, d)
                if visual:
                    self._visuals[d] = visual

    def _get_depth_and_visual(self, desired_depth):
        if False:
            return 10
        '\n        Returns a tuple of (depth, visual) for the requested\n        depth.\n\n        Falls back to the root depth and visual if the requested\n        depth is unavailable.\n        '
        if desired_depth in self._visuals:
            return (desired_depth, self._visuals[desired_depth])
        logger.info('%s bit colour depth not available. Falling back to root depth: %s.', desired_depth, self.root_depth)
        return (self.root_depth, self._visuals[self.root_depth])

    @staticmethod
    def get_visual_for_depth(screen, depth):
        if False:
            while True:
                i = 10
        '\n        Returns the visual object of the screen @ some depth\n\n        For an ARGB visual -> depth=32\n        For a RGB visual   -> depth=24\n        '
        allowed = screen.allowed_depths
        if depth not in [x.depth for x in allowed]:
            logger.warning('Unsupported colour depth: %s', depth)
            return
        for i in allowed:
            if i.depth == depth:
                if i.visuals:
                    return i.visuals[0]

class PseudoScreen:
    """
    This may be a Xinerama screen or a RandR CRTC, both of which are
    rectangular sections of an actual Screen.
    """

    def __init__(self, conn, x, y, width, height):
        if False:
            while True:
                i = 10
        self.conn = conn
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class Colormap:

    def __init__(self, conn, cid):
        if False:
            i = 10
            return i + 15
        self.conn = conn
        self.cid = cid

    def alloc_color(self, color):
        if False:
            print('Hello World!')
        '\n        Flexible color allocation.\n        '
        try:
            return self.conn.conn.core.AllocNamedColor(self.cid, len(color), color).reply()
        except xcffib.xproto.NameError:

            def x8to16(i):
                if False:
                    i = 10
                    return i + 15
                return 65535 * (i & 255) // 255
            try:
                color = hex(color)
            except ValueError:
                logger.error('Colormap failed to allocate %s', color)
                color = '#ff0000'
            r = x8to16(int(color[-6] + color[-5], 16))
            g = x8to16(int(color[-4] + color[-3], 16))
            b = x8to16(int(color[-2] + color[-1], 16))
            return self.conn.conn.core.AllocColor(self.cid, r, g, b).reply()

class Xinerama:

    def __init__(self, conn):
        if False:
            print('Hello World!')
        self.ext = conn.conn(xcffib.xinerama.key)

    def query_screens(self):
        if False:
            return 10
        r = self.ext.QueryScreens().reply()
        return r.screen_info

class RandR:

    def __init__(self, conn):
        if False:
            for i in range(10):
                print('nop')
        self.ext = conn.conn(xcffib.randr.key)
        self.ext.SelectInput(conn.default_screen.root.wid, xcffib.randr.NotifyMask.ScreenChange)

    def query_crtcs(self, root):
        if False:
            for i in range(10):
                print('nop')
        crtc_list = []
        for crtc in self.ext.GetScreenResources(root).reply().crtcs:
            crtc_info = self.ext.GetCrtcInfo(crtc, xcffib.CurrentTime).reply()
            crtc_dict = {'x': crtc_info.x, 'y': crtc_info.y, 'width': crtc_info.width, 'height': crtc_info.height}
            crtc_list.append(crtc_dict)
        return crtc_list

class XFixes:
    selection_mask = SelectionEventMask.SetSelectionOwner | SelectionEventMask.SelectionClientClose | SelectionEventMask.SelectionWindowDestroy

    def __init__(self, conn):
        if False:
            i = 10
            return i + 15
        self.conn = conn
        self.ext = conn.conn(xcffib.xfixes.key)
        self.ext.QueryVersion(xcffib.xfixes.MAJOR_VERSION, xcffib.xfixes.MINOR_VERSION)

    def select_selection_input(self, window, selection='PRIMARY'):
        if False:
            print('Hello World!')
        _selection = self.conn.atoms[selection]
        self.conn.xfixes.ext.SelectSelectionInput(window.wid, _selection, self.selection_mask)

class Connection:
    _extmap = {'xinerama': Xinerama, 'randr': RandR, 'xfixes': XFixes}

    def __init__(self, display):
        if False:
            print('Hello World!')
        self.conn = xcffib.connect(display=display)
        self._connected = True
        self.cursors = Cursors(self)
        self.setup = self.conn.get_setup()
        extensions = self.extensions()
        self.screens = [Screen(self, i) for i in self.setup.roots]
        self.default_screen = self.screens[self.conn.pref_screen]
        for i in extensions:
            if i in self._extmap:
                setattr(self, i, self._extmap[i](self))
        self.atoms = AtomCache(self)
        self.code_to_syms = {}
        self.sym_to_codes = None
        self.refresh_keymap()
        self.modmap = None
        self.refresh_modmap()
        self._cmaps = {}

    def colormap(self, desired_depth):
        if False:
            for i in range(10):
                print('nop')
        if desired_depth in self._cmaps:
            return self._cmaps[desired_depth]
        (_, visual) = self.default_screen._get_depth_and_visual(desired_depth)
        cmap = self.conn.generate_id()
        self.conn.core.CreateColormap(xcffib.xproto.ColormapAlloc._None, cmap, self.default_screen.root.wid, visual.visual_id, is_checked=True).check()
        self._cmaps[desired_depth] = cmap
        return cmap

    @property
    def pseudoscreens(self):
        if False:
            print('Hello World!')
        pseudoscreens = []
        if hasattr(self, 'xinerama'):
            for (i, s) in enumerate(self.xinerama.query_screens()):
                scr = PseudoScreen(self, s.x_org, s.y_org, s.width, s.height)
                pseudoscreens.append(scr)
        elif hasattr(self, 'randr'):
            for i in self.randr.query_crtcs(self.screens[0].root.wid):
                scr = PseudoScreen(self, i['x'], i['y'], i['width'], i['height'])
                pseudoscreens.append(scr)
        return pseudoscreens

    def finalize(self):
        if False:
            return 10
        self.cursors.finalize()
        self.disconnect()

    def refresh_keymap(self, first=None, count=None):
        if False:
            while True:
                i = 10
        if first is None:
            first = self.setup.min_keycode
            count = self.setup.max_keycode - self.setup.min_keycode + 1
        q = self.conn.core.GetKeyboardMapping(first, count).reply()
        assert len(q.keysyms) % q.keysyms_per_keycode == 0
        for i in range(len(q.keysyms) // q.keysyms_per_keycode):
            self.code_to_syms[first + i] = q.keysyms[i * q.keysyms_per_keycode:(i + 1) * q.keysyms_per_keycode]
        sym_to_codes = {}
        for (k, s) in self.code_to_syms.items():
            for sym in s:
                if sym == 0:
                    continue
                if sym not in sym_to_codes:
                    sym_to_codes[sym] = [k]
                elif k not in sym_to_codes[sym]:
                    sym_to_codes[sym].append(k)
        self.sym_to_codes = sym_to_codes

    def refresh_modmap(self):
        if False:
            i = 10
            return i + 15
        reply = self.conn.core.GetModifierMapping().reply()
        modmap = {}
        names = (repeat(name, reply.keycodes_per_modifier) for name in ModMasks)
        for (name, keycode) in zip(chain.from_iterable(names), reply.keycodes):
            value = modmap.setdefault(name, [])
            value.append(keycode)
        self.modmap = modmap

    def get_modifier(self, keycode):
        if False:
            return 10
        'Return the modifier matching keycode'
        for (n, l) in self.modmap.items():
            if keycode in l:
                return n
        return None

    def keysym_to_keycode(self, keysym):
        if False:
            return 10
        return self.sym_to_codes.get(keysym, [0])

    def keycode_to_keysym(self, keycode, modifier):
        if False:
            while True:
                i = 10
        if keycode >= len(self.code_to_syms) or modifier >= len(self.code_to_syms[keycode]):
            return 0
        return self.code_to_syms[keycode][modifier]

    def create_window(self, x, y, width, height, desired_depth=32):
        if False:
            return 10
        (depth, visual) = self.default_screen._get_depth_and_visual(desired_depth)
        wid = self.conn.generate_id()
        value_mask = CW.BackPixmap | CW.BorderPixel | CW.EventMask | CW.Colormap
        values = [xcffib.xproto.BackPixmap._None, 0, EventMask.StructureNotify | EventMask.Exposure, self.colormap(depth)]
        self.conn.core.CreateWindow(depth, wid, self.default_screen.root.wid, x, y, width, height, 0, WindowClass.InputOutput, visual.visual_id, value_mask, values)
        return window.XWindow(self, wid)

    def disconnect(self):
        if False:
            while True:
                i = 10
        with contextlib.suppress(xcffib.ConnectionException):
            self.conn.disconnect()
        self._connected = False

    def flush(self):
        if False:
            while True:
                i = 10
        if self._connected:
            return self.conn.flush()

    def xsync(self):
        if False:
            while True:
                i = 10
        self.conn.core.GetInputFocus().reply()

    def get_setup(self):
        if False:
            print('Hello World!')
        return self.conn.get_setup()

    def extensions(self):
        if False:
            i = 10
            return i + 15
        return set((i.name.to_string().lower() for i in self.conn.core.ListExtensions().reply().names))

    def fixup_focus(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the X11 focus is set to None, all keypress events are discarded,\n        which makes our hotkeys not work. This fixes up the focus so it is not\n        None.\n        '
        window = self.conn.core.GetInputFocus().reply().focus
        if window == xcffib.xproto.InputFocus._None:
            self.conn.core.SetInputFocus(xcffib.xproto.InputFocus.PointerRoot, xcffib.xproto.InputFocus.PointerRoot, xcffib.xproto.Time.CurrentTime)

    @functools.lru_cache()
    def color_pixel(self, name):
        if False:
            return 10
        pixel = self.screens[0].default_colormap.alloc_color(name).pixel
        return pixel | 255 << 24

class Painter:

    def __init__(self, display):
        if False:
            while True:
                i = 10
        self.conn = xcffib.connect(display=display)
        self.setup = self.conn.get_setup()
        self.screens = [Screen(self, i) for i in self.setup.roots]
        self.default_screen = self.screens[self.conn.pref_screen]
        self.conn.core.SetCloseDownMode(xcffib.xproto.CloseDown.RetainPermanent)
        self.atoms = AtomCache(self)
        self.width = -1
        self.height = -1

    def paint(self, screen, image_path, mode=None):
        if False:
            return 10
        try:
            with open(image_path, 'rb') as f:
                (image, _) = cairocffi.pixbuf.decode_to_image_surface(f.read())
        except IOError:
            logger.exception('Could not load wallpaper:')
            return
        root_windows = screen.qtile.core.conn.pseudoscreens
        width = max((win.x + win.width for win in root_windows))
        height = max((win.y + win.height for win in root_windows))
        try:
            root_pixmap = self.default_screen.root.get_property('_XROOTPMAP_ID', xcffib.xproto.Atom.PIXMAP, int)
        except xcffib.ConnectionException:
            root_pixmap = None
        if not root_pixmap:
            root_pixmap = self.default_screen.root.get_property('ESETROOT_PMAP_ID', xcffib.xproto.Atom.PIXMAP, int)
        if root_pixmap and (self.width == width and self.height == height):
            root_pixmap = root_pixmap[0]
        else:
            self.width = width
            self.height = height
            root_pixmap = self.conn.generate_id()
            self.conn.core.CreatePixmap(self.default_screen.root_depth, root_pixmap, self.default_screen.root.wid, self.width, self.height)
        for depth in self.default_screen.allowed_depths:
            for visual in depth.visuals:
                if visual.visual_id == self.default_screen.root_visual:
                    root_visual = visual
                    break
        surface = cairocffi.xcb.XCBSurface(self.conn, root_pixmap, root_visual, self.width, self.height)
        context = cairocffi.Context(surface)
        with context:
            context.translate(screen.x, screen.y)
            if mode == 'fill':
                context.rectangle(0, 0, screen.width, screen.height)
                context.clip()
                image_w = image.get_width()
                image_h = image.get_height()
                width_ratio = screen.width / image_w
                if width_ratio * image_h >= screen.height:
                    context.scale(width_ratio)
                else:
                    height_ratio = screen.height / image_h
                    context.translate(-(image_w * height_ratio - screen.width) // 2, 0)
                    context.scale(height_ratio)
            elif mode == 'stretch':
                context.scale(sx=screen.width / image.get_width(), sy=screen.height / image.get_height())
            context.set_source_surface(image)
            context.paint()
        self.conn.core.ChangeProperty(xcffib.xproto.PropMode.Replace, self.default_screen.root.wid, self.atoms['_XROOTPMAP_ID'], xcffib.xproto.Atom.PIXMAP, 32, 1, [root_pixmap])
        self.conn.core.ChangeProperty(xcffib.xproto.PropMode.Replace, self.default_screen.root.wid, self.atoms['ESETROOT_PMAP_ID'], xcffib.xproto.Atom.PIXMAP, 32, 1, [root_pixmap])
        self.conn.core.ChangeWindowAttributes(self.default_screen.root.wid, CW.BackPixmap, [root_pixmap])
        self.conn.core.ClearArea(0, self.default_screen.root.wid, 0, 0, self.width, self.height)
        self.conn.flush()

    def __del__(self):
        if False:
            while True:
                i = 10
        self.conn.disconnect()

def get_keysym(key: str) -> int:
    if False:
        i = 10
        return i + 15
    keysym = keysyms.get(key.lower())
    if not keysym:
        raise XCBQError('Unknown key: %s' % key)
    return keysym

def translate_modifiers(mask: int) -> list[str]:
    if False:
        print('Hello World!')
    r = []
    for (k, v) in ModMasks.items():
        if mask & v:
            r.append(k)
    return r

def translate_masks(modifiers: list[str]) -> int:
    if False:
        return 10
    '\n    Translate a modifier mask specified as a list of strings into an or-ed\n    bit representation.\n    '
    masks = []
    for i in modifiers:
        try:
            masks.append(ModMasks[i.lower()])
        except KeyError as e:
            raise XCBQError('Unknown modifier: %s' % i) from e
    if masks:
        return functools.reduce(operator.or_, masks)
    else:
        return 0