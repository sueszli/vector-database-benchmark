"""Basic wrapping of Windows controls"""
from __future__ import unicode_literals
from __future__ import print_function
import copy
import time
import re
import ctypes
import win32api
import win32gui
import win32con
import win32process
import win32event
import six
import pywintypes
import warnings
from ..windows import win32defines, win32functions, win32structures
from .. import controlproperties
from ..actionlogger import ActionLogger
from .. import keyboard
from .. import mouse
from ..timings import Timings
from .. import timings
from .. import handleprops
from ..windows.win32_element_info import HwndElementInfo
from .. import backend
from .. import WindowNotFoundError
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
from .menuwrapper import Menu
from .win_base_wrapper import WinBaseWrapper
from ..base_wrapper import BaseMeta
from .. import deprecated

class ControlNotEnabled(RuntimeError):
    """Raised when a control is not enabled"""
    pass

class ControlNotVisible(RuntimeError):
    """Raised when a control is not visible"""
    pass

class InvalidWindowHandle(RuntimeError):
    """Raised when an invalid handle is passed to HwndWrapper"""

    def __init__(self, hwnd):
        if False:
            while True:
                i = 10
        'Initialise the RuntimError parent with the mesage'
        RuntimeError.__init__(self, 'Handle {0} is not a vaild window handle'.format(hwnd))

class HwndMeta(BaseMeta):
    """Metaclass for HwndWrapper objects"""
    re_wrappers = {}
    str_wrappers = {}

    def __init__(cls, name, bases, attrs):
        if False:
            i = 10
            return i + 15
        '\n        Register the class names\n\n        Both the regular expression or the classes directly are registered.\n        '
        BaseMeta.__init__(cls, name, bases, attrs)
        for win_class in cls.windowclasses:
            HwndMeta.re_wrappers[re.compile(win_class)] = cls
            HwndMeta.str_wrappers[win_class] = cls

    @staticmethod
    def find_wrapper(element):
        if False:
            return 10
        'Find the correct wrapper for this native element'
        if isinstance(element, six.integer_types):
            element = HwndElementInfo(element)
        class_name = element.class_name
        try:
            return HwndMeta.str_wrappers[class_name]
        except KeyError:
            wrapper_match = None
            for (regex, wrapper) in HwndMeta.re_wrappers.items():
                if regex.match(class_name):
                    wrapper_match = wrapper
                    HwndMeta.str_wrappers[class_name] = wrapper
                    return wrapper
        if handleprops.is_toplevel_window(element.handle):
            wrapper_match = DialogWrapper
        if wrapper_match is None:
            wrapper_match = HwndWrapper
        return wrapper_match

@six.add_metaclass(HwndMeta)
class HwndWrapper(WinBaseWrapper):
    """
    Default wrapper for controls.

    All other wrappers are derived from this.

    This class wraps a lot of functionality of underlying windows API
    features for working with windows.

    Most of the methods apply to every single window type. For example
    you can click() on any window.

    Most of the methods of this class are simple wrappers around
    API calls and as such they try do the simplest thing possible.

    An HwndWrapper object can be passed directly to a ctypes wrapped
    C function - and it will get converted to a Long with the value of
    it's handle (see ctypes, _as_parameter_).
    """
    handle = None

    def __new__(cls, element):
        if False:
            while True:
                i = 10
        'Construct the control wrapper'
        return super(HwndWrapper, cls)._create_wrapper(cls, element, HwndWrapper)

    def __init__(self, element_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the control\n\n        * **element_info** is either a valid HwndElementInfo or it can be an\n          instance or subclass of HwndWrapper.\n        If the handle is not valid then an InvalidWindowHandle error\n        is raised.\n        '
        if isinstance(element_info, six.integer_types):
            element_info = HwndElementInfo(element_info)
        if hasattr(element_info, 'element_info'):
            element_info = element_info.element_info
        WinBaseWrapper.__init__(self, element_info, backend.registry.backends['win32'])
        if not handleprops.iswindow(self.handle):
            raise InvalidWindowHandle(self.handle)
        self._as_parameter_ = self.handle

    @property
    def writable_props(self):
        if False:
            for i in range(10):
                print('nop')
        'Extend default properties list.'
        props = super(HwndWrapper, self).writable_props
        props.extend(['style', 'exstyle', 'user_data', 'context_help_id', 'fonts', 'client_rects', 'is_unicode', 'menu_items', 'automation_id'])
        return props

    def style(self):
        if False:
            while True:
                i = 10
        '\n        Returns the style of window\n\n        Return value is a long.\n\n        Combination of WS_* and specific control specific styles.\n        See HwndWrapper.has_style() to easily check if the window has a\n        particular style.\n        '
        return handleprops.style(self)
    Style = deprecated(style)

    def exstyle(self):
        if False:
            return 10
        '\n        Returns the Extended style of window\n\n        Return value is a long.\n\n        Combination of WS_* and specific control specific styles.\n        See HwndWrapper.has_style() to easily check if the window has a\n        particular style.\n        '
        return handleprops.exstyle(self)
    ExStyle = deprecated(exstyle, deprecated_name='ExStyle')

    def automation_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the .NET name of the control'
        return self.element_info.auto_id

    def control_type(self):
        if False:
            return 10
        'Return the .NET type of the control'
        return self.element_info.control_type

    def full_control_type(self):
        if False:
            return 10
        'Return the .NET type of the control (full, uncut)'
        return self.element_info.full_control_type

    def user_data(self):
        if False:
            return 10
        '\n        Extra data associted with the window\n\n        This value is a long value that has been associated with the window\n        and rarely has useful data (or at least data that you know the use\n        of).\n        '
        return handleprops.userdata(self)
    UserData = deprecated(user_data)

    def context_help_id(self):
        if False:
            i = 10
            return i + 15
        'Return the Context Help ID of the window'
        return handleprops.contexthelpid(self)
    ContextHelpID = deprecated(context_help_id, deprecated_name='ContextHelpID')

    def is_active(self):
        if False:
            print('Hello World!')
        'Whether the window is active or not'
        return self.top_level_parent() == self.get_active()
    IsActive = deprecated(is_active)

    def is_unicode(self):
        if False:
            i = 10
            return i + 15
        '\n        Whether the window is unicode or not\n\n        A window is Unicode if it was registered by the Wide char version\n        of RegisterClass(Ex).\n        '
        return handleprops.isunicode(self)
    IsUnicode = deprecated(is_unicode)

    def client_rect(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the client rectangle of window\n\n        The client rectangle is the window rectangle minus any borders that\n        are not available to the control for drawing.\n\n        Both top and left are always 0 for this method.\n\n        This method returns a RECT structure, Which has attributes - top,\n        left, right, bottom. and has methods width() and height().\n        See win32structures.RECT for more information.\n        '
        return handleprops.clientrect(self)
    ClientRect = deprecated(client_rect)

    def font(self):
        if False:
            return 10
        '\n        Return the font of the window\n\n        The font of the window is used to draw the text of that window.\n        It is a structure which has attributes for font name, height, width\n        etc.\n\n        See win32structures.LOGFONTW for more information.\n        '
        return handleprops.font(self)
    Font = deprecated(font)

    def has_style(self, style):
        if False:
            i = 10
            return i + 15
        'Return True if the control has the specified style'
        return handleprops.has_style(self, style)
    HasStyle = deprecated(has_style)

    def has_exstyle(self, exstyle):
        if False:
            while True:
                i = 10
        'Return True if the control has the specified extended style'
        return handleprops.has_exstyle(self, exstyle)
    HasExStyle = deprecated(has_exstyle, deprecated_name='HasExStyle')

    def is_dialog(self):
        if False:
            return 10
        'Return true if the control is a top level window'
        if not 'isdialog' in self._cache.keys():
            self._cache['isdialog'] = handleprops.is_toplevel_window(self)
        return self._cache['isdialog']

    def client_rects(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the client rect for each item in this control\n\n        It is a list of rectangles for the control. It is frequently over-ridden\n        to extract all rectangles from a control with multiple items.\n\n        It is always a list with one or more rectangles:\n\n          * First elemtent is the client rectangle of the control\n          * Subsequent elements contain the client rectangle of any items of\n            the control (e.g. items in a listbox/combobox, tabs in a\n            tabcontrol)\n        '
        return [self.client_rect()]
    ClientRects = deprecated(client_rects)

    def fonts(self):
        if False:
            print('Hello World!')
        '\n        Return the font for each item in this control\n\n        It is a list of fonts for the control. It is frequently over-ridden\n        to extract all fonts from a control with multiple items.\n\n        It is always a list with one or more fonts:\n\n          * First elemtent is the control font\n          * Subsequent elements contain the font of any items of\n            the control (e.g. items in a listbox/combobox, tabs in a\n            tabcontrol)\n        '
        return [self.font()]
    Fonts = deprecated(fonts)

    def send_command(self, commandID):
        if False:
            i = 10
            return i + 15
        return self.send_message(win32defines.WM_COMMAND, commandID)
    SendCommand = deprecated(send_command)

    def post_command(self, commandID):
        if False:
            print('Hello World!')
        return self.post_message(win32defines.WM_COMMAND, commandID)
    PostCommand = deprecated(post_command)

    def _ensure_enough_privileges(self, message_name):
        if False:
            print('Hello World!')
        'Ensure the Python process has enough rights to send some window messages'
        pid = handleprops.processid(self.handle)
        if not handleprops.has_enough_privileges(pid):
            raise RuntimeError('Not enough rights to use {} message/function for target process (to resolve it run the script as Administrator)'.format(message_name))

    def send_message(self, message, wparam=0, lparam=0):
        if False:
            while True:
                i = 10
        'Send a message to the control and wait for it to return'
        wParamAddress = wparam
        if hasattr(wparam, 'mem_address'):
            wParamAddress = wparam.mem_address
        lParamAddress = lparam
        if hasattr(lparam, 'mem_address'):
            lParamAddress = lparam.mem_address
        CArgObject = type(ctypes.byref(ctypes.c_int(0)))
        if isinstance(wparam, CArgObject):
            wParamAddress = ctypes.addressof(wparam._obj)
        if isinstance(lparam, CArgObject):
            lParamAddress = ctypes.addressof(lparam._obj)
        return win32gui.SendMessage(self.handle, message, wParamAddress, lParamAddress)
    SendMessage = deprecated(send_message)

    def send_chars(self, chars, with_spaces=True, with_tabs=True, with_newlines=True):
        if False:
            while True:
                i = 10
        '\n        Silently send a character string to the control in an inactive window\n\n        If a virtual key with no corresponding character is encountered\n        (e.g. VK_LEFT, VK_DELETE), a KeySequenceError is raised. Consider using\n        the method send_keystrokes for such input.\n        '
        input_locale_id = win32functions.GetKeyboardLayout(0)
        keys = keyboard.parse_keys(chars, with_spaces, with_tabs, with_newlines)
        for key in keys:
            key_info = key.get_key_info()
            flags = key_info[2]
            unicode_char = flags & keyboard.KEYEVENTF_UNICODE == keyboard.KEYEVENTF_UNICODE
            if unicode_char:
                (_, char) = key_info[:2]
                vk = win32functions.VkKeyScanExW(chr(char), input_locale_id) & 255
                scan = win32functions.MapVirtualKeyW(vk, 0)
            else:
                (vk, scan) = key_info[:2]
                char = win32functions.MapVirtualKeyW(vk, 2)
            if char > 0:
                lparam = 1 << 0 | scan << 16 | (flags & 1) << 24
                win32api.SendMessage(self.handle, win32con.WM_CHAR, char, lparam)
            else:
                raise keyboard.KeySequenceError('no WM_CHAR code for {key}, use method send_keystrokes instead'.format(key=key))

    def send_keystrokes(self, keystrokes, with_spaces=True, with_tabs=True, with_newlines=True):
        if False:
            while True:
                i = 10
        '\n        Silently send keystrokes to the control in an inactive window\n\n        It parses modifiers Shift(+), Control(^), Menu(%) and Sequences like "{TAB}", "{ENTER}"\n        For more information about Sequences and Modifiers navigate to module `keyboard`_\n\n        .. _`keyboard`: pywinauto.keyboard.html\n\n        Due to the fact that each application handles input differently and this method\n        is meant to be used on inactive windows, it may work only partially depending\n        on the target app. If the window being inactive is not essential, use the robust\n        `type_keys`_ method.\n\n        .. _`type_keys`: pywinauto.base_wrapper.html#pywinauto.base_wrapper.BaseWrapper.type_keys\n        '
        PBYTE256 = ctypes.c_ubyte * 256
        win32gui.SendMessage(self.handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        target_thread_id = win32functions.GetWindowThreadProcessId(self.handle, None)
        current_thread_id = win32functions.GetCurrentThreadId()
        attach_success = win32functions.AttachThreadInput(target_thread_id, current_thread_id, True) != 0
        if not attach_success:
            warnings.warn("Failed to attach app's thread to the current thread's message queue", UserWarning, stacklevel=2)
        keyboard_state_stack = [PBYTE256()]
        win32functions.GetKeyboardState(keyboard_state_stack[-1])
        input_locale_id = win32functions.GetKeyboardLayout(0)
        context_code = 0
        keys = keyboard.parse_keys(keystrokes, with_spaces, with_tabs, with_newlines)
        key_combos_present = any([isinstance(k, keyboard.EscapedKeyAction) for k in keys])
        if key_combos_present:
            warnings.warn('Key combinations may or may not work depending on the target app', UserWarning, stacklevel=2)
        try:
            for key in keys:
                (vk, scan, flags) = key.get_key_info()
                if vk == keyboard.VK_MENU or context_code == 1:
                    (down_msg, up_msg) = (win32con.WM_SYSKEYDOWN, win32con.WM_SYSKEYUP)
                else:
                    (down_msg, up_msg) = (win32con.WM_KEYDOWN, win32con.WM_KEYUP)
                repeat = 1
                shift_state = 0
                unicode_codepoint = flags & keyboard.KEYEVENTF_UNICODE != 0
                if unicode_codepoint:
                    char = chr(scan)
                    vk_with_flags = win32functions.VkKeyScanExW(char, input_locale_id)
                    vk = vk_with_flags & 255
                    shift_state = (vk_with_flags & 65280) >> 8
                    scan = win32functions.MapVirtualKeyW(vk, 0)
                if key.down and vk > 0:
                    new_keyboard_state = copy.deepcopy(keyboard_state_stack[-1])
                    new_keyboard_state[vk] |= 128
                    if shift_state & 1 == 1:
                        new_keyboard_state[keyboard.VK_SHIFT] |= 128
                    keyboard_state_stack.append(new_keyboard_state)
                    lparam = repeat << 0 | scan << 16 | (flags & 1) << 24 | context_code << 29 | 0 << 31
                    win32functions.SetKeyboardState(keyboard_state_stack[-1])
                    win32functions.PostMessage(self.handle, down_msg, vk, lparam)
                    if vk == keyboard.VK_MENU:
                        context_code = 1
                    time.sleep(0.01)
                if key.up and vk > 0:
                    keyboard_state_stack.pop()
                    lparam = repeat << 0 | scan << 16 | (flags & 1) << 24 | context_code << 29 | 1 << 30 | 1 << 31
                    win32functions.PostMessage(self.handle, up_msg, vk, lparam)
                    win32functions.SetKeyboardState(keyboard_state_stack[-1])
                    if vk == keyboard.VK_MENU:
                        context_code = 0
                    time.sleep(0.01)
        except pywintypes.error as e:
            if e.winerror == 1400:
                warnings.warn('Application exited before the end of keystrokes', UserWarning, stacklevel=2)
            else:
                warnings.warn(e.strerror, UserWarning, stacklevel=2)
            win32functions.SetKeyboardState(keyboard_state_stack[0])
        if attach_success:
            win32functions.AttachThreadInput(target_thread_id, current_thread_id, False)

    def send_message_timeout(self, message, wparam=0, lparam=0, timeout=None, timeoutflags=win32defines.SMTO_NORMAL):
        if False:
            i = 10
            return i + 15
        '\n        Send a message to the control and wait for it to return or to timeout\n\n        If no timeout is given then a default timeout of .01 of a second will\n        be used.\n        '
        if timeout is None:
            timeout = Timings.sendmessagetimeout_timeout
        try:
            (_, result) = win32gui.SendMessageTimeout(int(self.handle), message, wparam, lparam, timeoutflags, int(timeout * 1000))
        except Exception as exc:
            result = str(exc)
        return result
    SendMessageTimeout = deprecated(send_message_timeout)

    def post_message(self, message, wparam=0, lparam=0):
        if False:
            while True:
                i = 10
        'Post a message to the control message queue and return'
        return win32functions.PostMessage(self, message, wparam, lparam)
    PostMessage = deprecated(post_message)

    def notify_parent(self, message, controlID=None):
        if False:
            i = 10
            return i + 15
        'Send the notification message to parent of this control'
        if controlID is None:
            controlID = self.control_id()
        if controlID is None:
            return win32defines.TRUE
        return self.parent().post_message(win32defines.WM_COMMAND, win32functions.MakeLong(message, controlID), self)
    NotifyParent = deprecated(notify_parent)

    def wait_for_idle(self):
        if False:
            while True:
                i = 10
        'Backend specific function to wait for idle state of a thread or a window'
        win32functions.WaitGuiThreadIdle(self.handle)

    def click(self, button='left', pressed='', coords=(0, 0), double=False, absolute=False):
        if False:
            while True:
                i = 10
        "\n        Simulates a mouse click on the control\n\n        This method sends WM_* messages to the control, to do a more\n        'realistic' mouse click use click_input() which uses mouse_event() API\n        to perform the click.\n\n        This method does not require that the control be visible on the screen\n        (i.e. it can be hidden beneath another window and it will still work).\n        "
        self.verify_actionable()
        self._ensure_enough_privileges('WM_*BUTTONDOWN/UP')
        _perform_click(self, button, pressed, coords, double, absolute=absolute)
        return self
    Click = deprecated(click)

    def close_click(self, button='left', pressed='', coords=(0, 0), double=False):
        if False:
            i = 10
            return i + 15
        '\n        Perform a click action that should make the window go away\n\n        The only difference from click is that there are extra delays\n        before and after the click action.\n        '
        time.sleep(Timings.before_closeclick_wait)
        _perform_click(self, button, pressed, coords, double)

        def has_closed():
            if False:
                print('Hello World!')
            closed = not (handleprops.iswindow(self) or handleprops.iswindow(self.parent()))
            if not closed:
                try:
                    _perform_click(self, button, pressed, coords, double)
                except Exception:
                    return True
            return closed
        timings.wait_until(Timings.closeclick_dialog_close_wait, Timings.closeclick_retry, has_closed)
        time.sleep(Timings.after_closeclick_wait)
        return self
    CloseClick = deprecated(close_click)

    def close_alt_f4(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the window by pressing Alt+F4 keys.'
        time.sleep(Timings.before_closeclick_wait)
        self.type_keys('%{F4}')
        time.sleep(Timings.after_closeclick_wait)
        return self
    CloseAltF4 = deprecated(close_alt_f4)

    def double_click(self, button='left', pressed='', coords=(0, 0)):
        if False:
            for i in range(10):
                print('nop')
        'Perform a double click action'
        _perform_click(self, button, pressed, coords, double=True)
        return self
    DoubleClick = deprecated(double_click)

    def right_click(self, pressed='', coords=(0, 0)):
        if False:
            print('Hello World!')
        'Perform a right click action'
        _perform_click(self, 'right', 'right ' + pressed, coords, button_up=False)
        _perform_click(self, 'right', pressed, coords, button_down=False)
        return self
    RightClick = deprecated(right_click)

    def press_mouse(self, button='left', coords=(0, 0), pressed=''):
        if False:
            print('Hello World!')
        'Press the mouse button'
        _perform_click(self, button, pressed, coords, button_down=True, button_up=False)
        return self
    PressMouse = deprecated(press_mouse)

    def release_mouse(self, button='left', coords=(0, 0), pressed=''):
        if False:
            return 10
        'Release the mouse button'
        _perform_click(self, button, pressed, coords, button_down=False, button_up=True)
        return self
    ReleaseMouse = deprecated(release_mouse)

    def move_mouse(self, coords=(0, 0), pressed='', absolute=False):
        if False:
            return 10
        'Move the mouse by WM_MOUSEMOVE'
        if not absolute:
            self.actions.log('Moving mouse to relative (client) coordinates ' + str(coords).replace('\n', ', '))
        _perform_click(self, button='move', coords=coords, absolute=absolute, pressed=pressed)
        win32functions.WaitGuiThreadIdle(self.handle)
        return self
    MoveMouse = deprecated(move_mouse)

    def drag_mouse(self, button='left', press_coords=(0, 0), release_coords=(0, 0), pressed=''):
        if False:
            while True:
                i = 10
        'Drag the mouse'
        if isinstance(press_coords, win32structures.POINT):
            press_coords = (press_coords.x, press_coords.y)
        if isinstance(release_coords, win32structures.POINT):
            release_coords = (release_coords.x, release_coords.y)
        _pressed = pressed
        if not _pressed:
            _pressed = 'left'
        self.press_mouse(button, press_coords, pressed=pressed)
        for i in range(5):
            self.move_mouse((press_coords[0] + i, press_coords[1]), pressed=_pressed)
            time.sleep(Timings.drag_n_drop_move_mouse_wait)
        self.move_mouse(release_coords, pressed=_pressed)
        time.sleep(Timings.before_drop_wait)
        self.release_mouse(button, release_coords, pressed=pressed)
        time.sleep(Timings.after_drag_n_drop_wait)
        return self
    DragMouse = deprecated(drag_mouse)

    def set_window_text(self, text, append=False):
        if False:
            print('Hello World!')
        'Set the text of the window'
        self.verify_actionable()
        if append:
            text = self.window_text() + text
        text = ctypes.c_wchar_p(six.text_type(text))
        self.post_message(win32defines.WM_SETTEXT, 0, text)
        win32functions.WaitGuiThreadIdle(self.handle)
        self.actions.log('Set text to the ' + self.friendly_class_name() + ': ' + str(text))
        return self
    SetWindowText = deprecated(set_window_text)

    def debug_message(self, text):
        if False:
            print('Hello World!')
        'Write some debug text over the window'
        dc = win32functions.CreateDC('DISPLAY', None, None, None)
        if not dc:
            raise ctypes.WinError()
        rect = self.rectangle()
        ret = win32functions.DrawText(dc, six.text_type(text), len(text), ctypes.byref(rect), win32defines.DT_SINGLELINE)
        win32functions.DeleteDC(dc)
        if not ret:
            raise ctypes.WinError()
        return self
    DebugMessage = deprecated(debug_message)

    def set_transparency(self, alpha=120):
        if False:
            while True:
                i = 10
        'Set the window transparency from 0 to 255 by alpha attribute'
        if not 0 <= alpha <= 255:
            raise ValueError('alpha should be in [0, 255] interval!')
        win32gui.SetWindowLong(self.handle, win32defines.GWL_EXSTYLE, self.exstyle() | win32con.WS_EX_LAYERED)
        win32gui.SetLayeredWindowAttributes(self.handle, win32api.RGB(0, 0, 0), alpha, win32con.LWA_ALPHA)
    SetTransparency = deprecated(set_transparency)

    def popup_window(self):
        if False:
            for i in range(10):
                print('nop')
        'Return owned enabled Popup window wrapper if shown.\n\n        If there is no enabled popups at that time, it returns **self**.\n        See MSDN reference:\n        https://msdn.microsoft.com/en-us/library/windows/desktop/ms633515.aspx\n\n        Please do not use in production code yet - not tested fully\n        '
        popup = win32functions.GetWindow(self, win32defines.GW_ENABLEDPOPUP)
        return popup
    PopupWindow = deprecated(popup_window)

    def owner(self):
        if False:
            while True:
                i = 10
        'Return the owner window for the window if it exists\n\n        Returns None if there is no owner.\n        '
        owner = win32functions.GetWindow(self, win32defines.GW_OWNER)
        if owner:
            return HwndWrapper(owner)
        else:
            return None
    Owner = deprecated(owner)

    def _menu_handle(self):
        if False:
            while True:
                i = 10
        'Simple overridable method to get the menu handle'
        hMenu = win32gui.GetMenu(self.handle)
        is_main_menu = True
        if not hMenu:
            self._ensure_enough_privileges('MN_GETHMENU')
            hMenu = self.send_message(self.handle, win32defines.MN_GETHMENU)
            is_main_menu = False
        return (hMenu, is_main_menu)

    def menu(self):
        if False:
            print('Hello World!')
        'Return the menu of the control'
        (hMenu, is_main_menu) = self._menu_handle()
        if hMenu:
            return Menu(self, hMenu, is_main_menu=is_main_menu)
        return None
    Menu = deprecated(menu)

    def menu_item(self, path, exact=False):
        if False:
            print('Hello World!')
        'Return the menu item specified by path\n\n        Path can be a string in the form "MenuItem->MenuItem->MenuItem..."\n        where each MenuItem is the text of an item at that level of the menu.\n        E.g. ::\n\n          File->Export->ExportAsPNG\n\n        spaces are not important so you could also have written... ::\n\n          File -> Export -> Export As PNG\n\n        '
        if self.appdata is not None:
            menu_appdata = self.appdata['menu_items']
        else:
            menu_appdata = None
        menu = self.menu()
        if menu:
            return self.menu().get_menu_path(path, appdata=menu_appdata, exact=exact)[-1]
        raise RuntimeError('There is no menu.')
    MenuItem = deprecated(menu_item)

    def menu_items(self):
        if False:
            return 10
        'Return the menu items for the dialog\n\n        If there are no menu items then return an empty list\n        '
        if self.is_dialog() and self.menu():
            return self.menu().get_properties()['menu_items']
        else:
            return []
    MenuItems = deprecated(menu_items)

    def menu_select(self, path, exact=False):
        if False:
            return 10
        'Find a menu item specified by the path\n\n        The full path syntax is specified in:\n        :py:meth:`.controls.menuwrapper.Menu.get_menu_path`\n        '
        self.verify_actionable()
        self.menu_item(path, exact=exact).select()
    MenuSelect = deprecated(menu_select)

    def move_window(self, x=None, y=None, width=None, height=None):
        if False:
            i = 10
            return i + 15
        'Move the window to the new coordinates\n\n        * **x** Specifies the new left position of the window.\n          Defaults to the current left position of the window.\n        * **y** Specifies the new top position of the window.\n          Defaults to the current top position of the window.\n        * **width** Specifies the new width of the window. Defaults to the\n          current width of the window.\n        * **height** Specifies the new height of the window. Default to the\n          current height of the window.\n        '
        cur_rect = self.rectangle()
        if x is None:
            x = cur_rect.left
        else:
            try:
                y = x.top
                width = x.width()
                height = x.height()
                x = x.left
            except AttributeError:
                pass
        if y is None:
            y = cur_rect.top
        if width is None:
            width = cur_rect.width()
        if height is None:
            height = cur_rect.height()
        ret = win32functions.MoveWindow(self, x, y, width, height, True)
        if not ret:
            raise ctypes.WinError()
        win32functions.WaitGuiThreadIdle(self.handle)
        time.sleep(Timings.after_movewindow_wait)
    MoveWindow = deprecated(move_window)

    def close(self, wait_time=0):
        if False:
            return 10
        'Close the window\n\n        Code modified from http://msdn.microsoft.com/msdnmag/issues/02/08/CQA/\n\n        '
        window_text = self.window_text()
        self.post_message(win32defines.WM_CLOSE)

        def has_closed():
            if False:
                return 10
            return not (handleprops.iswindow(self) and self.is_visible())
        if not wait_time:
            wait_time = Timings.closeclick_dialog_close_wait
        try:
            timings.wait_until(wait_time, Timings.closeclick_retry, has_closed)
        except timings.TimeoutError:
            raise WindowNotFoundError
        self.actions.log('Closed window "{0}"'.format(window_text))
    Close = deprecated(close)

    def maximize(self):
        if False:
            print('Hello World!')
        'Maximize the window'
        win32functions.ShowWindow(self, win32defines.SW_MAXIMIZE)
        self.actions.log('Maximized window "{0}"'.format(self.window_text()))
        return self
    Maximize = deprecated(maximize)

    def minimize(self):
        if False:
            for i in range(10):
                print('nop')
        'Minimize the window'
        win32functions.ShowWindow(self, win32defines.SW_MINIMIZE)
        self.actions.log('Minimized window "{0}"'.format(self.window_text()))
        return self
    Minimize = deprecated(minimize)

    def restore(self):
        if False:
            while True:
                i = 10
        'Restore the window to its previous state (normal or maximized)'
        win32functions.ShowWindow(self, win32defines.SW_RESTORE)
        self.actions.log('Restored window "{0}"'.format(self.window_text()))
        return self
    Restore = deprecated(restore)

    def get_show_state(self):
        if False:
            return 10
        "Get the show state and Maximized/minimzed/restored state\n\n        Returns a value that is a union of the following\n\n        * SW_HIDE the window is hidden.\n        * SW_MAXIMIZE the window is maximized\n        * SW_MINIMIZE the window is minimized\n        * SW_RESTORE the window is in the 'restored'\n          state (neither minimized or maximized)\n        * SW_SHOW The window is not hidden\n        "
        wp = win32structures.WINDOWPLACEMENT()
        wp.lenght = ctypes.sizeof(wp)
        ret = win32functions.GetWindowPlacement(self, ctypes.byref(wp))
        if not ret:
            raise ctypes.WinError()
        return wp.showCmd
    GetShowState = deprecated(get_show_state)

    def is_minimized(self):
        if False:
            i = 10
            return i + 15
        'Indicate whether the window is minimized or not'
        return self.get_show_state() == win32defines.SW_SHOWMINIMIZED

    def is_maximized(self):
        if False:
            print('Hello World!')
        'Indicate whether the window is maximized or not'
        return self.get_show_state() == win32defines.SW_SHOWMAXIMIZED

    def is_normal(self):
        if False:
            for i in range(10):
                print('nop')
        'Indicate whether the window is normal (i.e. not minimized and not maximized)'
        return self.get_show_state() == win32defines.SW_SHOWNORMAL

    def get_focus(self):
        if False:
            while True:
                i = 10
        'Return the control in the process of this window that has the Focus\n        '
        gui_info = win32structures.GUITHREADINFO()
        gui_info.cbSize = ctypes.sizeof(gui_info)
        window_thread_id = win32functions.GetWindowThreadProcessId(self.handle, None)
        ret = win32functions.GetGUIThreadInfo(window_thread_id, ctypes.byref(gui_info))
        if not ret:
            return None
        return HwndWrapper(gui_info.hwndFocus)
    GetFocus = deprecated(get_focus)

    def set_focus(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the focus to this control.\n\n        Bring the window to the foreground first.\n        The system restricts which processes can set the foreground window\n        (https://msdn.microsoft.com/en-us/library/windows/desktop/ms633539(v=vs.85).aspx)\n        so the mouse cursor is removed from the screen to prevent any side effects.\n        '
        if not self.has_focus():
            mouse.move(coords=(-10000, 500))
            if self.is_minimized():
                if self.was_maximized():
                    self.maximize()
                else:
                    self.restore()
            else:
                win32gui.ShowWindow(self.handle, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(self.handle)
            win32functions.WaitGuiThreadIdle(self.handle)
            time.sleep(Timings.after_setfocus_wait)
        return self
    SetFocus = deprecated(set_focus)

    def has_focus(self):
        if False:
            while True:
                i = 10
        'Check the window is in focus (foreground)'
        return self.handle == win32gui.GetForegroundWindow()

    def has_keyboard_focus(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the keyboard focus on this control.'
        control_thread = win32functions.GetWindowThreadProcessId(self.handle, None)
        win32process.AttachThreadInput(control_thread, win32api.GetCurrentThreadId(), 1)
        focused = win32gui.GetFocus()
        win32process.AttachThreadInput(control_thread, win32api.GetCurrentThreadId(), 0)
        win32functions.WaitGuiThreadIdle(self.handle)
        return self.handle == focused

    def set_keyboard_focus(self):
        if False:
            while True:
                i = 10
        'Set the keyboard focus to this control.'
        control_thread = win32functions.GetWindowThreadProcessId(self.handle, None)
        win32process.AttachThreadInput(control_thread, win32api.GetCurrentThreadId(), 1)
        win32functions.SetFocus(self.handle)
        win32process.AttachThreadInput(control_thread, win32api.GetCurrentThreadId(), 0)
        win32functions.WaitGuiThreadIdle(self.handle)
        time.sleep(Timings.after_setfocus_wait)
        return self

    def set_application_data(self, appdata):
        if False:
            return 10
        'Application data is data from a previous run of the software\n\n        It is essential for running scripts written for one spoke language\n        on a different spoken language\n        '
        self.appdata = appdata
    _scroll_types = {'left': {'line': win32defines.SB_LINELEFT, 'page': win32defines.SB_PAGELEFT, 'end': win32defines.SB_LEFT}, 'right': {'line': win32defines.SB_LINERIGHT, 'page': win32defines.SB_PAGERIGHT, 'end': win32defines.SB_RIGHT}, 'up': {'line': win32defines.SB_LINEUP, 'page': win32defines.SB_PAGEUP, 'end': win32defines.SB_TOP}, 'down': {'line': win32defines.SB_LINEDOWN, 'page': win32defines.SB_PAGEDOWN, 'end': win32defines.SB_BOTTOM}}
    SetApplicationData = deprecated(set_application_data)

    def scroll(self, direction, amount, count=1, retry_interval=None):
        if False:
            print('Hello World!')
        'Ask the control to scroll itself\n\n        **direction** can be any of "up", "down", "left", "right"\n        **amount** can be one of "line", "page", "end"\n        **count** (optional) the number of times to scroll\n        '
        self._ensure_enough_privileges('WM_HSCROLL/WM_VSCROLL')
        if direction.lower() in ('left', 'right'):
            message = win32defines.WM_HSCROLL
        elif direction.lower() in ('up', 'down'):
            message = win32defines.WM_VSCROLL
        try:
            scroll_type = self._scroll_types[direction.lower()][amount.lower()]
        except KeyError:
            raise ValueError('Wrong arguments:\n                direction can be any of "up", "down", "left", "right"\n                amount can be any of "line", "page", "end"\n                ')
        if retry_interval is None:
            retry_interval = Timings.scroll_step_wait
        while count > 0:
            self.send_message(message, scroll_type)
            time.sleep(retry_interval)
            count -= 1
        return self
    Scroll = deprecated(scroll)

    def get_toolbar(self):
        if False:
            while True:
                i = 10
        'Get the first child toolbar if it exists'
        for child in self.children():
            if child.__class__.__name__ == 'ToolbarWrapper':
                return child
        return None
    GetToolbar = deprecated(get_toolbar)
    ClickInput = deprecated(WinBaseWrapper.click_input)
    DoubleClickInput = deprecated(WinBaseWrapper.double_click_input)
    RightClickInput = deprecated(WinBaseWrapper.right_click_input)
    VerifyVisible = deprecated(WinBaseWrapper.verify_visible)
    _NeedsImageProp = deprecated(WinBaseWrapper._needs_image_prop, deprecated_name='_NeedsImageProp')
    FriendlyClassName = deprecated(WinBaseWrapper.friendly_class_name)
    Class = deprecated(WinBaseWrapper.class_name, deprecated_name='Class')
    WindowText = deprecated(WinBaseWrapper.window_text)
    ControlID = deprecated(WinBaseWrapper.control_id, deprecated_name='ControlID')
    IsVisible = deprecated(WinBaseWrapper.is_visible)
    IsEnabled = deprecated(WinBaseWrapper.is_enabled)
    Rectangle = deprecated(WinBaseWrapper.rectangle)
    ClientToScreen = deprecated(WinBaseWrapper.client_to_screen)
    ProcessID = deprecated(WinBaseWrapper.process_id, deprecated_name='ProcessID')
    IsDialog = deprecated(WinBaseWrapper.is_dialog)
    Parent = deprecated(WinBaseWrapper.parent)
    TopLevelParent = deprecated(WinBaseWrapper.top_level_parent)
    Texts = deprecated(WinBaseWrapper.texts)
    Children = deprecated(WinBaseWrapper.children)
    CaptureAsImage = deprecated(WinBaseWrapper.capture_as_image)
    GetProperties = deprecated(WinBaseWrapper.get_properties)
    DrawOutline = deprecated(WinBaseWrapper.draw_outline)
    IsChild = deprecated(WinBaseWrapper.is_child)
    VerifyActionable = deprecated(WinBaseWrapper.verify_actionable)
    VerifyEnabled = deprecated(WinBaseWrapper.verify_enabled)
    PressMouseInput = deprecated(WinBaseWrapper.press_mouse_input)
    ReleaseMouseInput = deprecated(WinBaseWrapper.release_mouse_input)
    MoveMouseInput = deprecated(WinBaseWrapper.move_mouse_input)
    DragMouseInput = deprecated(WinBaseWrapper.drag_mouse_input)
    WheelMouseInput = deprecated(WinBaseWrapper.wheel_mouse_input)
    TypeKeys = deprecated(WinBaseWrapper.type_keys)

class DialogWrapper(HwndWrapper):
    """Wrap a dialog"""
    friendlyclassname = 'Dialog'
    can_be_label = True

    def __init__(self, hwnd):
        if False:
            return 10
        'Initialize the DialogWrapper\n\n        The only extra functionality here is to modify self.friendlyclassname\n        to make it "Dialog" if the class is "#32770" otherwise to leave it\n        the same as the window class.\n        '
        HwndWrapper.__init__(self, hwnd)
        if self.class_name() == '#32770':
            self.friendlyclassname = 'Dialog'
        else:
            self.friendlyclassname = self.class_name()

    def run_tests(self, tests_to_run=None, ref_controls=None):
        if False:
            while True:
                i = 10
        'Run the tests on dialog'
        from .. import tests
        controls = [self] + self.children()
        if ref_controls is not None:
            matched_flags = controlproperties.SetReferenceControls(controls, ref_controls)
        return tests.run_tests(controls, tests_to_run)
    RunTests = deprecated(run_tests)

    def write_to_xml(self, filename):
        if False:
            print('Hello World!')
        'Write the dialog an XML file (requires elementtree)'
        controls = [self] + self.children()
        props = [ctrl.get_properties() for ctrl in controls]
        from .. import xml_helpers
        xml_helpers.WriteDialogToFile(filename, props)
    WriteToXML = deprecated(write_to_xml)

    def client_area_rect(self):
        if False:
            while True:
                i = 10
        'Return the client area rectangle\n\n        From MSDN:\n        The client area of a control is the bounds of the control, minus the\n        nonclient elements such as scroll bars, borders, title bars, and\n        menus.\n        '
        rect = win32structures.RECT(self.rectangle())
        self.send_message(win32defines.WM_NCCALCSIZE, 0, ctypes.byref(rect))
        return rect
    ClientAreaRect = deprecated(client_area_rect)

    def hide_from_taskbar(self):
        if False:
            while True:
                i = 10
        'Hide the dialog from the Windows taskbar'
        win32functions.ShowWindow(self, win32defines.SW_HIDE)
        win32functions.SetWindowLongPtr(self, win32defines.GWL_EXSTYLE, self.exstyle() | win32defines.WS_EX_TOOLWINDOW)
        win32functions.ShowWindow(self, win32defines.SW_SHOW)
    HideFromTaskbar = deprecated(hide_from_taskbar)

    def show_in_taskbar(self):
        if False:
            i = 10
            return i + 15
        'Show the dialog in the Windows taskbar'
        win32functions.ShowWindow(self, win32defines.SW_HIDE)
        win32functions.SetWindowLongPtr(self, win32defines.GWL_EXSTYLE, self.exstyle() | win32defines.WS_EX_APPWINDOW)
        win32functions.ShowWindow(self, win32defines.SW_SHOW)
    ShowInTaskbar = deprecated(show_in_taskbar)

    def is_in_taskbar(self):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the dialog is shown in the Windows taskbar\n\n        Thanks to David Heffernan for the idea:\n        http://stackoverflow.com/questions/30933219/hide-window-from-taskbar-without-using-ws-ex-toolwindow\n        A window is represented in the taskbar if:\n        It has no owner and it does not have the WS_EX_TOOLWINDOW extended style,\n        or it has the WS_EX_APPWINDOW extended style.\n        '
        return self.has_exstyle(win32defines.WS_EX_APPWINDOW) or (self.owner() is None and (not self.has_exstyle(win32defines.WS_EX_TOOLWINDOW)))
    IsInTaskbar = deprecated(is_in_taskbar)

    def force_close(self):
        if False:
            while True:
                i = 10
        "Close the dialog forcefully using WM_QUERYENDSESSION and return the result\n\n        Window has let us know that it doesn't want to die - so we abort\n        this means that the app is not hung - but knows it doesn't want\n        to close yet - e.g. it is asking the user if they want to save.\n        "
        self.send_message_timeout(win32defines.WM_QUERYENDSESSION, timeout=0.5, timeoutflags=win32defines.SMTO_ABORTIFHUNG)
        pid = ctypes.c_ulong()
        win32functions.GetWindowThreadProcessId(self.handle, ctypes.byref(pid))
        try:
            process_wait_handle = win32api.OpenProcess(win32con.SYNCHRONIZE | win32con.PROCESS_TERMINATE, 0, pid.value)
        except win32gui.error:
            return True
        result = win32event.WaitForSingleObject(process_wait_handle, int(Timings.after_windowclose_timeout * 1000))
        return result != win32con.WAIT_TIMEOUT

def _perform_click(ctrl, button='left', pressed='', coords=(0, 0), double=False, button_down=True, button_up=True, absolute=False):
    if False:
        print('Hello World!')
    'Low level method for performing click operations'
    if ctrl is None:
        ctrl = HwndWrapper(win32functions.GetDesktopWindow())
    ctrl.verify_actionable()
    ctrl_text = ctrl.window_text()
    if ctrl_text is None:
        ctrl_text = six.text_type(ctrl_text)
    ctrl_friendly_class_name = ctrl.friendly_class_name()
    if isinstance(coords, win32structures.RECT):
        coords = coords.mid_point()
    elif isinstance(coords, win32structures.POINT):
        coords = [coords.x, coords.y]
    else:
        coords = list(coords)
    if absolute:
        coords = ctrl.client_to_screen(coords)
    msgs = []
    if not double:
        if button.lower() == 'left':
            if button_down:
                msgs.append(win32defines.WM_LBUTTONDOWN)
            if button_up:
                msgs.append(win32defines.WM_LBUTTONUP)
        elif button.lower() == 'middle':
            if button_down:
                msgs.append(win32defines.WM_MBUTTONDOWN)
            if button_up:
                msgs.append(win32defines.WM_MBUTTONUP)
        elif button.lower() == 'right':
            if button_down:
                msgs.append(win32defines.WM_RBUTTONDOWN)
            if button_up:
                msgs.append(win32defines.WM_RBUTTONUP)
        elif button.lower() == 'move':
            msgs.append(win32defines.WM_MOUSEMOVE)
    elif button.lower() == 'left':
        msgs = (win32defines.WM_LBUTTONDOWN, win32defines.WM_LBUTTONUP, win32defines.WM_LBUTTONDBLCLK, win32defines.WM_LBUTTONUP)
    elif button.lower() == 'middle':
        msgs = (win32defines.WM_MBUTTONDOWN, win32defines.WM_MBUTTONUP, win32defines.WM_MBUTTONDBLCLK, win32defines.WM_MBUTTONUP)
    elif button.lower() == 'right':
        msgs = (win32defines.WM_RBUTTONDOWN, win32defines.WM_RBUTTONUP, win32defines.WM_RBUTTONDBLCLK, win32defines.WM_RBUTTONUP)
    elif button.lower() == 'move':
        msgs.append(win32defines.WM_MOUSEMOVE)
    (flags, click_point) = _calc_flags_and_coords(pressed, coords)
    for msg in msgs:
        win32functions.PostMessage(ctrl, msg, win32structures.WPARAM(flags), win32structures.LPARAM(click_point))
        time.sleep(Timings.sendmessagetimeout_timeout)
        win32functions.WaitGuiThreadIdle(ctrl.handle)
    time.sleep(Timings.after_click_wait)
    if button.lower() == 'move':
        message = 'Moved mouse over ' + ctrl_friendly_class_name + ' "' + ctrl_text + '" to screen point ' + str(tuple(coords)) + ' by WM_MOUSEMOVE'
    else:
        message = 'Clicked ' + ctrl_friendly_class_name + ' "' + ctrl_text + '" by ' + str(button) + ' button event ' + str(tuple(coords))
        if double:
            message = 'Double-c' + message[1:]
    ActionLogger().log(message)
_mouse_flags = {'left': win32defines.MK_LBUTTON, 'right': win32defines.MK_RBUTTON, 'middle': win32defines.MK_MBUTTON, 'shift': win32defines.MK_SHIFT, 'control': win32defines.MK_CONTROL}

def _calc_flags_and_coords(pressed, coords):
    if False:
        return 10
    'Calculate the flags to use and the coordinates for mouse actions'
    flags = 0
    for key in pressed.split():
        flags |= _mouse_flags[key.lower()]
    click_point = win32functions.MakeLong(coords[1], coords[0])
    return (flags, click_point)

class _DummyControl(dict):
    """A subclass of dict so that we can assign attributes"""
    pass

def get_dialog_props_from_handle(hwnd):
    if False:
        print('Hello World!')
    'Get the properties of all the controls as a list of dictionaries'
    try:
        controls = [hwnd]
        controls.extend(hwnd.children())
    except AttributeError:
        controls = [HwndWrapper(hwnd)]
        controls.extend(controls[0].children())
    props = []
    for ctrl in controls:
        ctrl_props = _DummyControl(ctrl.get_properties())
        ctrl_props.handle = ctrl.handle
        ctrl_props['rectangle'] -= controls[0].rectangle()
        props.append(ctrl_props)
    return props
GetDialogPropsFromHandle = deprecated(get_dialog_props_from_handle)
backend.register('win32', HwndElementInfo, HwndWrapper)
backend.registry.backends['win32'].dialog_class = DialogWrapper
backend.activate('win32')