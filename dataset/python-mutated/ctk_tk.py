import tkinter
import sys
import os
import platform
import ctypes
from typing import Union, Tuple, Optional
from packaging import version
from .widgets.theme import ThemeManager
from .widgets.scaling import CTkScalingBaseClass
from .widgets.appearance_mode import CTkAppearanceModeBaseClass
from customtkinter.windows.widgets.utility.utility_functions import pop_from_dict_by_set, check_kwargs_empty
CTK_PARENT_CLASS = tkinter.Tk

class CTk(CTK_PARENT_CLASS, CTkAppearanceModeBaseClass, CTkScalingBaseClass):
    """
    Main app window with dark titlebar on Windows and macOS.
    For detailed information check out the documentation.
    """
    _valid_tk_constructor_arguments: set = {'screenName', 'baseName', 'className', 'useTk', 'sync', 'use'}
    _valid_tk_configure_arguments: set = {'bd', 'borderwidth', 'class', 'menu', 'relief', 'screen', 'use', 'container', 'cursor', 'height', 'highlightthickness', 'padx', 'pady', 'takefocus', 'visual', 'width'}
    _deactivate_macos_window_header_manipulation: bool = False
    _deactivate_windows_window_header_manipulation: bool = False

    def __init__(self, fg_color: Optional[Union[str, Tuple[str, str]]]=None, **kwargs):
        if False:
            return 10
        self._enable_macos_dark_title_bar()
        CTK_PARENT_CLASS.__init__(self, **pop_from_dict_by_set(kwargs, self._valid_tk_constructor_arguments))
        CTkAppearanceModeBaseClass.__init__(self)
        CTkScalingBaseClass.__init__(self, scaling_type='window')
        check_kwargs_empty(kwargs, raise_error=True)
        self._current_width = 600
        self._current_height = 500
        self._min_width: int = 0
        self._min_height: int = 0
        self._max_width: int = 1000000
        self._max_height: int = 1000000
        self._last_resizable_args: Union[Tuple[list, dict], None] = None
        self._fg_color = ThemeManager.theme['CTk']['fg_color'] if fg_color is None else self._check_color_type(fg_color)
        super().configure(bg=self._apply_appearance_mode(self._fg_color))
        self.title('CTk')
        self._iconbitmap_method_called = False
        self._state_before_windows_set_titlebar_color = None
        self._window_exists = False
        self._withdraw_called_before_window_exists = False
        self._iconify_called_before_window_exists = False
        self._block_update_dimensions_event = False
        self.focused_widget_before_widthdraw = None
        if sys.platform.startswith('win'):
            self.after(200, self._windows_set_titlebar_icon)
        if sys.platform.startswith('win'):
            self._windows_set_titlebar_color(self._get_appearance_mode())
        self.bind('<Configure>', self._update_dimensions_event)
        self.bind('<FocusIn>', self._focus_in_event)

    def destroy(self):
        if False:
            print('Hello World!')
        self._disable_macos_dark_title_bar()
        tkinter.Tk.destroy(self)
        CTkAppearanceModeBaseClass.destroy(self)
        CTkScalingBaseClass.destroy(self)

    def _focus_in_event(self, event):
        if False:
            print('Hello World!')
        if sys.platform == 'darwin':
            self.lift()

    def _update_dimensions_event(self, event=None):
        if False:
            return 10
        if not self._block_update_dimensions_event:
            detected_width = super().winfo_width()
            detected_height = super().winfo_height()
            if self._current_width != self._reverse_window_scaling(detected_width) or self._current_height != self._reverse_window_scaling(detected_height):
                self._current_width = self._reverse_window_scaling(detected_width)
                self._current_height = self._reverse_window_scaling(detected_height)

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        if False:
            print('Hello World!')
        super()._set_scaling(new_widget_scaling, new_window_scaling)
        super().minsize(self._apply_window_scaling(self._current_width), self._apply_window_scaling(self._current_height))
        super().maxsize(self._apply_window_scaling(self._current_width), self._apply_window_scaling(self._current_height))
        super().geometry(f'{self._apply_window_scaling(self._current_width)}x{self._apply_window_scaling(self._current_height)}')
        self.after(1000, self._set_scaled_min_max)

    def block_update_dimensions_event(self):
        if False:
            print('Hello World!')
        self._block_update_dimensions_event = False

    def unblock_update_dimensions_event(self):
        if False:
            i = 10
            return i + 15
        self._block_update_dimensions_event = False

    def _set_scaled_min_max(self):
        if False:
            print('Hello World!')
        if self._min_width is not None or self._min_height is not None:
            super().minsize(self._apply_window_scaling(self._min_width), self._apply_window_scaling(self._min_height))
        if self._max_width is not None or self._max_height is not None:
            super().maxsize(self._apply_window_scaling(self._max_width), self._apply_window_scaling(self._max_height))

    def withdraw(self):
        if False:
            i = 10
            return i + 15
        if self._window_exists is False:
            self._withdraw_called_before_window_exists = True
        super().withdraw()

    def iconify(self):
        if False:
            while True:
                i = 10
        if self._window_exists is False:
            self._iconify_called_before_window_exists = True
        super().iconify()

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        if self._window_exists is False:
            if sys.platform.startswith('win'):
                if not self._withdraw_called_before_window_exists and (not self._iconify_called_before_window_exists):
                    self.deiconify()
            self._window_exists = True
        super().update()

    def mainloop(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self._window_exists:
            if sys.platform.startswith('win'):
                self._windows_set_titlebar_color(self._get_appearance_mode())
                if not self._withdraw_called_before_window_exists and (not self._iconify_called_before_window_exists):
                    self.deiconify()
            self._window_exists = True
        super().mainloop(*args, **kwargs)

    def resizable(self, width: bool=None, height: bool=None):
        if False:
            while True:
                i = 10
        current_resizable_values = super().resizable(width, height)
        self._last_resizable_args = ([], {'width': width, 'height': height})
        if sys.platform.startswith('win'):
            self._windows_set_titlebar_color(self._get_appearance_mode())
        return current_resizable_values

    def minsize(self, width: int=None, height: int=None):
        if False:
            i = 10
            return i + 15
        self._min_width = width
        self._min_height = height
        if self._current_width < width:
            self._current_width = width
        if self._current_height < height:
            self._current_height = height
        super().minsize(self._apply_window_scaling(self._min_width), self._apply_window_scaling(self._min_height))

    def maxsize(self, width: int=None, height: int=None):
        if False:
            i = 10
            return i + 15
        self._max_width = width
        self._max_height = height
        if self._current_width > width:
            self._current_width = width
        if self._current_height > height:
            self._current_height = height
        super().maxsize(self._apply_window_scaling(self._max_width), self._apply_window_scaling(self._max_height))

    def geometry(self, geometry_string: str=None):
        if False:
            for i in range(10):
                print('nop')
        if geometry_string is not None:
            super().geometry(self._apply_geometry_scaling(geometry_string))
            (width, height, x, y) = self._parse_geometry_string(geometry_string)
            if width is not None and height is not None:
                self._current_width = max(self._min_width, min(width, self._max_width))
                self._current_height = max(self._min_height, min(height, self._max_height))
        else:
            return self._reverse_geometry_scaling(super().geometry())

    def configure(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'fg_color' in kwargs:
            self._fg_color = self._check_color_type(kwargs.pop('fg_color'))
            super().configure(bg=self._apply_appearance_mode(self._fg_color))
            for child in self.winfo_children():
                try:
                    child.configure(bg_color=self._fg_color)
                except Exception:
                    pass
        super().configure(**pop_from_dict_by_set(kwargs, self._valid_tk_configure_arguments))
        check_kwargs_empty(kwargs)

    def cget(self, attribute_name: str) -> any:
        if False:
            i = 10
            return i + 15
        if attribute_name == 'fg_color':
            return self._fg_color
        else:
            return super().cget(attribute_name)

    def wm_iconbitmap(self, bitmap=None, default=None):
        if False:
            print('Hello World!')
        self._iconbitmap_method_called = True
        super().wm_iconbitmap(bitmap, default)

    def iconbitmap(self, bitmap=None, default=None):
        if False:
            i = 10
            return i + 15
        self._iconbitmap_method_called = True
        super().wm_iconbitmap(bitmap, default)

    def _windows_set_titlebar_icon(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if not self._iconbitmap_method_called:
                customtkinter_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.iconbitmap(os.path.join(customtkinter_directory, 'assets', 'icons', 'CustomTkinter_icon_Windows.ico'))
        except Exception:
            pass

    @classmethod
    def _enable_macos_dark_title_bar(cls):
        if False:
            i = 10
            return i + 15
        if sys.platform == 'darwin' and (not cls._deactivate_macos_window_header_manipulation):
            if version.parse(platform.python_version()) < version.parse('3.10'):
                if version.parse(tkinter.Tcl().call('info', 'patchlevel')) >= version.parse('8.6.9'):
                    os.system('defaults write -g NSRequiresAquaSystemAppearance -bool No')

    @classmethod
    def _disable_macos_dark_title_bar(cls):
        if False:
            return 10
        if sys.platform == 'darwin' and (not cls._deactivate_macos_window_header_manipulation):
            if version.parse(platform.python_version()) < version.parse('3.10'):
                if version.parse(tkinter.Tcl().call('info', 'patchlevel')) >= version.parse('8.6.9'):
                    os.system('defaults delete -g NSRequiresAquaSystemAppearance')

    def _windows_set_titlebar_color(self, color_mode: str):
        if False:
            while True:
                i = 10
        '\n        Set the titlebar color of the window to light or dark theme on Microsoft Windows.\n\n        Credits for this function:\n        https://stackoverflow.com/questions/23836000/can-i-change-the-title-bar-in-tkinter/70724666#70724666\n\n        MORE INFO:\n        https://docs.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute\n        '
        if sys.platform.startswith('win') and (not self._deactivate_windows_window_header_manipulation):
            if self._window_exists:
                self._state_before_windows_set_titlebar_color = self.state()
                if self._state_before_windows_set_titlebar_color != 'iconic' or self._state_before_windows_set_titlebar_color != 'withdrawn':
                    self.focused_widget_before_widthdraw = self.focus_get()
                    super().withdraw()
            else:
                self.focused_widget_before_widthdraw = self.focus_get()
                super().withdraw()
                super().update()
            if color_mode.lower() == 'dark':
                value = 1
            elif color_mode.lower() == 'light':
                value = 0
            else:
                return
            try:
                hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(ctypes.c_int(value)), ctypes.sizeof(ctypes.c_int(value))) != 0:
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, ctypes.byref(ctypes.c_int(value)), ctypes.sizeof(ctypes.c_int(value)))
            except Exception as err:
                print(err)
            if self._window_exists or True:
                if self._state_before_windows_set_titlebar_color == 'normal':
                    self.deiconify()
                elif self._state_before_windows_set_titlebar_color == 'iconic':
                    self.iconify()
                elif self._state_before_windows_set_titlebar_color == 'zoomed':
                    self.state('zoomed')
                else:
                    self.state(self._state_before_windows_set_titlebar_color)
            else:
                pass
            if self.focused_widget_before_widthdraw is not None:
                self.after(1, self.focused_widget_before_widthdraw.focus)
                self.focused_widget_before_widthdraw = None

    def _set_appearance_mode(self, mode_string: str):
        if False:
            print('Hello World!')
        super()._set_appearance_mode(mode_string)
        if sys.platform.startswith('win'):
            self._windows_set_titlebar_color(mode_string)
        super().configure(bg=self._apply_appearance_mode(self._fg_color))