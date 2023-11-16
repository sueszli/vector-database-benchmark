import tkinter
from packaging import version
import sys
import os
import platform
import ctypes
from typing import Union, Tuple, Optional
from .widgets.theme import ThemeManager
from .widgets.scaling import CTkScalingBaseClass
from .widgets.appearance_mode import CTkAppearanceModeBaseClass
from customtkinter.windows.widgets.utility.utility_functions import pop_from_dict_by_set, check_kwargs_empty

class CTkToplevel(tkinter.Toplevel, CTkAppearanceModeBaseClass, CTkScalingBaseClass):
    """
    Toplevel window with dark titlebar on Windows and macOS.
    For detailed information check out the documentation.
    """
    _valid_tk_toplevel_arguments: set = {'master', 'bd', 'borderwidth', 'class', 'container', 'cursor', 'height', 'highlightbackground', 'highlightthickness', 'menu', 'relief', 'screen', 'takefocus', 'use', 'visual', 'width'}
    _deactivate_macos_window_header_manipulation: bool = False
    _deactivate_windows_window_header_manipulation: bool = False

    def __init__(self, *args, fg_color: Optional[Union[str, Tuple[str, str]]]=None, **kwargs):
        if False:
            while True:
                i = 10
        self._enable_macos_dark_title_bar()
        super().__init__(*args, **pop_from_dict_by_set(kwargs, self._valid_tk_toplevel_arguments))
        CTkAppearanceModeBaseClass.__init__(self)
        CTkScalingBaseClass.__init__(self, scaling_type='window')
        check_kwargs_empty(kwargs, raise_error=True)
        try:
            if sys.platform.startswith('win'):
                customtkinter_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.after(200, lambda : self.iconbitmap(os.path.join(customtkinter_directory, 'assets', 'icons', 'CustomTkinter_icon_Windows.ico')))
        except Exception:
            pass
        self._current_width = 200
        self._current_height = 200
        self._min_width: int = 0
        self._min_height: int = 0
        self._max_width: int = 1000000
        self._max_height: int = 1000000
        self._last_resizable_args: Union[Tuple[list, dict], None] = None
        self._fg_color = ThemeManager.theme['CTkToplevel']['fg_color'] if fg_color is None else self._check_color_type(fg_color)
        super().configure(bg=self._apply_appearance_mode(self._fg_color))
        super().title('CTkToplevel')
        self._iconbitmap_method_called = True
        self._state_before_windows_set_titlebar_color = None
        self._windows_set_titlebar_color_called = False
        self._withdraw_called_after_windows_set_titlebar_color = False
        self._iconify_called_after_windows_set_titlebar_color = False
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
            for i in range(10):
                print('nop')
        self._disable_macos_dark_title_bar()
        tkinter.Toplevel.destroy(self)
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
            detected_width = self.winfo_width()
            detected_height = self.winfo_height()
            if self._current_width != self._reverse_window_scaling(detected_width) or self._current_height != self._reverse_window_scaling(detected_height):
                self._current_width = self._reverse_window_scaling(detected_width)
                self._current_height = self._reverse_window_scaling(detected_height)

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        if False:
            for i in range(10):
                print('nop')
        super()._set_scaling(new_widget_scaling, new_window_scaling)
        super().minsize(self._apply_window_scaling(self._current_width), self._apply_window_scaling(self._current_height))
        super().maxsize(self._apply_window_scaling(self._current_width), self._apply_window_scaling(self._current_height))
        super().geometry(f'{self._apply_window_scaling(self._current_width)}x{self._apply_window_scaling(self._current_height)}')
        self.after(1000, self._set_scaled_min_max)

    def block_update_dimensions_event(self):
        if False:
            while True:
                i = 10
        self._block_update_dimensions_event = False

    def unblock_update_dimensions_event(self):
        if False:
            i = 10
            return i + 15
        self._block_update_dimensions_event = False

    def _set_scaled_min_max(self):
        if False:
            return 10
        if self._min_width is not None or self._min_height is not None:
            super().minsize(self._apply_window_scaling(self._min_width), self._apply_window_scaling(self._min_height))
        if self._max_width is not None or self._max_height is not None:
            super().maxsize(self._apply_window_scaling(self._max_width), self._apply_window_scaling(self._max_height))

    def geometry(self, geometry_string: str=None):
        if False:
            return 10
        if geometry_string is not None:
            super().geometry(self._apply_geometry_scaling(geometry_string))
            (width, height, x, y) = self._parse_geometry_string(geometry_string)
            if width is not None and height is not None:
                self._current_width = max(self._min_width, min(width, self._max_width))
                self._current_height = max(self._min_height, min(height, self._max_height))
        else:
            return self._reverse_geometry_scaling(super().geometry())

    def withdraw(self):
        if False:
            print('Hello World!')
        if self._windows_set_titlebar_color_called:
            self._withdraw_called_after_windows_set_titlebar_color = True
        super().withdraw()

    def iconify(self):
        if False:
            while True:
                i = 10
        if self._windows_set_titlebar_color_called:
            self._iconify_called_after_windows_set_titlebar_color = True
        super().iconify()

    def resizable(self, width: bool=None, height: bool=None):
        if False:
            while True:
                i = 10
        current_resizable_values = super().resizable(width, height)
        self._last_resizable_args = ([], {'width': width, 'height': height})
        if sys.platform.startswith('win'):
            self.after(10, lambda : self._windows_set_titlebar_color(self._get_appearance_mode()))
        return current_resizable_values

    def minsize(self, width=None, height=None):
        if False:
            print('Hello World!')
        self._min_width = width
        self._min_height = height
        if self._current_width < width:
            self._current_width = width
        if self._current_height < height:
            self._current_height = height
        super().minsize(self._apply_window_scaling(self._min_width), self._apply_window_scaling(self._min_height))

    def maxsize(self, width=None, height=None):
        if False:
            while True:
                i = 10
        self._max_width = width
        self._max_height = height
        if self._current_width > width:
            self._current_width = width
        if self._current_height > height:
            self._current_height = height
        super().maxsize(self._apply_window_scaling(self._max_width), self._apply_window_scaling(self._max_height))

    def configure(self, **kwargs):
        if False:
            while True:
                i = 10
        if 'fg_color' in kwargs:
            self._fg_color = self._check_color_type(kwargs.pop('fg_color'))
            super().configure(bg=self._apply_appearance_mode(self._fg_color))
            for child in self.winfo_children():
                try:
                    child.configure(bg_color=self._fg_color)
                except Exception:
                    pass
        super().configure(**pop_from_dict_by_set(kwargs, self._valid_tk_toplevel_arguments))
        check_kwargs_empty(kwargs)

    def cget(self, attribute_name: str) -> any:
        if False:
            return 10
        if attribute_name == 'fg_color':
            return self._fg_color
        else:
            return super().cget(attribute_name)

    def wm_iconbitmap(self, bitmap=None, default=None):
        if False:
            for i in range(10):
                print('nop')
        self._iconbitmap_method_called = True
        super().wm_iconbitmap(bitmap, default)

    def _windows_set_titlebar_icon(self):
        if False:
            while True:
                i = 10
        try:
            if not self._iconbitmap_method_called:
                customtkinter_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.iconbitmap(os.path.join(customtkinter_directory, 'assets', 'icons', 'CustomTkinter_icon_Windows.ico'))
        except Exception:
            pass

    @classmethod
    def _enable_macos_dark_title_bar(cls):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        '\n        Set the titlebar color of the window to light or dark theme on Microsoft Windows.\n\n        Credits for this function:\n        https://stackoverflow.com/questions/23836000/can-i-change-the-title-bar-in-tkinter/70724666#70724666\n\n        MORE INFO:\n        https://docs.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute\n        '
        if sys.platform.startswith('win') and (not self._deactivate_windows_window_header_manipulation):
            self._state_before_windows_set_titlebar_color = self.state()
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
            self._windows_set_titlebar_color_called = True
            self.after(5, self._revert_withdraw_after_windows_set_titlebar_color)
            if self.focused_widget_before_widthdraw is not None:
                self.after(10, self.focused_widget_before_widthdraw.focus)
                self.focused_widget_before_widthdraw = None

    def _revert_withdraw_after_windows_set_titlebar_color(self):
        if False:
            return 10
        ' if in a short time (5ms) after '
        if self._windows_set_titlebar_color_called:
            if self._withdraw_called_after_windows_set_titlebar_color:
                pass
            elif self._iconify_called_after_windows_set_titlebar_color:
                super().iconify()
            elif self._state_before_windows_set_titlebar_color == 'normal':
                self.deiconify()
            elif self._state_before_windows_set_titlebar_color == 'iconic':
                self.iconify()
            elif self._state_before_windows_set_titlebar_color == 'zoomed':
                self.state('zoomed')
            else:
                self.state(self._state_before_windows_set_titlebar_color)
            self._windows_set_titlebar_color_called = False
            self._withdraw_called_after_windows_set_titlebar_color = False
            self._iconify_called_after_windows_set_titlebar_color = False

    def _set_appearance_mode(self, mode_string):
        if False:
            print('Hello World!')
        super()._set_appearance_mode(mode_string)
        if sys.platform.startswith('win'):
            self._windows_set_titlebar_color(mode_string)
        super().configure(bg=self._apply_appearance_mode(self._fg_color))