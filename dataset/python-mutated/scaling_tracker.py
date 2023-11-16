import tkinter
import sys
from typing import Callable

class ScalingTracker:
    deactivate_automatic_dpi_awareness = False
    window_widgets_dict = {}
    window_dpi_scaling_dict = {}
    widget_scaling = 1
    window_scaling = 1
    update_loop_running = False
    update_loop_interval = 100
    loop_pause_after_new_scaling = 1500

    @classmethod
    def get_widget_scaling(cls, widget) -> float:
        if False:
            while True:
                i = 10
        window_root = cls.get_window_root_of_widget(widget)
        return cls.window_dpi_scaling_dict[window_root] * cls.widget_scaling

    @classmethod
    def get_window_scaling(cls, window) -> float:
        if False:
            print('Hello World!')
        window_root = cls.get_window_root_of_widget(window)
        return cls.window_dpi_scaling_dict[window_root] * cls.window_scaling

    @classmethod
    def set_widget_scaling(cls, widget_scaling_factor: float):
        if False:
            return 10
        cls.widget_scaling = max(widget_scaling_factor, 0.4)
        cls.update_scaling_callbacks_all()

    @classmethod
    def set_window_scaling(cls, window_scaling_factor: float):
        if False:
            for i in range(10):
                print('nop')
        cls.window_scaling = max(window_scaling_factor, 0.4)
        cls.update_scaling_callbacks_all()

    @classmethod
    def get_window_root_of_widget(cls, widget):
        if False:
            i = 10
            return i + 15
        current_widget = widget
        while isinstance(current_widget, tkinter.Tk) is False and isinstance(current_widget, tkinter.Toplevel) is False:
            current_widget = current_widget.master
        return current_widget

    @classmethod
    def update_scaling_callbacks_all(cls):
        if False:
            return 10
        for (window, callback_list) in cls.window_widgets_dict.items():
            for set_scaling_callback in callback_list:
                if not cls.deactivate_automatic_dpi_awareness:
                    set_scaling_callback(cls.window_dpi_scaling_dict[window] * cls.widget_scaling, cls.window_dpi_scaling_dict[window] * cls.window_scaling)
                else:
                    set_scaling_callback(cls.widget_scaling, cls.window_scaling)

    @classmethod
    def update_scaling_callbacks_for_window(cls, window):
        if False:
            i = 10
            return i + 15
        for set_scaling_callback in cls.window_widgets_dict[window]:
            if not cls.deactivate_automatic_dpi_awareness:
                set_scaling_callback(cls.window_dpi_scaling_dict[window] * cls.widget_scaling, cls.window_dpi_scaling_dict[window] * cls.window_scaling)
            else:
                set_scaling_callback(cls.widget_scaling, cls.window_scaling)

    @classmethod
    def add_widget(cls, widget_callback: Callable, widget):
        if False:
            i = 10
            return i + 15
        window_root = cls.get_window_root_of_widget(widget)
        if window_root not in cls.window_widgets_dict:
            cls.window_widgets_dict[window_root] = [widget_callback]
        else:
            cls.window_widgets_dict[window_root].append(widget_callback)
        if window_root not in cls.window_dpi_scaling_dict:
            cls.window_dpi_scaling_dict[window_root] = cls.get_window_dpi_scaling(window_root)
        if not cls.update_loop_running:
            window_root.after(100, cls.check_dpi_scaling)
            cls.update_loop_running = True

    @classmethod
    def remove_widget(cls, widget_callback, widget):
        if False:
            i = 10
            return i + 15
        window_root = cls.get_window_root_of_widget(widget)
        try:
            cls.window_widgets_dict[window_root].remove(widget_callback)
        except:
            pass

    @classmethod
    def remove_window(cls, window_callback, window):
        if False:
            for i in range(10):
                print('nop')
        try:
            del cls.window_widgets_dict[window]
        except:
            pass

    @classmethod
    def add_window(cls, window_callback, window):
        if False:
            while True:
                i = 10
        if window not in cls.window_widgets_dict:
            cls.window_widgets_dict[window] = [window_callback]
        else:
            cls.window_widgets_dict[window].append(window_callback)
        if window not in cls.window_dpi_scaling_dict:
            cls.window_dpi_scaling_dict[window] = cls.get_window_dpi_scaling(window)

    @classmethod
    def activate_high_dpi_awareness(cls):
        if False:
            return 10
        ' make process DPI aware, customtkinter elements will get scaled automatically,\n            only gets activated when CTk object is created '
        if not cls.deactivate_automatic_dpi_awareness:
            if sys.platform == 'darwin':
                pass
            elif sys.platform.startswith('win'):
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            else:
                pass

    @classmethod
    def get_window_dpi_scaling(cls, window) -> float:
        if False:
            print('Hello World!')
        if not cls.deactivate_automatic_dpi_awareness:
            if sys.platform == 'darwin':
                return 1
            elif sys.platform.startswith('win'):
                from ctypes import windll, pointer, wintypes
                DPI100pc = 96
                DPI_type = 0
                window_hwnd = wintypes.HWND(window.winfo_id())
                monitor_handle = windll.user32.MonitorFromWindow(window_hwnd, wintypes.DWORD(2))
                (x_dpi, y_dpi) = (wintypes.UINT(), wintypes.UINT())
                windll.shcore.GetDpiForMonitor(monitor_handle, DPI_type, pointer(x_dpi), pointer(y_dpi))
                return (x_dpi.value + y_dpi.value) / (2 * DPI100pc)
            else:
                return 1
        else:
            return 1

    @classmethod
    def check_dpi_scaling(cls):
        if False:
            print('Hello World!')
        new_scaling_detected = False
        for window in cls.window_widgets_dict:
            if window.winfo_exists() and (not window.state() == 'iconic'):
                current_dpi_scaling_value = cls.get_window_dpi_scaling(window)
                if current_dpi_scaling_value != cls.window_dpi_scaling_dict[window]:
                    cls.window_dpi_scaling_dict[window] = current_dpi_scaling_value
                    if sys.platform.startswith('win'):
                        window.attributes('-alpha', 0.15)
                    window.block_update_dimensions_event()
                    cls.update_scaling_callbacks_for_window(window)
                    window.unblock_update_dimensions_event()
                    if sys.platform.startswith('win'):
                        window.attributes('-alpha', 1)
                    new_scaling_detected = True
        for app in cls.window_widgets_dict.keys():
            try:
                if new_scaling_detected:
                    app.after(cls.loop_pause_after_new_scaling, cls.check_dpi_scaling)
                else:
                    app.after(cls.update_loop_interval, cls.check_dpi_scaling)
                return
            except Exception:
                continue
        cls.update_loop_running = False