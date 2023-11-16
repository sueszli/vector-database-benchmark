"""Module showing how to work with the task bar

This module will likely change significantly in the future!
"""
import warnings
from . import findwindows
from . import application
warnings.warn('The taskbar module is still very experimental', FutureWarning)

def TaskBarHandle():
    if False:
        i = 10
        return i + 15
    "Return the first window that has a class name 'Shell_TrayWnd'"
    return findwindows.find_elements(class_name='Shell_TrayWnd')[0].handle

def _click_hidden_tray_icon(reqd_button, mouse_button='left', exact=False, by_tooltip=False, double=False):
    if False:
        return 10
    popup_dlg = explorer_app.window(class_name='NotifyIconOverflowWindow')
    try:
        popup_toolbar = popup_dlg.OverflowNotificationAreaToolbar.wait('visible')
        button_index = popup_toolbar.button(reqd_button, exact=exact, by_tooltip=by_tooltip).index
    except Exception:
        ShowHiddenIconsButton.click_input()
        popup_dlg = explorer_app.window(class_name='NotifyIconOverflowWindow')
        popup_toolbar = popup_dlg.OverflowNotificationAreaToolbar.wait('visible')
        button_index = popup_toolbar.button(reqd_button, exact=exact, by_tooltip=by_tooltip).index
    popup_toolbar.button(button_index).click_input(button=mouse_button, double=double)

def ClickSystemTrayIcon(button, exact=False, by_tooltip=False, double=False):
    if False:
        i = 10
        return i + 15
    'Click on a visible tray icon given by button'
    SystemTrayIcons.button(button, exact=exact, by_tooltip=by_tooltip).click_input(double=double)

def RightClickSystemTrayIcon(button, exact=False, by_tooltip=False):
    if False:
        while True:
            i = 10
    'Right click on a visible tray icon given by button'
    SystemTrayIcons.button(button, exact=exact, by_tooltip=by_tooltip).click_input(button='right')

def ClickHiddenSystemTrayIcon(button, exact=False, by_tooltip=False, double=False):
    if False:
        for i in range(10):
            print('nop')
    'Click on a hidden tray icon given by button'
    _click_hidden_tray_icon(button, exact=exact, by_tooltip=by_tooltip, double=double)

def RightClickHiddenSystemTrayIcon(button, exact=False, by_tooltip=False):
    if False:
        i = 10
        return i + 15
    'Right click on a hidden tray icon given by button'
    _click_hidden_tray_icon(button, mouse_button='right', exact=exact, by_tooltip=by_tooltip)
explorer_app = application.Application().connect(handle=TaskBarHandle())
TaskBar = explorer_app.window(handle=TaskBarHandle())
try:
    StartButton = explorer_app.window(name='Start', class_name='Button').wait('exists', 0.1)
except Exception:
    StartButton = TaskBar.Start
SystemTray = TaskBar.TrayNotifyWnd
Clock = TaskBar.TrayClockWClass
ShowDesktop = TaskBar.TrayShowDesktopButtonWClass
SystemTrayIcons = TaskBar.by(class_name='ToolbarWindow32', found_index=0)
RunningApplications = TaskBar.MSTaskListWClass
try:
    LangPanel = TaskBar.CiceroUIWndFrame.wait('exists', 0.1)
except Exception:
    LangPanel = TaskBar.TrayInputIndicatorWClass
ShowHiddenIconsButton = [ch for ch in TaskBar.children() if ch.friendly_class_name() == 'Button'][-1]