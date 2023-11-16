"""Tests for taskbar.py"""
import unittest
import sys
import os
sys.path.append('.')
from pywinauto import taskbar
from pywinauto import findwindows
from pywinauto.windows.application import Application
from pywinauto.windows.application import ProcessNotFoundError
from pywinauto.base_application import WindowSpecification
from pywinauto.sysinfo import is_x64_Python, is_x64_OS
from pywinauto.windows import win32defines
from pywinauto.timings import wait_until
import pywinauto.actionlogger
from pywinauto.timings import Timings
from pywinauto.controls.common_controls import ToolbarWrapper
from pywinauto import mouse
from pywinauto import Desktop
mfc_samples_folder = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples')
if is_x64_Python():
    mfc_samples_folder = os.path.join(mfc_samples_folder, 'x64')
_ready_timeout = 60
_retry_interval = 0.5

def _toggle_notification_area_icons(show_all=True, debug_img=None):
    if False:
        while True:
            i = 10
    '\n    A helper function to change \'Show All Icons\' settings.\n    On a succesful execution the function returns an original\n    state of \'Show All Icons\' checkbox.\n\n    The helper works only for an "English" version of Windows,\n    on non-english versions of Windows the \'Notification Area Icons\'\n    window should be accessed with a localized title"\n    '
    Application().start('explorer.exe')
    class_name = 'CabinetWClass'

    def _cabinetwclass_exist():
        if False:
            print('Hello World!')
        "Verify if at least one active 'CabinetWClass' window is created"
        l = findwindows.find_elements(active_only=True, class_name=class_name)
        return len(l) > 0
    wait_until(_ready_timeout, _retry_interval, _cabinetwclass_exist)
    handle = findwindows.find_elements(active_only=True, class_name=class_name)[-1].handle
    window = WindowSpecification({'handle': handle, 'backend': 'win32'})
    explorer = Application().connect(pid=window.process_id())
    cur_state = None
    try:
        cmd_str = 'control /name Microsoft.NotificationAreaIcons'
        for _ in range(3):
            window.wait('ready', timeout=_ready_timeout)
            window.AddressBandRoot.click_input()
            explorer.wait_cpu_usage_lower(threshold=2, timeout=_ready_timeout)
            window.type_keys(cmd_str, with_spaces=True, set_foreground=True)
            cmbx_spec = window.AddressBandRoot.ComboBoxEx
            if cmbx_spec.exists(timeout=_ready_timeout, retry_interval=_retry_interval):
                texts = cmbx_spec.texts()
                if texts and texts[0] == cmd_str:
                    break
            window.type_keys('{ESC}' * 3)
        window.type_keys('{ENTER}', with_spaces=True, set_foreground=True)
        explorer.wait_cpu_usage_lower(threshold=5, timeout=_ready_timeout)
        notif_area = Desktop().window(name='Notification Area Icons', class_name=class_name)
        notif_area.wait('ready', timeout=_ready_timeout)
        cur_state = notif_area.CheckBox.get_check_state()
        if bool(cur_state) != show_all:
            notif_area.CheckBox.click_input()
        notif_area.Ok.click_input()
        explorer.wait_cpu_usage_lower(threshold=5, timeout=_ready_timeout)
    except Exception as e:
        if debug_img:
            from PIL import ImageGrab
            ImageGrab.grab().save('%s.jpg' % debug_img, 'JPEG')
        l = pywinauto.actionlogger.ActionLogger()
        l.log('RuntimeError in _toggle_notification_area_icons')
        raise e
    finally:
        window.close()
    return cur_state

def _wait_minimized(dlg):
    if False:
        while True:
            i = 10
    "A helper function to verify that the specified dialog is minimized\n\n    Basically, WaitNot('visible', timeout=30) would work too, just\n    wanted to make sure the dlg is really got to the 'minimized' state\n    because we test hiding the window to the tray.\n    "
    wait_until(timeout=_ready_timeout, retry_interval=_retry_interval, func=lambda : dlg.get_show_state() == win32defines.SW_SHOWMINIMIZED)
    return True

class TaskbarTestCases(unittest.TestCase):
    """Unit tests for the taskbar"""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.tm = _ready_timeout
        app = Application(backend='win32')
        app.start(os.path.join(mfc_samples_folder, u'TrayMenu.exe'), wait_for_idle=False)
        self.app = app
        self.dlg = app.top_window()
        mouse.move((-500, 200))
        self.dlg.wait('ready', timeout=self.tm)

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Close the application after tests'
        self.dlg.send_message(win32defines.WM_CLOSE)
        self.dlg.wait_not('ready')
        l = pywinauto.actionlogger.ActionLogger()
        try:
            for i in range(2):
                l.log('Look for unclosed sample apps')
                app = Application()
                app.connect(path='TrayMenu.exe')
                l.log('Forse closing a leftover app: {0}'.format(app))
                app.kill()
        except ProcessNotFoundError:
            l.log('No more leftovers. All good.')

    def testTaskbar(self):
        if False:
            i = 10
            return i + 15
        taskbar.TaskBar.wait('visible', timeout=self.tm)
    '\n    def testStartButton(self): # TODO: fix it for AppVeyor\n        taskbar.StartButton.click_input()\n\n        sample_app_exe = os.path.join(mfc_samples_folder, u"TrayMenu.exe")\n        start_menu = taskbar.explorer_app.window(class_name=\'DV2ControlHost\')\n        start_menu.SearchEditBoxWrapperClass.click_input()\n        start_menu.SearchEditBoxWrapperClass.type_keys(\n           sample_app_exe() + \'{ENTER}\',\n           with_spaces=True, set_foreground=False\n           )\n\n        time.sleep(5)\n        app = Application.connect(path=sample_app_exe())\n        dlg = app.top_window()\n        Wait(\'ready\', timeout=self.tm)\n    '

    def testSystemTray(self):
        if False:
            i = 10
            return i + 15
        taskbar.SystemTray.wait('visible', timeout=self.tm)

    def testClock(self):
        if False:
            while True:
                i = 10
        'Test opening/closing of a system clock applet'
        self.dlg.minimize()
        _wait_minimized(self.dlg)
        taskbar.Clock.click_input()
        ClockWindow = taskbar.explorer_app.window(class_name='ClockFlyoutWindow')
        ClockWindow.wait('visible', timeout=self.tm)
        taskbar.Clock.type_keys('{ESC}', set_foreground=False)
        ClockWindow.wait_not('visible', timeout=self.tm)

    def testClickVisibleIcon(self):
        if False:
            return 10
        '\n        Test minimizing a sample app into the visible area of the tray\n        and restoring the app back\n        '
        if is_x64_Python() != is_x64_OS():
            return
        orig_hid_state = _toggle_notification_area_icons(show_all=True, debug_img='%s_01' % self.id())
        self.dlg.minimize()
        _wait_minimized(self.dlg)
        menu_window = [None]

        def _show_popup_menu():
            if False:
                for i in range(10):
                    print('nop')
            taskbar.explorer_app.wait_cpu_usage_lower(threshold=5, timeout=self.tm)
            taskbar.RightClickSystemTrayIcon('MFCTrayDemo')
            children = self.app.top_window().children()
            if not children:
                menu = self.app.windows(visible=True)[0].children()[0]
            else:
                menu = children[0]
            res = isinstance(menu, ToolbarWrapper) and menu.is_visible()
            menu_window[0] = menu
            return res
        wait_until(self.tm, _retry_interval, _show_popup_menu)
        menu_window[0].menu_bar_click_input('#2', self.app)
        popup_window = self.app.top_window()
        hdl = self.dlg.popup_window()
        self.assertEqual(popup_window.handle, hdl)
        taskbar.ClickSystemTrayIcon('MFCTrayDemo', double=True)
        self.dlg.wait('active', timeout=self.tm)
        _toggle_notification_area_icons(show_all=orig_hid_state, debug_img='%s_02' % self.id())

    def testClickHiddenIcon(self):
        if False:
            return 10
        '\n        Test minimizing a sample app into the hidden area of the tray\n        and restoring the app back\n        '
        if is_x64_Python() != is_x64_OS():
            return
        orig_hid_state = _toggle_notification_area_icons(show_all=False, debug_img='%s_01' % self.id())
        self.dlg.minimize()
        _wait_minimized(self.dlg)
        app2 = Application()
        app2.start(os.path.join(mfc_samples_folder, u'TrayMenu.exe'))
        dlg2 = app2.top_window()
        dlg2.wait('visible', timeout=self.tm)
        dlg2.minimize()
        _wait_minimized(dlg2)
        taskbar.explorer_app.wait_cpu_usage_lower(threshold=5, timeout=40)
        taskbar.ClickHiddenSystemTrayIcon('MFCTrayDemo', double=True)
        self.dlg.wait('visible', timeout=self.tm)
        _toggle_notification_area_icons(show_all=orig_hid_state, debug_img='%s_02' % self.id())
        dlg2.send_message(win32defines.WM_CLOSE)

    def testClickCustomizeButton(self):
        if False:
            for i in range(10):
                print('nop')
        "Test click on the 'show hidden icons' button"
        self.dlg.minimize()
        _wait_minimized(self.dlg)
        orig_hid_state = _toggle_notification_area_icons(show_all=False, debug_img='%s_01' % self.id())
        app2 = Application()
        app2.start(os.path.join(mfc_samples_folder, u'TrayMenu.exe'))
        dlg2 = app2.top_window()
        dlg2.wait('visible', timeout=self.tm)
        dlg2.minimize()
        _wait_minimized(dlg2)
        taskbar.ShowHiddenIconsButton.click_input()
        niow_dlg = taskbar.explorer_app.window(class_name='NotifyIconOverflowWindow')
        niow_dlg.OverflowNotificationAreaToolbar.wait('ready', timeout=self.tm)
        niow_dlg.SysLink.click_input()
        nai = Desktop().window(name='Notification Area Icons', class_name='CabinetWClass')
        nai.wait('ready')
        origAlwaysShow = nai.CheckBox.get_check_state()
        if not origAlwaysShow:
            nai.CheckBox.click_input()
        nai.OK.click()
        _toggle_notification_area_icons(show_all=orig_hid_state, debug_img='%s_02' % self.id())
        dlg2.send_message(win32defines.WM_CLOSE)
if __name__ == '__main__':
    unittest.main()