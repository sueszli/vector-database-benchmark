"""Tests for HwndWrapper"""
from __future__ import print_function
from __future__ import unicode_literals
import six
import time
import ctypes
import locale
import sys
import os
import unittest
sys.path.append('.')
import mock
from pywinauto.application import Application
from pywinauto.controls.hwndwrapper import HwndWrapper
from pywinauto.controls.hwndwrapper import InvalidWindowHandle
from pywinauto.controls.hwndwrapper import get_dialog_props_from_handle
from pywinauto.windows import win32structures
from pywinauto.windows import win32defines
from pywinauto.findwindows import ElementNotFoundError
from pywinauto.sysinfo import is_x64_Python
from pywinauto.sysinfo import is_x64_OS
from pywinauto.windows.remote_memory_block import RemoteMemoryBlock
from pywinauto.timings import Timings
from pywinauto import clipboard
from pywinauto.base_wrapper import ElementNotEnabled
from pywinauto.base_wrapper import ElementNotVisible
from pywinauto import findbestmatch
from pywinauto import keyboard
from pywinauto import Desktop
from pywinauto import timings
from pywinauto import WindowNotFoundError
mfc_samples_folder = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples')
if is_x64_Python():
    mfc_samples_folder = os.path.join(mfc_samples_folder, 'x64')

def _notepad_exe():
    if False:
        for i in range(10):
            print('nop')
    if is_x64_Python() or not is_x64_OS():
        return 'C:\\Windows\\System32\\notepad.exe'
    else:
        return 'C:\\Windows\\SysWOW64\\notepad.exe'

class HwndWrapperTests(unittest.TestCase):
    """Unit tests for the HwndWrapper class"""

    def setUp(self):
        if False:
            print('Hello World!')
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application().start(os.path.join(mfc_samples_folder, u'CmnCtrl3.exe'))
        self.dlg = self.app.Common_Controls_Sample
        self.dlg.TabControl.select('CButton (Command Link)')
        self.ctrl = HwndWrapper(self.dlg.Command_button_here.handle)

    def test_get_active_hwnd(self):
        if False:
            print('Hello World!')
        focused_element = self.dlg.find().get_active()
        self.assertTrue(type(focused_element) is HwndWrapper or issubclass(type(focused_element), HwndWrapper))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the application after tests'
        self.app.kill()

    def test_close_not_found(self):
        if False:
            print('Hello World!')
        'Test dialog close handle non existing window'
        wrp = self.dlg.find()
        with mock.patch.object(timings, 'wait_until') as mock_wait_until:
            mock_wait_until.side_effect = timings.TimeoutError
            self.assertRaises(WindowNotFoundError, wrp.close)

    def test_scroll(self):
        if False:
            for i in range(10):
                print('nop')
        'Test control scrolling'
        self.dlg.TabControl.select('CNetworkAddressCtrl')
        ctrl = HwndWrapper(self.dlg.TypeListBox.handle)
        self.assertRaises(ValueError, ctrl.scroll, 'bbbb', 'line')
        self.assertRaises(ValueError, ctrl.scroll, 'left', 'aaaa')
        self.assertEqual(ctrl.item_rect(0).top, 0)
        ctrl.scroll('down', 'page', 2)
        self.assertEqual(ctrl.item_rect(0).top < -10, True)

    def testInvalidHandle(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that an exception is raised with an invalid window handle'
        self.assertRaises(InvalidWindowHandle, HwndWrapper, -1)

    def testFriendlyClassName(self):
        if False:
            i = 10
            return i + 15
        'Test getting the friendly classname of the control'
        self.assertEqual(self.ctrl.friendly_class_name(), 'Button')

    def testClass(self):
        if False:
            i = 10
            return i + 15
        'Test getting the classname of the control'
        self.assertEqual(self.ctrl.class_name(), 'Button')

    def testWindowText(self):
        if False:
            while True:
                i = 10
        'Test getting the window Text of the control'
        self.assertEqual(HwndWrapper(self.dlg.Set.handle).window_text(), u'Set')

    def testStyle(self):
        if False:
            while True:
                i = 10
        self.dlg.style()
        self.assertEqual(self.ctrl.style(), win32defines.WS_CHILD | win32defines.WS_VISIBLE | win32defines.WS_TABSTOP | win32defines.BS_COMMANDLINK)

    def testExStyle(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.ctrl.exstyle(), win32defines.WS_EX_NOPARENTNOTIFY | win32defines.WS_EX_LEFT | win32defines.WS_EX_LTRREADING | win32defines.WS_EX_RIGHTSCROLLBAR)

    def testControlID(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.ctrl.control_id(), 1037)
        self.dlg.control_id()

    def testUserData(self):
        if False:
            return 10
        self.ctrl.user_data()
        self.dlg.user_data()

    def testContextHelpID(self):
        if False:
            return 10
        self.ctrl.context_help_id()
        self.dlg.context_help_id()

    def testIsVisible(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.ctrl.is_visible(), True)
        self.assertEqual(self.dlg.is_visible(), True)

    def testIsUnicode(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.ctrl.is_unicode(), True)
        self.assertEqual(self.dlg.is_unicode(), True)

    def testIsEnabled(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.ctrl.is_enabled(), True)
        self.assertEqual(self.dlg.is_enabled(), True)

    def testRectangle(self):
        if False:
            print('Hello World!')
        'Test getting the rectangle of the dialog'
        rect = self.dlg.rectangle()
        self.assertNotEqual(rect.top, None)
        self.assertNotEqual(rect.left, None)
        self.assertNotEqual(rect.bottom, None)
        self.assertNotEqual(rect.right, None)
        if abs(rect.height() - 423) > 5:
            self.assertEqual(rect.height(), 423)
        if abs(rect.width() - 506) > 5:
            self.assertEqual(rect.width(), 506)

    def testClientRect(self):
        if False:
            while True:
                i = 10
        rect = self.dlg.rectangle()
        cli = self.dlg.client_rect()
        self.assertEqual(cli.left, 0)
        self.assertEqual(cli.top, 0)
        assert cli.width() < rect.width()
        assert cli.height() < rect.height()

    def testFont(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(self.dlg.font(), self.ctrl.font())

    def testProcessID(self):
        if False:
            return 10
        self.assertEqual(self.ctrl.process_id(), self.dlg.process_id())
        self.assertNotEqual(self.ctrl.process_id(), 0)

    def testHasStyle(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.ctrl.has_style(win32defines.WS_CHILD), True)
        self.assertEqual(self.dlg.has_style(win32defines.WS_CHILD), False)
        self.assertEqual(self.ctrl.has_style(win32defines.WS_SYSMENU), False)
        self.assertEqual(self.dlg.has_style(win32defines.WS_SYSMENU), True)

    def testHasExStyle(self):
        if False:
            return 10
        self.assertEqual(self.ctrl.has_exstyle(win32defines.WS_EX_NOPARENTNOTIFY), True)
        self.assertEqual(self.dlg.has_exstyle(win32defines.WS_EX_NOPARENTNOTIFY), False)
        self.assertEqual(self.ctrl.has_exstyle(win32defines.WS_EX_APPWINDOW), False)

    def testIsDialog(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.ctrl.is_dialog(), False)
        self.assertEqual(self.dlg.is_dialog(), True)

    def testParent(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.ctrl.parent().parent(), self.dlg.handle)

    def testTopLevelParent(self):
        if False:
            return 10
        self.assertEqual(self.ctrl.top_level_parent(), self.dlg.handle)
        self.assertEqual(self.dlg.top_level_parent(), self.dlg.handle)

    def test_get_active_desktop_hwnd(self):
        if False:
            print('Hello World!')
        focused_element = Desktop(backend='win32').get_active()
        self.assertTrue(type(focused_element) is HwndWrapper or issubclass(type(focused_element), HwndWrapper))

    def testTexts(self):
        if False:
            return 10
        self.assertEqual(self.dlg.texts(), ['Common Controls Sample'])
        self.assertEqual(HwndWrapper(self.dlg.Show.handle).texts(), [u'Show'])
        self.assertEqual(self.dlg.by(class_name='Button', found_index=2).texts(), [u'Elevation Icon'])

    def testFoundIndex(self):
        if False:
            print('Hello World!')
        'Test an access to a control by found_index'
        ctl = self.dlg.by(class_name='Button', found_index=3)
        self.assertEqual(ctl.texts(), [u'Show'])
        ctl.draw_outline('blue')
        ctl = self.dlg.by(class_name='Button', found_index=3333)
        self.assertRaises(ElementNotFoundError, ctl.find)

    def testSearchWithPredicateFunc(self):
        if False:
            while True:
                i = 10
        'Test an access to a control by filtering with a predicate function'

        def is_checkbox(elem):
            if False:
                i = 10
                return i + 15
            res = False
            if elem.handle is None:
                return False
            hwwrp = HwndWrapper(elem.handle)
            if hwwrp.friendly_class_name() == u'CheckBox':
                if hwwrp.texts() == [u'Show']:
                    res = True
            return res
        ctl = self.dlg.by(predicate_func=is_checkbox)
        self.assertEqual(ctl.texts(), [u'Show'])
        ctl.draw_outline('red')

    def testClientRects(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.ctrl.client_rects()[0], self.ctrl.client_rect())
        self.assertEqual(self.dlg.client_rects()[0], self.dlg.client_rect())

    def testFonts(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.ctrl.fonts()[0], self.ctrl.font())
        self.assertEqual(self.dlg.fonts()[0], self.dlg.font())

    def testChildren(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.ctrl.children(), [])
        self.assertNotEqual(self.dlg.children(), [])

    def testIsChild(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.ctrl.is_child(self.dlg.find()), True)
        self.assertEqual(self.dlg.is_child(self.ctrl), False)

    def testSendMessage(self):
        if False:
            while True:
                i = 10
        vk = self.dlg.send_message(win32defines.WM_GETDLGCODE)
        self.assertEqual(0, vk)
        code = self.dlg.Edit.send_message(win32defines.WM_GETDLGCODE)
        expected = 137
        self.assertEqual(expected, code)

    def test_send_chars(self):
        if False:
            while True:
                i = 10
        testString = 'Hello World'
        self.dlg.minimize()
        self.dlg.Edit.send_chars(testString)
        actual = self.dlg.Edit.texts()[0]
        expected = 'Hello World'
        self.assertEqual(expected, actual)

    def test_send_chars_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(keyboard.KeySequenceError):
            testString = 'Hello{LEFT 2}{DEL 2}'
            self.dlg.minimize()
            self.dlg.Edit.send_chars(testString)

    def test_send_keystrokes_multikey_characters(self):
        if False:
            i = 10
            return i + 15
        testString = 'Hawaii#{%}@$'
        self.dlg.minimize()
        self.dlg.Edit.send_keystrokes(testString)
        actual = self.dlg.Edit.texts()[0]
        expected = 'Hawaii#%@$'
        self.assertEqual(expected, actual)

    def test_send_keystrokes_virtual_keys_left_del_back(self):
        if False:
            return 10
        testString = '+hello123{LEFT 2}{DEL 2}{BACKSPACE} +world'
        self.dlg.minimize()
        self.dlg.Edit.send_keystrokes(testString)
        actual = self.dlg.Edit.texts()[0]
        expected = 'Hello World'
        self.assertEqual(expected, actual)

    def test_send_keystrokes_virtual_keys_shift(self):
        if False:
            for i in range(10):
                print('nop')
        testString = '+hello +world'
        self.dlg.minimize()
        self.dlg.Edit.send_keystrokes(testString)
        actual = self.dlg.Edit.texts()[0]
        expected = 'Hello World'
        self.assertEqual(expected, actual)

    def test_send_keystrokes_virtual_keys_ctrl(self):
        if False:
            print('Hello World!')
        testString = '^a^c{RIGHT}^v'
        self.dlg.minimize()
        self.dlg.Edit.send_keystrokes(testString)
        actual = self.dlg.Edit.texts()[0]
        expected = 'and the note goes here ...and the note goes here ...'
        self.assertEqual(expected, actual)

    def testSendMessageTimeout(self):
        if False:
            return 10
        default_timeout = Timings.sendmessagetimeout_timeout
        Timings.sendmessagetimeout_timeout = 0.1
        vk = self.dlg.send_message_timeout(win32defines.WM_GETDLGCODE)
        self.assertEqual(0, vk)
        code = self.dlg.Show.send_message_timeout(win32defines.WM_GETDLGCODE)
        expected = 8192
        Timings.sendmessagetimeout_timeout = default_timeout
        self.assertEqual(expected, code)

    def testPostMessage(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(0, self.dlg.post_message(win32defines.WM_PAINT))
        self.assertNotEqual(0, self.dlg.Show.post_message(win32defines.WM_PAINT))

    def testNotifyParent(self):
        if False:
            i = 10
            return i + 15
        'Call notify_parent to ensure it does not raise'
        self.ctrl.notify_parent(1234)

    def testGetProperties(self):
        if False:
            while True:
                i = 10
        'Test getting the properties for the HwndWrapped control'
        props = self.dlg.get_properties()
        self.assertEqual(self.dlg.friendly_class_name(), props['friendly_class_name'])
        self.assertEqual(self.dlg.texts(), props['texts'])
        for prop_name in props:
            self.assertEqual(getattr(self.dlg, prop_name)(), props[prop_name])

    def test_capture_as_image_multi_monitor(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('win32api.EnumDisplayMonitors') as mon_device:
            mon_device.return_value = (1, 2)
            rect = self.dlg.rectangle()
            expected = (rect.width(), rect.height())
            result = self.dlg.capture_as_image().size
            self.assertEqual(expected, result)

    def testEquals(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(self.ctrl, self.dlg.handle)
        self.assertEqual(self.ctrl, self.ctrl.handle)
        self.assertEqual(self.ctrl, self.ctrl)

    def testMoveWindow_same(self):
        if False:
            print('Hello World!')
        'Test calling move_window without any parameters'
        prevRect = self.dlg.rectangle()
        self.dlg.move_window()
        self.assertEqual(prevRect, self.dlg.rectangle())

    def testMoveWindow(self):
        if False:
            i = 10
            return i + 15
        'Test moving the window'
        dlgClientRect = self.ctrl.parent().rectangle()
        prev_rect = self.ctrl.rectangle() - dlgClientRect
        new_rect = win32structures.RECT(prev_rect)
        new_rect.left -= 1
        new_rect.top -= 1
        new_rect.right += 2
        new_rect.bottom += 2
        self.ctrl.move_window(new_rect.left, new_rect.top, new_rect.width(), new_rect.height())
        time.sleep(0.1)
        print('prev_rect = ', prev_rect)
        print('new_rect = ', new_rect)
        print('dlgClientRect = ', dlgClientRect)
        print('self.ctrl.rectangle() = ', self.ctrl.rectangle())
        self.assertEqual(self.ctrl.rectangle(), new_rect + dlgClientRect)
        self.ctrl.move_window(prev_rect)
        self.assertEqual(self.ctrl.rectangle(), prev_rect + dlgClientRect)

    def testMaximize(self):
        if False:
            i = 10
            return i + 15
        self.dlg.maximize()
        self.assertEqual(self.dlg.get_show_state(), win32defines.SW_SHOWMAXIMIZED)
        self.dlg.restore()

    def testMinimize(self):
        if False:
            print('Hello World!')
        self.dlg.minimize()
        self.assertEqual(self.dlg.get_show_state(), win32defines.SW_SHOWMINIMIZED)
        self.dlg.restore()

    def testRestore(self):
        if False:
            return 10
        self.dlg.maximize()
        self.dlg.restore()
        self.assertEqual(self.dlg.get_show_state(), win32defines.SW_SHOWNORMAL)
        self.dlg.minimize()
        self.dlg.restore()
        self.assertEqual(self.dlg.get_show_state(), win32defines.SW_SHOWNORMAL)

    def testGetFocus(self):
        if False:
            return 10
        self.assertNotEqual(self.dlg.get_focus(), None)
        self.assertEqual(self.dlg.get_focus(), self.ctrl.get_focus())
        self.dlg.Set.set_focus()
        self.assertEqual(self.dlg.get_focus(), self.dlg.Set.handle)

    def test_issue_318(self):
        if False:
            return 10
        self.dlg.restore()
        self.dlg.minimize()
        self.dlg.set_focus()
        self.assertTrue(self.dlg.is_normal())
        self.assertTrue(self.dlg.is_active())
        self.dlg.maximize()
        self.dlg.minimize()
        self.dlg.set_focus()
        self.assertTrue(self.dlg.is_maximized())
        self.assertTrue(self.dlg.is_active())
        self.dlg.restore()

    def testSetFocus(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(self.dlg.get_focus(), self.dlg.Set.handle)
        self.dlg.Set.set_focus()
        self.assertEqual(self.dlg.get_focus(), self.dlg.Set.handle)

    def testHasKeyboardFocus(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.dlg.set.has_keyboard_focus())
        self.dlg.set.set_keyboard_focus()
        self.assertTrue(self.dlg.set.has_keyboard_focus())

    def testSetKeyboardFocus(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(self.dlg.get_focus(), self.dlg.set.handle)
        self.dlg.set.set_keyboard_focus()
        self.assertEqual(self.dlg.get_focus(), self.dlg.set.handle)

    def test_pretty_print(self):
        if False:
            print('Hello World!')
        'Test __str__ method for HwndWrapper based controls'
        if six.PY3:
            assert_regex = self.assertRegex
        else:
            assert_regex = self.assertRegexpMatches
        wrp = self.dlg.find()
        assert_regex(wrp.__str__(), "^hwndwrapper.DialogWrapper - 'Common Controls Sample', Dialog$")
        assert_regex(wrp.__repr__(), "^<hwndwrapper.DialogWrapper - 'Common Controls Sample', Dialog, [0-9-]+>$")
        wrp = self.ctrl
        assert_regex(wrp.__str__(), "^win32_controls.ButtonWrapper - 'Command button here', Button$")
        assert_regex(wrp.__repr__(), "^<win32_controls.ButtonWrapper - 'Command button here', Button, [0-9-]+>$")
        wrp = self.dlg.TabControl.find()
        assert_regex(wrp.__str__(), "^common_controls.TabControlWrapper - '', TabControl$")
        assert_regex(wrp.__repr__(), "^<common_controls.TabControlWrapper - '', TabControl, [0-9-]+>$")

    def test_children_generator(self):
        if False:
            return 10
        dlg = self.dlg.find()
        children = [child for child in dlg.iter_children()]
        self.assertSequenceEqual(dlg.children(), children)

class SendKeystrokesTests(unittest.TestCase):
    """Unit tests for the SendKeyStrokes class"""

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        notepad2_mod_folder = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\Notepad2-mod')
        if is_x64_Python():
            notepad2_mod_folder = os.path.join(notepad2_mod_folder, 'x64')
        self.app = Application().start(os.path.join(notepad2_mod_folder, u'Notepad2.exe'))
        self.dlg = self.app.window(name_re='.*Untitled - Notepad2-mod', visible=None)
        self.ctrl = HwndWrapper(self.dlg.Scintilla.handle)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Close the application after tests'
        self.app.kill()

    def test_send_keystrokes_enter(self):
        if False:
            while True:
                i = 10
        expected = 'some test string'
        self.dlg.minimize()
        self.ctrl.send_keystrokes(expected)
        self.ctrl.send_keystrokes('{ENTER}')
        self.dlg.restore()
        actual = self.ctrl.window_text()
        self.assertEqual(expected + '\r\n', actual)

class HwndWrapperMenuTests(unittest.TestCase):
    """Unit tests for menu actions of the HwndWrapper class"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application().start(os.path.join(mfc_samples_folder, u'RowList.exe'))
        self.dlg = self.app.RowListSampleApplication
        self.ctrl = self.app.RowListSampleApplication.ListView.find()

    def tearDown(self):
        if False:
            return 10
        'Close the application after tests'
        self.dlg.send_message(win32defines.WM_CLOSE)

    def testMenuItems(self):
        if False:
            i = 10
            return i + 15
        'Test getting menu items'
        self.assertEqual(self.ctrl.menu_items(), [])
        self.assertEqual(self.dlg.menu_items()[1]['text'], '&View')

    def testMenuSelect(self):
        if False:
            print('Hello World!')
        'Test selecting a menu item'
        if self.dlg.menu_item('View -> Toolbar').is_checked():
            self.dlg.menu_select('View -> Toolbar')
        self.assertEqual(self.dlg.menu_item('View -> Toolbar').is_checked(), False)
        self.dlg.menu_select('View -> Toolbar')
        self.assertEqual(self.dlg.menu_item('View -> Toolbar').is_checked(), True)

    def testClose(self):
        if False:
            i = 10
            return i + 15
        'Test the Close() method of windows'
        self.dlg.menu_select('Help->About RowList...')
        self.app.AboutRowList.wait('visible', 20)
        self.assertTrue(self.app.window(name='About RowList').is_visible(), True)
        self.app.window(name='About RowList', class_name='#32770').close(1)
        try:
            self.app.window(name='About RowList', class_name='#32770').find()
        except ElementNotFoundError:
            print('ElementNotFoundError exception is raised as expected. OK.')
        self.assertEqual(self.dlg.is_visible(), True)

    def testCloseClick_bug(self):
        if False:
            print('Hello World!')
        self.dlg.menu_select('Help->About RowList...')
        self.app.AboutRowList.wait('visible', 10)
        self.assertEqual(self.app.AboutRowList.exists(), True)
        self.app.AboutRowList.CloseButton.close_click()
        self.assertEqual(self.app.AboutRowList.exists(), False)

    def testCloseAltF4(self):
        if False:
            i = 10
            return i + 15
        self.dlg.menu_select('Help->About RowList...')
        AboutRowList = self.app.window(name='About RowList', active_only=True, class_name='#32770')
        AboutWrapper = AboutRowList.wait('enabled')
        AboutRowList.close_alt_f4()
        AboutRowList.wait_not('visible')
        self.assertNotEqual(AboutWrapper.is_visible(), True)

class HwndWrapperMouseTests(unittest.TestCase):
    """Unit tests for mouse actions of the HwndWrapper class"""

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application().start(os.path.join(mfc_samples_folder, u'CmnCtrl3.exe'))
        self.dlg = self.app.Common_Controls_Sample
        self.dlg.TabControl.select('CButton (Command Link)')
        self.ctrl = HwndWrapper(self.dlg.NoteEdit.handle)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Close the application after tests'
        try:
            self.dlg.close(0.5)
        except Exception:
            pass
        finally:
            self.app.kill()

    def testClick(self):
        if False:
            while True:
                i = 10
        self.ctrl.click(coords=(50, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (9, 9))

    def testClickInput(self):
        if False:
            return 10
        self.ctrl.click_input(coords=(50, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (9, 9))

    def testDoubleClick(self):
        if False:
            return 10
        self.ctrl.double_click(coords=(50, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (8, 13))

    def testDoubleClickInput(self):
        if False:
            i = 10
            return i + 15
        self.ctrl.double_click_input(coords=(80, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (13, 18))

    def testRightClickInput(self):
        if False:
            while True:
                i = 10
        self.dlg.Edit.type_keys('{HOME}')
        self.dlg.Edit.wait('enabled').right_click_input()
        self.app.PopupMenu.wait('ready').menu().get_menu_path('Select All')[0].click_input()
        self.dlg.Edit.type_keys('{DEL}')
        self.assertEqual(self.dlg.Edit.text_block(), '')

    def testPressMoveRelease(self):
        if False:
            return 10
        self.dlg.NoteEdit.press_mouse(coords=(0, 5))
        self.dlg.NoteEdit.move_mouse(coords=(65, 5))
        self.dlg.NoteEdit.release_mouse(coords=(65, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (0, 12))

    def testDragMouse(self):
        if False:
            return 10
        self.dlg.NoteEdit.drag_mouse(press_coords=(0, 5), release_coords=(65, 5))
        self.assertEqual(self.dlg.Edit.selection_indices(), (0, 12))
        self.dlg.NoteEdit.drag_mouse(press_coords=(65, 5), release_coords=(90, 5), pressed='shift')
        self.assertEqual(self.dlg.Edit.selection_indices(), (0, 17))

    def testDebugMessage(self):
        if False:
            while True:
                i = 10
        self.dlg.NoteEdit.debug_message('Test message')

    def testSetTransparency(self):
        if False:
            print('Hello World!')
        self.dlg.set_transparency()
        self.assertRaises(ValueError, self.dlg.set_transparency, 256)

class NonActiveWindowFocusTests(unittest.TestCase):
    """Regression unit tests for setting focus"""

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application()
        self.app.start(os.path.join(mfc_samples_folder, u'CmnCtrl3.exe'))
        self.app2 = Application().start(_notepad_exe())

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Close the application after tests'
        self.app.kill()
        self.app2.kill()

    def test_issue_240(self):
        if False:
            i = 10
            return i + 15
        'Check HwndWrapper.set_focus for a desktop without a focused window'
        ws = self.app.Common_Controls_Sample
        ws.TabControl.select('CButton (Command Link)')
        dlg1 = ws.find()
        dlg2 = self.app2.Notepad.find()
        dlg2.click(coords=(2, 2))
        dlg2.minimize()
        dlg2.restore()
        dlg1.set_focus()
        self.assertEqual(ws.get_focus(), ws.Edit.find())

class WindowWithoutMessageLoopFocusTests(unittest.TestCase):
    """
    Regression unit tests for setting focus when window does not have
    a message loop.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app1 = Application().start(u'cmd.exe', create_new_console=True, wait_for_idle=False)
        self.app2 = Application().start(os.path.join(mfc_samples_folder, u'CmnCtrl2.exe'))
        self.app2.wait_cpu_usage_lower(threshold=1.5, timeout=30, usage_interval=1)

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Close the application after tests'
        self.app1.kill()
        self.app2.kill()

    def test_issue_270(self):
        if False:
            while True:
                i = 10
        '\n        Set focus to a window without a message loop, then switch to a window\n        with one and type in it.\n        '
        self.app1.window().set_focus()
        self.app1.wait_cpu_usage_lower(threshold=1.5, timeout=30, usage_interval=1)
        self.app2.window().edit.type_keys('1')
        self.assertTrue(self.app2.window().is_active())

class NotepadRegressionTests(unittest.TestCase):
    """Regression unit tests for Notepad"""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application()
        self.app.start(_notepad_exe())
        self.dlg = self.app.window(name='Untitled - Notepad', class_name='Notepad')
        self.ctrl = HwndWrapper(self.dlg.Edit.handle)
        self.dlg.Edit.set_edit_text('Here is some text\r\n and some more')
        self.app2 = Application().start(_notepad_exe())

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the application after tests'
        try:
            self.app.UntitledNotepad.menu_select('File->Exit')
            self.app.Notepad["Do&n't Save"].click()
            self.app.Notepad["Do&n't Save"].wait_not('visible')
        except Exception:
            pass
        finally:
            self.app.kill()
        self.app2.kill()

    def testMenuSelectNotepad_bug(self):
        if False:
            print('Hello World!')
        'In notepad - MenuSelect Edit->Paste did not work'
        text = b'Here are some unicode characters \xef\xfc\r\n'
        self.app2.UntitledNotepad.Edit.wait('enabled')
        time.sleep(0.3)
        self.app2.UntitledNotepad.Edit.set_edit_text(text)
        time.sleep(0.3)
        self.assertEqual(self.app2.UntitledNotepad.Edit.text_block().encode(locale.getpreferredencoding()), text)
        Timings.after_menu_wait = 0.7
        self.app2.UntitledNotepad.menu_select('Edit->Select All')
        time.sleep(0.3)
        self.app2.UntitledNotepad.menu_select('Edit->Copy')
        time.sleep(0.3)
        self.assertEqual(clipboard.GetData().encode(locale.getpreferredencoding()), text)
        self.dlg.set_focus()
        self.dlg.menu_select('Edit->Select All')
        self.dlg.menu_select('Edit->Paste')
        self.dlg.menu_select('Edit->Paste')
        self.dlg.menu_select('Edit->Paste')
        self.app2.UntitledNotepad.menu_select('File->Exit')
        self.app2.window(name='Notepad', class_name='#32770')["Don't save"].click()
        self.assertEqual(self.dlg.Edit.text_block().encode(locale.getpreferredencoding()), text * 3)

class ControlStateTests(unittest.TestCase):
    """Unit tests for control states"""

    def setUp(self):
        if False:
            return 10
        'Start the application set some data and ensure the application\n        is in the state we want it.\n        '
        self.app = Application()
        self.app.start(os.path.join(mfc_samples_folder, u'CmnCtrl1.exe'))
        self.dlg = self.app.Common_Controls_Sample
        self.dlg.TabControl.select(4)
        self.ctrl = self.dlg.EditBox.find()

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Close the application after tests'
        self.app.kill()

    def test_VerifyEnabled(self):
        if False:
            i = 10
            return i + 15
        'Test for verify_enabled'
        self.assertRaises(ElementNotEnabled, self.ctrl.verify_enabled)

    def test_VerifyVisible(self):
        if False:
            return 10
        'Test for verify_visible'
        self.dlg.TabControl.select(3)
        self.assertRaises(ElementNotVisible, self.ctrl.verify_visible)

class DragAndDropTests(unittest.TestCase):
    """Unit tests for mouse actions like drag-n-drop"""

    def setUp(self):
        if False:
            print('Hello World!')
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application()
        self.app.start(os.path.join(mfc_samples_folder, u'CmnCtrl1.exe'))
        self.dlg = self.app.Common_Controls_Sample
        self.ctrl = self.dlg.TreeView.find()

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Close the application after tests'
        self.app.kill()

    def testDragMouseInput(self):
        if False:
            i = 10
            return i + 15
        'Test for drag_mouse_input'
        birds = self.ctrl.get_item('\\Birds')
        dogs = self.ctrl.get_item('\\Dogs')
        birds.click_input()
        time.sleep(5)
        self.ctrl.drag_mouse_input(dst=dogs.client_rect().mid_point(), src=birds.client_rect().mid_point(), absolute=False)
        dogs = self.ctrl.get_item('\\Dogs')
        self.assertEqual([child.text() for child in dogs.children()], [u'Birds', u'Dalmatian', u'German Shepherd', u'Great Dane'])

class GetDialogPropsFromHandleTest(unittest.TestCase):
    """Unit tests for mouse actions of the HwndWrapper class"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application()
        self.app.start(_notepad_exe())
        self.dlg = self.app.UntitledNotepad
        self.ctrl = HwndWrapper(self.dlg.Edit.handle)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the application after tests'
        self.dlg.close(0.5)
        self.app.kill()

    def test_GetDialogPropsFromHandle(self):
        if False:
            while True:
                i = 10
        'Test some small stuff regarding GetDialogPropsFromHandle'
        props_from_handle = get_dialog_props_from_handle(self.dlg.handle)
        props_from_dialog = get_dialog_props_from_handle(self.dlg)
        self.assertEqual(props_from_handle, props_from_dialog)

class SendEnterKeyTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application()
        self.app.start(_notepad_exe())
        self.dlg = self.app.UntitledNotepad
        self.ctrl = HwndWrapper(self.dlg.Edit.handle)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.dlg.menu_select('File -> Exit')
        try:
            self.app.Notepad["Do&n't Save"].click()
        except findbestmatch.MatchError:
            self.app.kill()

    def test_sendEnterChar(self):
        if False:
            print('Hello World!')
        self.ctrl.send_chars('Hello{ENTER}World')
        self.assertEqual('Hello\r\nWorld', self.dlg.Edit.window_text())

class SendKeystrokesAltComboTests(unittest.TestCase):
    """Unit test for Alt- combos sent via send_keystrokes"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        Timings.defaults()
        self.app = Application().start(os.path.join(mfc_samples_folder, u'CtrlTest.exe'))
        self.dlg = self.app.Control_Test_App

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.app.kill()

    def test_send_keystrokes_alt_combo(self):
        if False:
            for i in range(10):
                print('nop')
        self.dlg.send_keystrokes('%(sc)')
        self.assertTrue(self.app['Using C++ Derived Class'].exists())

class RemoteMemoryBlockTests(unittest.TestCase):
    """Unit tests for RemoteMemoryBlock"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        self.app = Application()
        self.app.start(os.path.join(mfc_samples_folder, u'CmnCtrl1.exe'))
        self.dlg = self.app.Common_Controls_Sample
        self.ctrl = self.dlg.TreeView.find()

    def tearDown(self):
        if False:
            print('Hello World!')
        'Close the application after tests'
        self.app.kill()

    def testGuardSignatureCorruption(self):
        if False:
            i = 10
            return i + 15
        mem = RemoteMemoryBlock(self.ctrl, 16)
        buf = ctypes.create_string_buffer(24)
        self.assertRaises(Exception, mem.Write, buf)
        mem.size = 24
        self.assertRaises(Exception, mem.Write, buf)
if __name__ == '__main__':
    unittest.main()