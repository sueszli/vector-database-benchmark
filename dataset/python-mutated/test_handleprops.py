"""Tests for handleprops.py"""
import unittest
import six
import os
import sys
import warnings
sys.path.append('.')
from pywinauto.windows import win32structures
from pywinauto.handleprops import children, classname, clientrect, contexthelpid, controlid, dumpwindow, exstyle, font, has_exstyle, has_style, is64bitprocess, is_toplevel_window, isenabled, isunicode, isvisible, iswindow, parent, processid, rectangle, style, text, userdata, is64bitbinary
from pywinauto.windows.application import Application
from pywinauto.sysinfo import is_x64_OS
from pywinauto.sysinfo import is_x64_Python
from pywinauto.timings import Timings

class HandlepropsTestCases(unittest.TestCase):
    """Unit tests for the handleprops module"""

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application().start('notepad')
        self.dlghandle = self.app.UntitledNotepad.handle
        self.edit_handle = self.app.UntitledNotepad.Edit.handle

    def tearDown(self):
        if False:
            return 10
        'Close the application after tests'
        self.app.kill()

    def test_text(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the text method returns correct result'
        self.assertEqual('Untitled - Notepad', text(self.dlghandle))
        self.assertEqual('', text(self.edit_handle))
        self.assertEqual('', text(sys.maxsize))
        self.assertEqual('', text(None))

    def test_classname(self):
        if False:
            i = 10
            return i + 15
        'Make sure the classname method returns correct result'
        self.assertEqual('Notepad', classname(self.dlghandle))
        self.assertEqual('Edit', classname(self.edit_handle))
        self.assertEqual('', classname(sys.maxsize))
        self.assertEqual(None, classname(None))

    def test_parent(self):
        if False:
            print('Hello World!')
        'Make sure the parent method returns correct result'
        self.assertEqual(None, parent(self.dlghandle))
        self.assertEqual(self.dlghandle, parent(self.edit_handle))
        self.assertEqual(None, parent(sys.maxsize))
        self.assertEqual(None, parent(None))

    def test_style(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the style method returns correct result'
        self.assertEqual(349110272, style(self.dlghandle))
        self.assertTrue((1344274692, 1345323268).__contains__, style(self.edit_handle))
        self.assertEqual(0, style(sys.maxsize))
        self.assertEqual(0, style(None))

    def test_exstyle(self):
        if False:
            i = 10
            return i + 15
        'Make sure the exstyle method returns correct result'
        self.assertEqual(272, exstyle(self.dlghandle))
        self.assertEqual(512, exstyle(self.edit_handle))
        self.assertEqual(0, exstyle(sys.maxsize))
        self.assertEqual(0, exstyle(None))

    def test_controlid(self):
        if False:
            return 10
        'Make sure the controlid method returns correct result'
        self.assertEqual(15, controlid(self.edit_handle))
        self.assertEqual(0, controlid(sys.maxsize))
        self.assertEqual(0, controlid(None))

    def test_userdata(self):
        if False:
            i = 10
            return i + 15
        'Make sure the userdata method returns correct result'
        self.assertEqual(0, userdata(self.dlghandle))
        self.assertEqual(0, userdata(self.edit_handle))
        self.assertEqual(0, userdata(sys.maxsize))
        self.assertEqual(0, userdata(None))

    def test_contexthelpid(self):
        if False:
            while True:
                i = 10
        'Make sure the contexthelpid method returns correct result'
        self.assertEqual(0, contexthelpid(self.dlghandle))
        self.assertEqual(0, contexthelpid(self.edit_handle))
        self.assertEqual(0, contexthelpid(sys.maxsize))
        self.assertEqual(0, contexthelpid(None))

    def test_iswindow(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the iswindow method returns correct result'
        self.assertEqual(True, iswindow(self.dlghandle))
        self.assertEqual(True, iswindow(self.edit_handle))
        self.assertEqual(False, iswindow(1))
        self.assertEqual(False, iswindow(sys.maxsize))
        self.assertEqual(False, iswindow(None))

    def test_isvisible(self):
        if False:
            while True:
                i = 10
        'Make sure the isvisible method returns correct result'
        self.assertEqual(True, isvisible(self.dlghandle))
        self.assertEqual(True, isvisible(self.edit_handle))
        self.assertEqual(False, isvisible(sys.maxsize))
        self.assertEqual(False, isvisible(None))

    def test_isunicode(self):
        if False:
            while True:
                i = 10
        'Make sure the isunicode method returns correct result'
        self.assertEqual(True, isunicode(self.dlghandle))
        self.assertEqual(True, isunicode(self.edit_handle))
        self.assertEqual(False, isunicode(sys.maxsize))
        self.assertEqual(False, isunicode(None))

    def test_isenabled(self):
        if False:
            i = 10
            return i + 15
        'Make sure the isenabled method returns correct result'
        self.assertEqual(False, isenabled(sys.maxsize))
        self.assertEqual(False, isenabled(None))
        self.assertEqual(True, isenabled(self.dlghandle))
        self.assertEqual(True, isenabled(self.edit_handle))
        self.app.UntitledNotepad.menu_select('Help->About Notepad')
        self.app.AboutNotepad.wait('ready')
        self.assertEqual(False, isenabled(self.dlghandle))
        self.app.AboutNotepad.OK.close_click()
        self.app.UntitledNotepad.menu_select('Edit->Replace')
        self.assertEqual(False, isenabled(self.app.Replace.by(name_re='Replace.*', class_name='Button', enabled=None).handle))
        self.app.Replace.Cancel.Click()

    def test_clientrect(self):
        if False:
            print('Hello World!')
        'Make sure clientrect() function works'
        self.assertEqual(0, clientrect(self.dlghandle).left)
        self.assertEqual(0, clientrect(self.edit_handle).left)
        self.assertEqual(0, clientrect(self.dlghandle).top)
        self.assertEqual(0, clientrect(self.edit_handle).top)
        self.assertEqual(True, rectangle(self.dlghandle).right > clientrect(self.dlghandle).right)
        self.assertEqual(True, rectangle(self.edit_handle).right > clientrect(self.edit_handle).right)
        self.assertEqual(True, rectangle(self.dlghandle).bottom > clientrect(self.dlghandle).bottom)
        self.assertEqual(True, rectangle(self.edit_handle).bottom > clientrect(self.edit_handle).bottom)

    def test_rectangle(self):
        if False:
            i = 10
            return i + 15
        'Make sure rectangle() function works'
        dlgrect = rectangle(self.dlghandle)
        self.assertEqual(True, dlgrect.left < dlgrect.right)
        self.assertEqual(True, dlgrect.top < dlgrect.bottom)
        editrect = rectangle(self.edit_handle)
        self.assertEqual(True, editrect.left < editrect.right)
        self.assertEqual(True, editrect.top < editrect.bottom)

    def test_font(self):
        if False:
            print('Hello World!')
        'Make sure font() function works'
        dlgfont = font(self.dlghandle)
        self.assertEqual(True, isinstance(dlgfont.lfFaceName, six.string_types))
        editfont = font(self.edit_handle)
        self.assertEqual(True, isinstance(editfont.lfFaceName, six.string_types))
        expected = win32structures.LOGFONTW()
        self.assertEqual(type(expected), type(font(sys.maxsize)))
        self.assertEqual(type(expected), type(font(None)))

    def test_processid(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure processid() function works'
        self.assertEqual(self.app.process, processid(self.dlghandle))
        self.assertEqual(self.app.process, processid(self.edit_handle))
        self.assertEqual(0, processid(sys.maxsize))
        self.assertEqual(0, processid(None))

    def test_children(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the children method returns correct result'
        self.assertEqual(2, len(children(self.dlghandle)))
        self.assertEqual([], children(self.edit_handle))

    def test_has_style(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the has_style method returns correct result'
        self.assertEqual(True, has_style(self.dlghandle, 983040))
        self.assertEqual(True, has_style(self.edit_handle, 4))
        self.assertEqual(False, has_style(self.dlghandle, 4))
        self.assertEqual(False, has_style(self.edit_handle, 1))

    def test_has_exstyle(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the has_exstyle method returns correct result'
        self.assertEqual(True, has_exstyle(self.dlghandle, 16))
        self.assertEqual(True, has_exstyle(self.edit_handle, 512))
        self.assertEqual(False, has_exstyle(self.dlghandle, 4))
        self.assertEqual(False, has_exstyle(self.edit_handle, 16))

    def test_is_toplevel_window(self):
        if False:
            return 10
        'Make sure is_toplevel_window() function works'
        self.assertEqual(True, is_toplevel_window(self.dlghandle))
        self.assertEqual(False, is_toplevel_window(self.edit_handle))
        self.app.UntitledNotepad.menu_select('Edit->Replace')
        self.assertEqual(True, is_toplevel_window(self.app.Replace.handle))
        self.assertEqual(False, is_toplevel_window(self.app.Replace.Cancel.handle))
        self.app.Replace.Cancel.click()

    def test_is64bitprocess(self):
        if False:
            i = 10
            return i + 15
        'Make sure a 64-bit process detection returns correct results'
        if is_x64_OS():
            expected_is64bit = False
            if is_x64_Python():
                exe32bit = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples\\RowList.exe')
                app = Application().start(exe32bit, timeout=20)
                pid = app.RowListSampleApplication.process_id()
                res_is64bit = is64bitprocess(pid)
                try:
                    self.assertEqual(expected_is64bit, res_is64bit)
                finally:
                    app.kill()
                expected_is64bit = True
        else:
            expected_is64bit = False
        res_is64bit = is64bitprocess(self.app.UntitledNotepad.process_id())
        self.assertEqual(expected_is64bit, res_is64bit)

    def test_is64bitbinary(self):
        if False:
            i = 10
            return i + 15
        exe32bit = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples\\RowList.exe')
        dll32bit = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples\\mfc100u.dll')
        self.assertEqual(is64bitbinary(exe32bit), False)
        self.assertEqual(is64bitbinary(dll32bit), None)
        warnings.filterwarnings('always', category=RuntimeWarning, append=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            is64bitbinary(dll32bit)
            assert len(w) >= 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert 'Cannot get binary type for file' in str(w[-1].message)

    def test_dumpwindow(self):
        if False:
            print('Hello World!')
        'Make sure dumpwindow() function works'
        dlgdump = dumpwindow(self.dlghandle)
        for (key, item) in dlgdump.items():
            self.assertEqual(item, globals()[key](self.dlghandle))
        editdump = dumpwindow(self.edit_handle)
        for (key, item) in editdump.items():
            self.assertEqual(item, globals()[key](self.edit_handle))
if __name__ == '__main__':
    unittest.main()