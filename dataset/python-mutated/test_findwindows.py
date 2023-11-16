"""Tests for findwindows.py"""
from __future__ import print_function
import unittest
import sys, os
sys.path.append('.')
from pywinauto.windows.application import Application
from pywinauto.sysinfo import is_x64_Python
from pywinauto.findwindows import find_window, find_windows
from pywinauto.findwindows import WindowNotFoundError
from pywinauto.findwindows import WindowAmbiguousError
from pywinauto.timings import Timings
mfc_samples_folder = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples')
if is_x64_Python():
    mfc_samples_folder = os.path.join(mfc_samples_folder, 'x64')
mfc_app_1 = os.path.join(mfc_samples_folder, u'CmnCtrl2.exe')

class FindWindowsTestCases(unittest.TestCase):
    """Unit tests for findwindows.py module"""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application(backend='win32')
        self.app = self.app.start(mfc_app_1)
        self.dlg = self.app.CommonControlsSample

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the application after tests'
        self.app.kill()

    def test_find_window(self):
        if False:
            print('Hello World!')
        'Test if function find_window() works as expected including raising the exceptions'
        ctrl = self.dlg.OK.find()
        handle = find_window(pid=self.app.process, best_match='OK', top_level_only=False)
        self.assertEqual(handle, ctrl.handle)
        self.assertRaises(WindowNotFoundError, find_window, pid=self.app.process, class_name='OK')
        self.assertRaises(WindowAmbiguousError, find_window, pid=self.app.process, class_name='Button', top_level_only=False)

    def test_find_windows(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if function find_windows() works as expected including raising the exceptions'
        ctrl_hwnds = [elem.handle for elem in self.dlg.children() if elem.class_name() == 'Edit']
        handles = find_windows(pid=self.app.process, class_name='Edit', top_level_only=False)
        self.assertEqual(set(handles), set(ctrl_hwnds))
        self.assertRaises(WindowNotFoundError, find_windows, pid=self.app.process, class_name='FakeClassName', found_index=1)
if __name__ == '__main__':
    unittest.main()