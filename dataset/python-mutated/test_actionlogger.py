"""Tests for actionlogger.py"""
import unittest
import os
import sys
import logging
import mock
from six.moves import reload_module
sys.path.append('.')
from pywinauto import actionlogger
from pywinauto.windows.application import Application
from pywinauto.sysinfo import is_x64_Python
from pywinauto.sysinfo import is_x64_OS
from pywinauto.timings import Timings

def _notepad_exe():
    if False:
        while True:
            i = 10
    if is_x64_Python() or not is_x64_OS():
        return 'C:\\Windows\\System32\\notepad.exe'
    else:
        return 'C:\\Windows\\SysWOW64\\notepad.exe'

class ActionLoggerOnStadardLoggerTestCases(unittest.TestCase):
    """Unit tests for the actionlogger based on _StandardLogger"""

    def setUp(self):
        if False:
            return 10
        'Set some data and ensure the application is in the state we want'
        Timings.fast()
        actionlogger.enable()
        self.app = Application().start(_notepad_exe())
        self.logger = logging.getLogger('pywinauto')
        self.out = self.logger.handlers[0].stream
        self.logger.handlers[0].stream = open('test_logging.txt', 'w')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Close the application after tests'
        self.logger.handlers[0].stream.close()
        self.logger.handlers[0].stream = self.out
        self.app.kill()

    def __lineCount(self):
        if False:
            for i in range(10):
                print('nop')
        'hack to get line count from current logger stream'
        self.logger = logging.getLogger('pywinauto')
        self.logger.handlers[0].stream.flush()
        os.fsync(self.logger.handlers[0].stream.fileno())
        with open(self.logger.handlers[0].stream.name, 'r') as f:
            return len(f.readlines())

    def testEnableDisable(self):
        if False:
            while True:
                i = 10
        actionlogger.enable()
        prev_line_count = self.__lineCount()
        self.app.UntitledNotepad.type_keys('Test pywinauto logging', with_spaces=True)
        self.assertEqual(self.__lineCount(), prev_line_count + 1)
        actionlogger.disable()
        self.app.UntitledNotepad.menu_select('Help->About Notepad')
        self.assertEqual(self.__lineCount(), prev_line_count + 1)
        actionlogger.enable()
        self.app.window(name='About Notepad').OK.click()
        self.assertEqual(self.__lineCount(), prev_line_count + 2)

class ActionLoggerOnCustomLoggerTestCases(unittest.TestCase):
    """Unit tests for the actionlogger based on _CustomLogger"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set a mock logger package in modules'
        self.mock_logger = mock.MagicMock()
        self.modules = {'logger': self.mock_logger}
        self.module_patcher = mock.patch.dict('sys.modules', self.modules)
        self.module_patcher.start()
        self.logger_patcher = None

    def tearDown(self):
        if False:
            print('Hello World!')
        'Clean ups'
        if self.logger_patcher:
            self.logger_patcher.stop()
        self.module_patcher.stop()
        reload_module(actionlogger)

    def test_import_clash(self):
        if False:
            i = 10
            return i + 15
        'Test a custom logger import clash: issue #315'
        self.module_patcher.stop()
        self.mock_logger.Logger.sectionStart = None
        self.module_patcher = mock.patch.dict('sys.modules', self.modules)
        self.module_patcher.start()
        reload_module(actionlogger)
        self.assertEqual(False, actionlogger._found_logger)
        active_logger = actionlogger.ActionLogger()
        self.assertEqual(actionlogger._StandardLogger, type(active_logger))

    def test_import_custom_logger(self):
        if False:
            return 10
        'Test if custom logger class can be imported'
        reload_module(actionlogger)
        self.assertEqual(True, actionlogger._found_logger)
        self.mock_logger.Logger.assert_not_called()
        active_logger = actionlogger.ActionLogger()
        self.assertEqual(actionlogger._CustomLogger, type(active_logger))

    def test_logger_disable_and_reset(self):
        if False:
            return 10
        'Test if the logger can be disabled and level reset'
        reload_module(actionlogger)
        self.logger_patcher = mock.patch('pywinauto.actionlogger.ActionLogger', spec=True)
        mockLogger = self.logger_patcher.start()
        actionlogger.disable()
        self.assertTrue(mockLogger.disable.called)
        actionlogger.reset_level()
        self.assertTrue(mockLogger.reset_level.called)

    def test_logger_enable_mapped_to_reset_level(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if the logger enable is mapped to reset_level'
        reload_module(actionlogger)
        self.logger_patcher = mock.patch('pywinauto.actionlogger.ActionLogger', spec=True)
        mockLogger = self.logger_patcher.start()
        actionlogger.enable()
        self.assertTrue(mockLogger.reset_level.called)
if __name__ == '__main__':
    unittest.main()