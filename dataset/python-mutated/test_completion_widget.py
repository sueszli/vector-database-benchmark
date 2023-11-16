import os
import tempfile
import shutil
import unittest
import pytest
from qtpy import QtCore, QtWidgets
from qtpy.QtTest import QTest
from qtconsole.console_widget import ConsoleWidget
from qtconsole.completion_widget import CompletionWidget
from . import no_display

class TemporaryDirectory(object):
    """
    Context manager for tempfile.mkdtemp().
    This class is available in python +v3.2.
    See: https://gist.github.com/cpelley/10e2eeaf60dacc7956bb
    """

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.dir_name = tempfile.mkdtemp()
        return self.dir_name

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.dir_name)
TemporaryDirectory = getattr(tempfile, 'TemporaryDirectory', TemporaryDirectory)

@pytest.mark.skipif(no_display, reason="Doesn't work without a display")
class TestCompletionWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        ' Create the application for the test case.\n        '
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        cls._app.setQuitOnLastWindowClosed(False)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ' Exit the application.\n        '
        QtWidgets.QApplication.quit()

    def setUp(self):
        if False:
            while True:
                i = 10
        ' Create the main widgets (ConsoleWidget)\n        '
        self.console = ConsoleWidget()
        self.text_edit = self.console._control

    def test_droplist_completer_shows(self):
        if False:
            i = 10
            return i + 15
        w = CompletionWidget(self.console)
        w.show_items(self.text_edit.textCursor(), ['item1', 'item2', 'item3'])
        self.assertTrue(w.isVisible())

    def test_droplist_completer_keyboard(self):
        if False:
            print('Hello World!')
        w = CompletionWidget(self.console)
        w.show_items(self.text_edit.textCursor(), ['item1', 'item2', 'item3'])
        QTest.keyClick(w, QtCore.Qt.Key_PageDown)
        QTest.keyClick(w, QtCore.Qt.Key_Enter)
        self.assertEqual(self.text_edit.toPlainText(), 'item3')

    def test_droplist_completer_mousepick(self):
        if False:
            while True:
                i = 10
        leftButton = QtCore.Qt.LeftButton
        w = CompletionWidget(self.console)
        w.show_items(self.text_edit.textCursor(), ['item1', 'item2', 'item3'])
        QTest.mouseClick(w.viewport(), leftButton, pos=QtCore.QPoint(19, 8))
        QTest.mouseRelease(w.viewport(), leftButton, pos=QtCore.QPoint(19, 8))
        QTest.mouseDClick(w.viewport(), leftButton, pos=QtCore.QPoint(19, 8))
        self.assertEqual(self.text_edit.toPlainText(), 'item1')
        self.assertFalse(w.isVisible())

    def test_common_path_complete(self):
        if False:
            while True:
                i = 10
        with TemporaryDirectory() as tmpdir:
            items = [os.path.join(tmpdir, 'common/common1/item1'), os.path.join(tmpdir, 'common/common1/item2'), os.path.join(tmpdir, 'common/common1/item3')]
            for item in items:
                os.makedirs(item)
            w = CompletionWidget(self.console)
            w.show_items(self.text_edit.textCursor(), items)
            self.assertEqual(w.currentItem().text(), '/item1')
            QTest.keyClick(w, QtCore.Qt.Key_Down)
            self.assertEqual(w.currentItem().text(), '/item2')
            QTest.keyClick(w, QtCore.Qt.Key_Down)
            self.assertEqual(w.currentItem().text(), '/item3')