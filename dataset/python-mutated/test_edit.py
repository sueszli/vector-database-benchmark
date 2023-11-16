"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import unittest
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QKeySequence
from AnyQt.QtTest import QTest
from Orange.widgets.data.utils.pythoneditor.tests import base
from Orange.widgets.data.utils.pythoneditor.tests.base import EditorTest

class Test(EditorTest):

    def test_overwrite_edit(self):
        if False:
            while True:
                i = 10
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClicks(self.qpart, 'stu')
        self.assertEqual(self.qpart.text, 'stuabcd')
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClicks(self.qpart, 'xy')
        self.assertEqual(self.qpart.text, 'stuxycd')
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClicks(self.qpart, 'z')
        self.assertEqual(self.qpart.text, 'stuxyzcd')

    def test_overwrite_backspace(self):
        if False:
            print('Hello World!')
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        for _ in range(3):
            QTest.keyClick(self.qpart, Qt.Key_Right)
        for _ in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Backspace)
        self.assertEqual(self.qpart.text, 'a  d')

    def test_overwrite_undo(self):
        if False:
            i = 10
            return i + 15
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClick(self.qpart, Qt.Key_Right)
        QTest.keyClick(self.qpart, Qt.Key_X)
        QTest.keyClick(self.qpart, Qt.Key_X)
        self.assertEqual(self.qpart.text, 'axxd')
        self.qpart.document().undo()
        self.qpart.document().undo()
        self.assertEqual(self.qpart.text, 'abcd')

    def test_home1(self):
        if False:
            print('Hello World!')
        ' Test the operation of the home key. '
        self.qpart.show()
        self.qpart.text = '  xx'
        self.qpart.cursorPosition = (100, 100)
        self.assertEqual(self.qpart.cursorPosition, (0, 4))

    def column(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the column at which the cursor is located.'
        return self.qpart.cursorPosition[1]

    def test_home2(self):
        if False:
            while True:
                i = 10
        ' Test the operation of the home key. '
        self.qpart.show()
        self.qpart.text = '\n\n    ' + 'x' * 10000
        self.qpart.cursorPosition = (100, 100)
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        if self.column() != 4:
            base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        self.assertEqual(self.column(), 4)
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        self.assertEqual(self.column(), 0)
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        self.assertEqual(self.column(), 4)
if __name__ == '__main__':
    unittest.main()