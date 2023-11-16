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
from AnyQt.QtTest import QTest
from Orange.widgets.data.utils.pythoneditor.tests.base import EditorTest

class Test(EditorTest):

    def test_1(self):
        if False:
            print('Hello World!')
        self.qpart.indentUseTabs = True
        self.qpart.text = 'ab\ncd'
        QTest.keyClick(self.qpart, Qt.Key_Down)
        QTest.keyClick(self.qpart, Qt.Key_Tab)
        self.assertEqual(self.qpart.text, 'ab\n\tcd')
        self.qpart.indentUseTabs = False
        QTest.keyClick(self.qpart, Qt.Key_Backspace)
        QTest.keyClick(self.qpart, Qt.Key_Tab)
        self.assertEqual(self.qpart.text, 'ab\n    cd')

    def test_2(self):
        if False:
            return 10
        self.qpart.indentUseTabs = True
        self.qpart.text = 'ab\n\t\tcd'
        self.qpart.cursorPosition = (1, 2)
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab\n\tcd')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab\ncd')

    def test_3(self):
        if False:
            for i in range(10):
                print('nop')
        self.qpart.indentUseTabs = False
        self.qpart.text = 'ab\n      cd'
        self.qpart.cursorPosition = (1, 6)
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab\n  cd')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab\ncd')

    def test_4(self):
        if False:
            i = 10
            return i + 15
        self.qpart.indentUseTabs = False
        self.qpart.text = '  ab\n  cd'
        self.qpart.selectedPosition = ((0, 2), (1, 3))
        QTest.keyClick(self.qpart, Qt.Key_Tab)
        self.assertEqual(self.qpart.text, '      ab\n      cd')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, '  ab\n  cd')

    def test_4b(self):
        if False:
            print('Hello World!')
        self.qpart.indentUseTabs = True
        self.qpart.text = 'ab\ncd\nef'
        self.qpart.position = (0, 0)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Tab)
        self.assertEqual(self.qpart.text, '\tab\ncd\nef')

    @unittest.skip
    def test_5(self):
        if False:
            i = 10
            return i + 15
        self.qpart.indentUseTabs = False
        self.qpart.text = '  ab\n  cd'
        self.qpart.selectedPosition = ((0, 2), (1, 3))
        QTest.keyClick(self.qpart, Qt.Key_Space, Qt.ShiftModifier | Qt.ControlModifier)
        self.assertEqual(self.qpart.text, '   ab\n   cd')
        QTest.keyClick(self.qpart, Qt.Key_Backspace, Qt.ShiftModifier | Qt.ControlModifier)
        self.assertEqual(self.qpart.text, '  ab\n  cd')

    def test_6(self):
        if False:
            print('Hello World!')
        self.qpart.indentUseTabs = False
        self.qpart.text = '    \t  \tab'
        self.qpart.cursorPosition = (0, 8)
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, '    \t  ab')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, '    \tab')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, '    ab')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab')
        self.qpart.decreaseIndentAction.trigger()
        self.assertEqual(self.qpart.text, 'ab')

    def test_7(self):
        if False:
            for i in range(10):
                print('nop')
        'Smartly indent python'
        QTest.keyClicks(self.qpart, 'def main():')
        QTest.keyClick(self.qpart, Qt.Key_Enter)
        self.assertEqual(self.qpart.cursorPosition, (1, 4))
        QTest.keyClicks(self.qpart, 'return 7')
        QTest.keyClick(self.qpart, Qt.Key_Enter)
        self.assertEqual(self.qpart.cursorPosition, (2, 0))
if __name__ == '__main__':
    unittest.main()