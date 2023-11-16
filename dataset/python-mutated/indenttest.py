"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from AnyQt.QtCore import Qt
from AnyQt.QtTest import QTest
from Orange.widgets.data.utils.pythoneditor.tests.base import EditorTest

class IndentTest(EditorTest):
    """Base class for tests
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        if hasattr(self, 'INDENT_WIDTH'):
            self.qpart.indentWidth = self.INDENT_WIDTH

    def setOrigin(self, text):
        if False:
            return 10
        self.qpart.text = '\n'.join(text)

    def verifyExpected(self, text):
        if False:
            i = 10
            return i + 15
        lines = self.qpart.text.split('\n')
        self.assertEqual(text, lines)

    def setCursorPosition(self, line, col):
        if False:
            for i in range(10):
                print('nop')
        self.qpart.cursorPosition = (line, col)

    def enter(self):
        if False:
            i = 10
            return i + 15
        QTest.keyClick(self.qpart, Qt.Key_Enter)

    def tab(self):
        if False:
            for i in range(10):
                print('nop')
        QTest.keyClick(self.qpart, Qt.Key_Tab)

    def type(self, text):
        if False:
            return 10
        QTest.keyClicks(self.qpart, text)

    def writeCursorPosition(self):
        if False:
            i = 10
            return i + 15
        (line, col) = self.qpart.cursorPosition
        text = '(%d,%d)' % (line, col)
        self.type(text)

    def writeln(self):
        if False:
            print('Hello World!')
        self.qpart.textCursor().insertText('\n')

    def alignLine(self, index):
        if False:
            return 10
        self.qpart._indenter.autoIndentBlock(self.qpart.document().findBlockByNumber(index), '')

    def alignAll(self):
        if False:
            while True:
                i = 10
        QTest.keyClick(self.qpart, Qt.Key_A, Qt.ControlModifier)
        self.qpart.autoIndentLineAction.trigger()