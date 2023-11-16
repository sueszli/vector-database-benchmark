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
from Orange.widgets.data.utils.pythoneditor.vim import _globalClipboard

class _Test(EditorTest):
    """Base class for tests
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.qpart.lines = ['The quick brown fox', 'jumps over the', 'lazy dog', 'back']
        self.qpart.vimModeIndicationChanged.connect(self._onVimModeChanged)
        self.qpart.vimModeEnabled = True
        self.vimMode = 'normal'

    def tearDown(self):
        if False:
            print('Hello World!')
        self.qpart.hide()
        super().tearDown()

    def _onVimModeChanged(self, _, mode):
        if False:
            i = 10
            return i + 15
        self.vimMode = mode

    def click(self, keys):
        if False:
            print('Hello World!')
        if isinstance(keys, str):
            for key in keys:
                if key.isupper() or key in '$%^<>':
                    QTest.keyClick(self.qpart, key, Qt.ShiftModifier)
                else:
                    QTest.keyClicks(self.qpart, key)
        else:
            QTest.keyClick(self.qpart, keys)

class Modes(_Test):

    def test_01(self):
        if False:
            while True:
                i = 10
        'Switch modes insert/normal\n        '
        self.assertEqual(self.vimMode, 'normal')
        self.click('i123')
        self.assertEqual(self.vimMode, 'insert')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')
        self.click('i4')
        self.assertEqual(self.vimMode, 'insert')
        self.assertEqual(self.qpart.lines[0], '1234The quick brown fox')

    def test_02(self):
        if False:
            return 10
        'Append with A\n        '
        self.qpart.cursorPosition = (2, 0)
        self.click('A')
        self.assertEqual(self.vimMode, 'insert')
        self.click('XY')
        self.assertEqual(self.qpart.lines[2], 'lazy dogXY')

    def test_03(self):
        if False:
            for i in range(10):
                print('nop')
        'Append with a\n        '
        self.qpart.cursorPosition = (2, 0)
        self.click('a')
        self.assertEqual(self.vimMode, 'insert')
        self.click('XY')
        self.assertEqual(self.qpart.lines[2], 'lXYazy dog')

    def test_04(self):
        if False:
            i = 10
            return i + 15
        'Mode line shows composite command start\n        '
        self.assertEqual(self.vimMode, 'normal')
        self.click('d')
        self.assertEqual(self.vimMode, 'd')
        self.click('w')
        self.assertEqual(self.vimMode, 'normal')

    def test_05(self):
        if False:
            return 10
        ' Replace mode\n        '
        self.assertEqual(self.vimMode, 'normal')
        self.click('R')
        self.assertEqual(self.vimMode, 'replace')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0], 'asdfquick brown fox')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')
        self.click('R')
        self.assertEqual(self.vimMode, 'replace')
        self.click(Qt.Key_Insert)
        self.assertEqual(self.vimMode, 'insert')

    def test_05a(self):
        if False:
            for i in range(10):
                print('nop')
        ' Replace mode - at end of line\n        '
        self.click('$')
        self.click('R')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0], 'The quick brown foxasdf')

    def test_06(self):
        if False:
            while True:
                i = 10
        ' Visual mode\n        '
        self.assertEqual(self.vimMode, 'normal')
        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')
        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click('i')
        self.assertEqual(self.vimMode, 'insert')

    def test_07(self):
        if False:
            print('Hello World!')
        ' Switch to visual on selection\n        '
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.ShiftModifier)
        self.assertEqual(self.vimMode, 'visual')

    def test_08(self):
        if False:
            return 10
        ' From VISUAL to VISUAL LINES\n        '
        self.click('v')
        self.click('kkk')
        self.click('V')
        self.assertEqual(self.qpart.selectedText, 'The quick brown fox')
        self.assertEqual(self.vimMode, 'visual lines')

    def test_09(self):
        if False:
            i = 10
            return i + 15
        ' From VISUAL LINES to VISUAL\n        '
        self.click('V')
        self.click('v')
        self.assertEqual(self.qpart.selectedText, 'The quick brown fox')
        self.assertEqual(self.vimMode, 'visual')

    def test_10(self):
        if False:
            return 10
        ' Insert mode with I\n        '
        self.qpart.lines[1] = '   indented line'
        self.click('j8lI')
        self.click('Z')
        self.assertEqual(self.qpart.lines[1], '   Zindented line')

class Move(_Test):

    def test_01(self):
        if False:
            i = 10
            return i + 15
        'Move hjkl\n        '
        self.click('ll')
        self.assertEqual(self.qpart.cursorPosition, (0, 2))
        self.click('jjj')
        self.assertEqual(self.qpart.cursorPosition, (3, 2))
        self.click('h')
        self.assertEqual(self.qpart.cursorPosition, (3, 1))
        self.click('k')
        self.assertIn(self.qpart.cursorPosition, ((2, 1), (2, 2)))

    def test_02(self):
        if False:
            return 10
        'w\n        '
        self.qpart.lines[0] = 'word, comma, word'
        self.qpart.cursorPosition = (0, 0)
        for column in (4, 6, 11, 13, 17, 0):
            self.click('w')
            self.assertEqual(self.qpart.cursorPosition[1], column)
        self.assertEqual(self.qpart.cursorPosition, (1, 0))

    def test_03(self):
        if False:
            return 10
        'e\n        '
        self.qpart.lines[0] = '  word, comma, word'
        self.qpart.cursorPosition = (0, 0)
        for column in (6, 7, 13, 14, 19, 5):
            self.click('e')
            self.assertEqual(self.qpart.cursorPosition[1], column)
        self.assertEqual(self.qpart.cursorPosition, (1, 5))

    def test_04(self):
        if False:
            return 10
        '$\n        '
        self.click('$')
        self.assertEqual(self.qpart.cursorPosition, (0, 19))
        self.click('$')
        self.assertEqual(self.qpart.cursorPosition, (0, 19))

    def test_05(self):
        if False:
            i = 10
            return i + 15
        '0\n        '
        self.qpart.cursorPosition = (0, 10)
        self.click('0')
        self.assertEqual(self.qpart.cursorPosition, (0, 0))

    def test_06(self):
        if False:
            for i in range(10):
                print('nop')
        'G\n        '
        self.qpart.cursorPosition = (0, 10)
        self.click('G')
        self.assertEqual(self.qpart.cursorPosition, (3, 0))

    def test_07(self):
        if False:
            print('Hello World!')
        'gg\n        '
        self.qpart.cursorPosition = (2, 10)
        self.click('gg')
        self.assertEqual(self.qpart.cursorPosition, (0, 0))

    def test_08(self):
        if False:
            for i in range(10):
                print('nop')
        ' b word back\n        '
        self.qpart.cursorPosition = (0, 19)
        self.click('b')
        self.assertEqual(self.qpart.cursorPosition, (0, 16))
        self.click('b')
        self.assertEqual(self.qpart.cursorPosition, (0, 10))

    def test_09(self):
        if False:
            for i in range(10):
                print('nop')
        ' % to jump to next braket\n        '
        self.qpart.lines[0] = '(asdf fdsa) xxx'
        self.qpart.cursorPosition = (0, 0)
        self.click('%')
        self.assertEqual(self.qpart.cursorPosition, (0, 10))

    def test_10(self):
        if False:
            for i in range(10):
                print('nop')
        ' ^ to jump to the first non-space char\n        '
        self.qpart.lines[0] = '    indented line'
        self.qpart.cursorPosition = (0, 14)
        self.click('^')
        self.assertEqual(self.qpart.cursorPosition, (0, 4))

    def test_11(self):
        if False:
            print('Hello World!')
        ' f to search forward\n        '
        self.click('fv')
        self.assertEqual(self.qpart.cursorPosition, (1, 7))

    def test_12(self):
        if False:
            for i in range(10):
                print('nop')
        ' F to search backward\n        '
        self.qpart.cursorPosition = (2, 0)
        self.click('Fv')
        self.assertEqual(self.qpart.cursorPosition, (1, 7))

    def test_13(self):
        if False:
            i = 10
            return i + 15
        ' t to search forward\n        '
        self.click('tv')
        self.assertEqual(self.qpart.cursorPosition, (1, 6))

    def test_14(self):
        if False:
            i = 10
            return i + 15
        ' T to search backward\n        '
        self.qpart.cursorPosition = (2, 0)
        self.click('Tv')
        self.assertEqual(self.qpart.cursorPosition, (1, 8))

    def test_15(self):
        if False:
            return 10
        ' f in a composite command\n        '
        self.click('dff')
        self.assertEqual(self.qpart.lines[0], 'ox')

    def test_16(self):
        if False:
            i = 10
            return i + 15
        ' E\n        '
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (0, 0)
        for pos in (5, 6, 8, 9):
            self.click('e')
            self.assertEqual(self.qpart.cursorPosition[1], pos)
        self.qpart.cursorPosition = (0, 0)
        for pos in (10, 22, 34, 45, 5):
            self.click('E')
            self.assertEqual(self.qpart.cursorPosition[1], pos)

    def test_17(self):
        if False:
            return 10
        ' W\n        '
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (0, 0)
        for pos in ((0, 12), (0, 24), (0, 35), (1, 0), (1, 6)):
            self.click('W')
            self.assertEqual(self.qpart.cursorPosition, pos)

    def test_18(self):
        if False:
            while True:
                i = 10
        ' B\n        '
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (1, 8)
        for pos in ((1, 6), (1, 0), (0, 35), (0, 24), (0, 12)):
            self.click('B')
            self.assertEqual(self.qpart.cursorPosition, pos)

    def test_19(self):
        if False:
            for i in range(10):
                print('nop')
        ' Enter, Return\n        '
        self.qpart.lines[1] = '   indented line'
        self.qpart.lines[2] = '     more indented line'
        self.click(Qt.Key_Enter)
        self.assertEqual(self.qpart.cursorPosition, (1, 3))
        self.click(Qt.Key_Return)
        self.assertEqual(self.qpart.cursorPosition, (2, 5))

class Del(_Test):

    def test_01a(self):
        if False:
            for i in range(10):
                print('nop')
        'Delete with x\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('xxxxx')
        self.assertEqual(self.qpart.lines[0], 'The  brown fox')
        self.assertEqual(_globalClipboard.value, 'k')

    def test_01b(self):
        if False:
            return 10
        'Delete with x. Use count\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('5x')
        self.assertEqual(self.qpart.lines[0], 'The  brown fox')
        self.assertEqual(_globalClipboard.value, 'quick')

    def test_02(self):
        if False:
            while True:
                i = 10
        'Composite delete with d. Left and right\n        '
        self.qpart.cursorPosition = (1, 1)
        self.click('dl')
        self.assertEqual(self.qpart.lines[1], 'jmps over the')
        self.click('dh')
        self.assertEqual(self.qpart.lines[1], 'mps over the')

    def test_03(self):
        if False:
            print('Hello World!')
        'Composite delete with d. Down\n        '
        self.qpart.cursorPosition = (0, 2)
        self.click('dj')
        self.assertEqual(self.qpart.lines[:], ['lazy dog', 'back'])
        self.assertEqual(self.qpart.cursorPosition[1], 0)
        self.qpart.cursorPosition = (1, 1)
        self.click('dj')
        self.assertEqual(self.qpart.lines[:], ['lazy dog', 'back'])
        self.click('k')
        self.click('dj')
        self.assertEqual(self.qpart.lines[:], [''])
        self.assertEqual(_globalClipboard.value, ['lazy dog', 'back'])

    def test_04(self):
        if False:
            while True:
                i = 10
        'Composite delete with d. Up\n        '
        self.qpart.cursorPosition = (0, 2)
        self.click('dk')
        self.assertEqual(len(self.qpart.lines), 4)
        self.qpart.cursorPosition = (2, 1)
        self.click('dk')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'back'])
        self.assertEqual(_globalClipboard.value, ['jumps over the', 'lazy dog'])
        self.assertEqual(self.qpart.cursorPosition[1], 0)

    def test_05(self):
        if False:
            while True:
                i = 10
        'Delete Count times\n        '
        self.click('3dw')
        self.assertEqual(self.qpart.lines[0], 'fox')
        self.assertEqual(_globalClipboard.value, 'The quick brown ')

    def test_06(self):
        if False:
            for i in range(10):
                print('nop')
        'Delete line\n        dd\n        '
        self.qpart.cursorPosition = (1, 0)
        self.click('dd')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'lazy dog', 'back'])

    def test_07(self):
        if False:
            return 10
        'Delete until end of file\n        G\n        '
        self.qpart.cursorPosition = (2, 0)
        self.click('dG')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the'])

    def test_08(self):
        if False:
            while True:
                i = 10
        'Delete until start of file\n        gg\n        '
        self.qpart.cursorPosition = (1, 0)
        self.click('dgg')
        self.assertEqual(self.qpart.lines[:], ['lazy dog', 'back'])

    def test_09(self):
        if False:
            return 10
        'Delete with X\n        '
        self.click('llX')
        self.assertEqual(self.qpart.lines[0], 'Te quick brown fox')

    def test_10(self):
        if False:
            i = 10
            return i + 15
        'Delete with D\n        '
        self.click('jll')
        self.click('2D')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'ju', 'back'])

class Edit(_Test):

    def test_01(self):
        if False:
            for i in range(10):
                print('nop')
        'Undo\n        '
        oldText = self.qpart.text
        self.click('ddu')
        modifiedText = self.qpart.text
        self.assertEqual(self.qpart.text, oldText)

    def test_02(self):
        if False:
            i = 10
            return i + 15
        'Change with C\n        '
        self.click('lllCpig')
        self.assertEqual(self.qpart.lines[0], 'Thepig')

    def test_03(self):
        if False:
            return 10
        ' Substitute with s\n        '
        self.click('j4sz')
        self.assertEqual(self.qpart.lines[1], 'zs over the')

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Replace char with r\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('rZ')
        self.assertEqual(self.qpart.lines[0], 'The Zuick brown fox')
        self.click('rW')
        self.assertEqual(self.qpart.lines[0], 'The Wuick brown fox')

    def test_05(self):
        if False:
            i = 10
            return i + 15
        'Change 2 words with c\n        '
        self.click('c2e')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0], 'asdf brown fox')

    def test_06(self):
        if False:
            while True:
                i = 10
        'Open new line with o\n        '
        self.qpart.lines = ['    indented line', '    next indented line']
        self.click('o')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[:], ['    indented line', '    asdf', '    next indented line'])

    def test_07(self):
        if False:
            while True:
                i = 10
        'Open new line with O\n\n        Check indentation\n        '
        self.qpart.lines = ['    indented line', '    next indented line']
        self.click('j')
        self.click('O')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[:], ['    indented line', '    asdf', '    next indented line'])

    def test_08(self):
        if False:
            print('Hello World!')
        ' Substitute with S\n        '
        self.qpart.lines = ['    indented line', '    next indented line']
        self.click('ljS')
        self.click('xyz')
        self.assertEqual(self.qpart.lines[:], ['    indented line', '    xyz'])

    def test_09(self):
        if False:
            for i in range(10):
                print('nop')
        ' % to jump to next braket\n        '
        self.qpart.lines[0] = '(asdf fdsa) xxx'
        self.qpart.cursorPosition = (0, 0)
        self.click('d%')
        self.assertEqual(self.qpart.lines[0], ' xxx')

    def test_10(self):
        if False:
            for i in range(10):
                print('nop')
        ' J join lines\n        '
        self.click('2J')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox jumps over the lazy dog', 'back'])

class Indent(_Test):

    def test_01(self):
        if False:
            while True:
                i = 10
        ' Increase indent with >j, decrease with <j\n        '
        self.click('>2j')
        self.assertEqual(self.qpart.lines[:], ['    The quick brown fox', '    jumps over the', '    lazy dog', 'back'])
        self.click('<j')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the', '    lazy dog', 'back'])

    def test_02(self):
        if False:
            print('Hello World!')
        ' Increase indent with >>, decrease with <<\n        '
        self.click('>>')
        self.click('>>')
        self.assertEqual(self.qpart.lines[0], '        The quick brown fox')
        self.click('<<')
        self.assertEqual(self.qpart.lines[0], '    The quick brown fox')

    def test_03(self):
        if False:
            return 10
        ' Autoindent with =j\n        '
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('=j')
        self.assertEqual(self.qpart.lines[:], ['    The quick brown fox', '    jumps over the', '    lazy dog', 'back'])

    def test_04(self):
        if False:
            return 10
        ' Autoindent with ==\n        '
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('==')
        self.assertEqual(self.qpart.lines[:], ['    The quick brown fox', '    jumps over the', 'lazy dog', 'back'])

    def test_11(self):
        if False:
            return 10
        ' Increase indent with >, decrease with < in visual mode\n        '
        self.click('v2>')
        self.assertEqual(self.qpart.lines[:2], ['        The quick brown fox', 'jumps over the'])
        self.click('v<')
        self.assertEqual(self.qpart.lines[:2], ['    The quick brown fox', 'jumps over the'])

    def test_12(self):
        if False:
            while True:
                i = 10
        ' Autoindent with = in visual mode\n        '
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('Vj=')
        self.assertEqual(self.qpart.lines[:], ['    The quick brown fox', '    jumps over the', '    lazy dog', 'back'])

class CopyPaste(_Test):

    def test_02(self):
        if False:
            print('Hello World!')
        'Paste text with p\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('5x')
        self.assertEqual(self.qpart.lines[0], 'The  brown fox')
        self.click('p')
        self.assertEqual(self.qpart.lines[0], 'The  quickbrown fox')

    def test_03(self):
        if False:
            return 10
        'Paste lines with p\n        '
        self.qpart.cursorPosition = (1, 2)
        self.click('2dd')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'back'])
        self.click('kkk')
        self.click('p')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the', 'lazy dog', 'back'])

    def test_04(self):
        if False:
            return 10
        'Paste lines with P\n        '
        self.qpart.cursorPosition = (1, 2)
        self.click('2dd')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'back'])
        self.click('P')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the', 'lazy dog', 'back'])

    def test_05(self):
        if False:
            i = 10
            return i + 15
        ' Yank line with yy\n        '
        self.click('y2y')
        self.click('jll')
        self.click('p')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the', 'The quick brown fox', 'jumps over the', 'lazy dog', 'back'])

    def test_06(self):
        if False:
            print('Hello World!')
        ' Yank until the end of line\n        '
        self.click('2wYo')
        self.click(Qt.Key_Escape)
        self.click('P')
        self.assertEqual(self.qpart.lines[1], 'brown fox')

    def test_08(self):
        if False:
            print('Hello World!')
        ' Composite yank with y, paste with P\n        '
        self.click('y2w')
        self.click('P')
        self.assertEqual(self.qpart.lines[0], 'The quick The quick brown fox')

class Visual(_Test):

    def test_01(self):
        if False:
            print('Hello World!')
        ' x\n        '
        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click('2w')
        self.assertEqual(self.qpart.selectedText, 'The quick ')
        self.click('x')
        self.assertEqual(self.qpart.lines[0], 'brown fox')
        self.assertEqual(self.vimMode, 'normal')

    def test_02(self):
        if False:
            return 10
        'Append with a\n        '
        self.click('vllA')
        self.click('asdf ')
        self.assertEqual(self.qpart.lines[0], 'The asdf quick brown fox')

    def test_03(self):
        if False:
            i = 10
            return i + 15
        'Replace with r\n        '
        self.qpart.cursorPosition = (0, 16)
        self.click('v8l')
        self.click('rz')
        self.assertEqual(self.qpart.lines[0:2], ['The quick brown zzz', 'zzzzz over the'])

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Replace selected lines with R\n        '
        self.click('vjl')
        self.click('R')
        self.click('Z')
        self.assertEqual(self.qpart.lines[:], ['Z', 'lazy dog', 'back'])

    def test_05(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset selection with u\n        '
        self.qpart.cursorPosition = (1, 3)
        self.click('vjl')
        self.click('u')
        self.assertEqual(self.qpart.selectedPosition, ((1, 3), (1, 3)))

    def test_06(self):
        if False:
            print('Hello World!')
        'Yank with y and paste with p\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('ve')
        self.click('y')
        self.click(Qt.Key_Escape)
        self.qpart.cursorPosition = (0, 16)
        self.click('ve')
        self.click('p')
        self.assertEqual(self.qpart.lines[0], 'The quick brown quick')

    def test_07(self):
        if False:
            i = 10
            return i + 15
        ' Replace word when pasting\n        '
        self.click('vey')
        self.click('ww')
        self.click('vep')
        self.assertEqual(self.qpart.lines[0], 'The quick The fox')

    def test_08(self):
        if False:
            print('Hello World!')
        'Change with c\n        '
        self.click('w')
        self.click('vec')
        self.click('slow')
        self.assertEqual(self.qpart.lines[0], 'The slow brown fox')

    def test_09(self):
        if False:
            print('Hello World!')
        ' Delete lines with X and D\n        '
        self.click('jvlX')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'lazy dog', 'back'])
        self.click('u')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the', 'lazy dog', 'back'])
        self.click('vjD')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'back'])

    def test_10(self):
        if False:
            print('Hello World!')
        ' Check if f works\n        '
        self.click('vfo')
        self.assertEqual(self.qpart.selectedText, 'The quick bro')

    def test_11(self):
        if False:
            print('Hello World!')
        ' J join lines\n        '
        self.click('jvjJ')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', 'jumps over the lazy dog', 'back'])

class VisualLines(_Test):

    def test_01(self):
        if False:
            while True:
                i = 10
        ' x Delete\n        '
        self.click('V')
        self.assertEqual(self.vimMode, 'visual lines')
        self.click('x')
        self.click('p')
        self.assertEqual(self.qpart.lines[:], ['jumps over the', 'The quick brown fox', 'lazy dog', 'back'])
        self.assertEqual(self.vimMode, 'normal')

    def test_02(self):
        if False:
            while True:
                i = 10
        ' Replace text when pasting\n        '
        self.click('Vy')
        self.click('j')
        self.click('Vp')
        self.assertEqual(self.qpart.lines[0:3], ['The quick brown fox', 'The quick brown fox', 'lazy dog'])

    def test_06(self):
        if False:
            return 10
        'Yank with y and paste with p\n        '
        self.qpart.cursorPosition = (0, 4)
        self.click('V')
        self.click('y')
        self.click(Qt.Key_Escape)
        self.qpart.cursorPosition = (0, 16)
        self.click('p')
        self.assertEqual(self.qpart.lines[0:3], ['The quick brown fox', 'The quick brown fox', 'jumps over the'])

    def test_07(self):
        if False:
            print('Hello World!')
        'Change with c\n        '
        self.click('Vc')
        self.click('slow')
        self.assertEqual(self.qpart.lines[0], 'slow')

class Repeat(_Test):

    def test_01(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat o\n        '
        self.click('o')
        self.click(Qt.Key_Escape)
        self.click('j2.')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', '', 'jumps over the', '', '', 'lazy dog', 'back'])

    def test_02(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat o. Use count from previous command\n        '
        self.click('2o')
        self.click(Qt.Key_Escape)
        self.click('j.')
        self.assertEqual(self.qpart.lines[:], ['The quick brown fox', '', '', 'jumps over the', '', '', 'lazy dog', 'back'])

    def test_03(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat O\n        '
        self.click('O')
        self.click(Qt.Key_Escape)
        self.click('2j2.')
        self.assertEqual(self.qpart.lines[:], ['', 'The quick brown fox', '', '', 'jumps over the', 'lazy dog', 'back'])

    def test_04(self):
        if False:
            return 10
        ' Repeat p\n        '
        self.click('ylp.')
        self.assertEqual(self.qpart.lines[0], 'TTThe quick brown fox')

    def test_05(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat p\n        '
        self.click('x...')
        self.assertEqual(self.qpart.lines[0], 'quick brown fox')

    def test_06(self):
        if False:
            return 10
        ' Repeat D\n        '
        self.click('Dj.')
        self.assertEqual(self.qpart.lines[:], ['', '', 'lazy dog', 'back'])

    def test_07(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat dw\n        '
        self.click('dw')
        self.click('j0.')
        self.assertEqual(self.qpart.lines[:], ['quick brown fox', 'over the', 'lazy dog', 'back'])

    def test_08(self):
        if False:
            while True:
                i = 10
        ' Repeat Visual x\n        '
        self.qpart.lines.append('one more')
        self.click('Vjx')
        self.click('.')
        self.assertEqual(self.qpart.lines[:], ['one more'])

    def test_09(self):
        if False:
            for i in range(10):
                print('nop')
        ' Repeat visual X\n        '
        self.qpart.lines.append('one more')
        self.click('vjX')
        self.click('.')
        self.assertEqual(self.qpart.lines[:], ['one more'])

    def test_10(self):
        if False:
            print('Hello World!')
        ' Repeat Visual >\n        '
        self.qpart.lines.append('one more')
        self.click('Vj>')
        self.click('3j')
        self.click('.')
        self.assertEqual(self.qpart.lines[:], ['    The quick brown fox', '    jumps over the', 'lazy dog', '    back', '    one more'])
if __name__ == '__main__':
    unittest.main()