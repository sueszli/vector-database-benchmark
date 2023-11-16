"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from AnyQt.QtCore import Qt, QMimeData
from AnyQt.QtWidgets import QApplication, QTextEdit
from AnyQt.QtGui import QKeyEvent, QKeySequence, QPalette, QTextCursor

class RectangularSelection:
    """This class does not replresent any object, but is part of Qutepart
    It just groups together Qutepart rectangular selection methods and fields
    """
    MIME_TYPE = 'text/rectangular-selection'
    MOUSE_MODIFIERS = (Qt.AltModifier | Qt.ControlModifier, Qt.AltModifier | Qt.ShiftModifier, Qt.AltModifier)
    _MAX_SIZE = 256

    def __init__(self, qpart):
        if False:
            for i in range(10):
                print('nop')
        self._qpart = qpart
        self._start = None
        qpart.cursorPositionChanged.connect(self._reset)
        qpart.textChanged.connect(self._reset)
        qpart.selectionChanged.connect(self._reset)

    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Cursor moved while Alt is not pressed, or text modified.\n        Reset rectangular selection'
        if self._start is not None:
            self._start = None
            self._qpart._updateExtraSelections()

    def isDeleteKeyEvent(self, keyEvent):
        if False:
            while True:
                i = 10
        'Check if key event should be handled as Delete command'
        return self._start is not None and (keyEvent.matches(QKeySequence.Delete) or (keyEvent.key() == Qt.Key_Backspace and keyEvent.modifiers() == Qt.NoModifier))

    def delete(self):
        if False:
            print('Hello World!')
        'Del or Backspace pressed. Delete selection'
        with self._qpart:
            for cursor in self.cursors():
                if cursor.hasSelection():
                    cursor.deleteChar()

    @staticmethod
    def isExpandKeyEvent(keyEvent):
        if False:
            for i in range(10):
                print('nop')
        'Check if key event should expand rectangular selection'
        return keyEvent.modifiers() & Qt.ShiftModifier and keyEvent.modifiers() & Qt.AltModifier and (keyEvent.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Down, Qt.Key_Up, Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Home, Qt.Key_End))

    def onExpandKeyEvent(self, keyEvent):
        if False:
            while True:
                i = 10
        'One of expand selection key events'
        if self._start is None:
            currentBlockText = self._qpart.textCursor().block().text()
            line = self._qpart.cursorPosition[0]
            visibleColumn = self._realToVisibleColumn(currentBlockText, self._qpart.cursorPosition[1])
            self._start = (line, visibleColumn)
        modifiersWithoutAltShift = keyEvent.modifiers() & ~(Qt.AltModifier | Qt.ShiftModifier)
        newEvent = QKeyEvent(QKeyEvent.Type(keyEvent.type()), keyEvent.key(), modifiersWithoutAltShift, keyEvent.text(), keyEvent.isAutoRepeat(), keyEvent.count())
        self._qpart.cursorPositionChanged.disconnect(self._reset)
        self._qpart.selectionChanged.disconnect(self._reset)
        super(self._qpart.__class__, self._qpart).keyPressEvent(newEvent)
        self._qpart.cursorPositionChanged.connect(self._reset)
        self._qpart.selectionChanged.connect(self._reset)

    def _visibleCharPositionGenerator(self, text):
        if False:
            print('Hello World!')
        currentPos = 0
        yield currentPos
        for char in text:
            if char == '\t':
                currentPos += self._qpart.indentWidth
                currentPos = currentPos // self._qpart.indentWidth * self._qpart.indentWidth
            else:
                currentPos += 1
            yield currentPos

    def _realToVisibleColumn(self, text, realColumn):
        if False:
            while True:
                i = 10
        'If \t is used, real position of symbol in block and visible position differs\n        This function converts real to visible\n        '
        generator = self._visibleCharPositionGenerator(text)
        for _ in range(realColumn):
            val = next(generator)
        val = next(generator)
        return val

    def _visibleToRealColumn(self, text, visiblePos):
        if False:
            for i in range(10):
                print('nop')
        'If \t is used, real position of symbol in block and visible position differs\n        This function converts visible to real.\n        Bigger value is returned, if visiblePos is in the middle of \t, None if text is too short\n        '
        if visiblePos == 0:
            return 0
        elif not '\t' in text:
            return visiblePos
        else:
            currentIndex = 1
            for currentVisiblePos in self._visibleCharPositionGenerator(text):
                if currentVisiblePos >= visiblePos:
                    return currentIndex - 1
                currentIndex += 1
            return None

    def cursors(self):
        if False:
            print('Hello World!')
        'Cursors for rectangular selection.\n        1 cursor for every line\n        '
        cursors = []
        if self._start is not None:
            (startLine, startVisibleCol) = self._start
            (currentLine, currentCol) = self._qpart.cursorPosition
            if abs(startLine - currentLine) > self._MAX_SIZE or abs(startVisibleCol - currentCol) > self._MAX_SIZE:
                self._qpart.userWarning.emit('Rectangular selection area is too big')
                self._start = None
                return []
            currentBlockText = self._qpart.textCursor().block().text()
            currentVisibleCol = self._realToVisibleColumn(currentBlockText, currentCol)
            for lineNumber in range(min(startLine, currentLine), max(startLine, currentLine) + 1):
                block = self._qpart.document().findBlockByNumber(lineNumber)
                cursor = QTextCursor(block)
                realStartCol = self._visibleToRealColumn(block.text(), startVisibleCol)
                realCurrentCol = self._visibleToRealColumn(block.text(), currentVisibleCol)
                if realStartCol is None:
                    realStartCol = block.length()
                if realCurrentCol is None:
                    realCurrentCol = block.length()
                cursor.setPosition(cursor.block().position() + min(realStartCol, block.length() - 1))
                cursor.setPosition(cursor.block().position() + min(realCurrentCol, block.length() - 1), QTextCursor.KeepAnchor)
                cursors.append(cursor)
        return cursors

    def selections(self):
        if False:
            i = 10
            return i + 15
        'Build list of extra selections for rectangular selection'
        selections = []
        cursors = self.cursors()
        if cursors:
            background = self._qpart.palette().color(QPalette.Highlight)
            foreground = self._qpart.palette().color(QPalette.HighlightedText)
            for cursor in cursors:
                selection = QTextEdit.ExtraSelection()
                selection.format.setBackground(background)
                selection.format.setForeground(foreground)
                selection.cursor = cursor
                selections.append(selection)
        return selections

    def isActive(self):
        if False:
            i = 10
            return i + 15
        'Some rectangle is selected'
        return self._start is not None

    def copy(self):
        if False:
            i = 10
            return i + 15
        'Copy to the clipboard'
        data = QMimeData()
        text = '\n'.join([cursor.selectedText() for cursor in self.cursors()])
        data.setText(text)
        data.setData(self.MIME_TYPE, text.encode('utf8'))
        QApplication.clipboard().setMimeData(data)

    def cut(self):
        if False:
            for i in range(10):
                print('nop')
        'Cut action. Copy and delete\n        '
        cursorPos = self._qpart.cursorPosition
        topLeft = (min(self._start[0], cursorPos[0]), min(self._start[1], cursorPos[1]))
        self.copy()
        self.delete()
        self._qpart.cursorPosition = topLeft

    def _indentUpTo(self, text, width):
        if False:
            while True:
                i = 10
        'Add space to text, so text width will be at least width.\n        Return text, which must be added\n        '
        visibleTextWidth = self._realToVisibleColumn(text, len(text))
        diff = width - visibleTextWidth
        if diff <= 0:
            return ''
        elif self._qpart.indentUseTabs and all((char == '\t' for char in text)):
            return '\t' * (diff // self._qpart.indentWidth) + ' ' * (diff % self._qpart.indentWidth)
        else:
            return ' ' * int(diff)

    def paste(self, mimeData):
        if False:
            print('Hello World!')
        'Paste recrangular selection.\n        Add space at the beginning of line, if necessary\n        '
        if self.isActive():
            self.delete()
        elif self._qpart.textCursor().hasSelection():
            self._qpart.textCursor().deleteChar()
        text = bytes(mimeData.data(self.MIME_TYPE)).decode('utf8')
        lines = text.splitlines()
        (cursorLine, cursorCol) = self._qpart.cursorPosition
        if cursorLine + len(lines) > len(self._qpart.lines):
            for _ in range(cursorLine + len(lines) - len(self._qpart.lines)):
                self._qpart.lines.append('')
        with self._qpart:
            for (index, line) in enumerate(lines):
                currentLine = self._qpart.lines[cursorLine + index]
                newLine = currentLine[:cursorCol] + self._indentUpTo(currentLine, cursorCol) + line + currentLine[cursorCol:]
                self._qpart.lines[cursorLine + index] = newLine
        self._qpart.cursorPosition = (cursorLine, cursorCol)

    def mousePressEvent(self, mouseEvent):
        if False:
            return 10
        cursor = self._qpart.cursorForPosition(mouseEvent.pos())
        self._start = (cursor.block().blockNumber(), cursor.positionInBlock())

    def mouseMoveEvent(self, mouseEvent):
        if False:
            return 10
        cursor = self._qpart.cursorForPosition(mouseEvent.pos())
        self._qpart.cursorPositionChanged.disconnect(self._reset)
        self._qpart.selectionChanged.disconnect(self._reset)
        self._qpart.setTextCursor(cursor)
        self._qpart.cursorPositionChanged.connect(self._reset)
        self._qpart.selectionChanged.connect(self._reset)