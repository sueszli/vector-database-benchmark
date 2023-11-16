import re
from unicodedata import category
from qtpy import QT6
from qtpy import QtCore, QtGui, QtWidgets

class CallTipWidget(QtWidgets.QLabel):
    """ Shows call tips by parsing the current text of Q[Plain]TextEdit.
    """

    def __init__(self, text_edit):
        if False:
            for i in range(10):
                print('nop')
        ' Create a call tip manager that is attached to the specified Qt\n            text edit widget.\n        '
        assert isinstance(text_edit, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__(None, QtCore.Qt.ToolTip)
        text_edit.destroyed.connect(self.deleteLater)
        self._hide_timer = QtCore.QBasicTimer()
        self._text_edit = text_edit
        self.setFont(text_edit.document().defaultFont())
        self.setForegroundRole(QtGui.QPalette.ToolTipText)
        self.setBackgroundRole(QtGui.QPalette.ToolTipBase)
        self.setPalette(QtWidgets.QToolTip.palette())
        self.setAlignment(QtCore.Qt.AlignLeft)
        self.setIndent(1)
        self.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.setMargin(1 + self.style().pixelMetric(QtWidgets.QStyle.PM_ToolTipLabelFrameWidth, None, self))
        self.setWindowOpacity(self.style().styleHint(QtWidgets.QStyle.SH_ToolTipLabel_Opacity, None, self, None) / 255.0)
        self.setWordWrap(True)

    def eventFilter(self, obj, event):
        if False:
            return 10
        ' Reimplemented to hide on certain key presses and on text edit focus\n            changes.\n        '
        if obj == self._text_edit:
            etype = event.type()
            if etype == QtCore.QEvent.KeyPress:
                key = event.key()
                if key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):
                    self.hide()
                elif key == QtCore.Qt.Key_Escape:
                    self.hide()
                    return True
            elif etype == QtCore.QEvent.FocusOut:
                self.hide()
            elif etype == QtCore.QEvent.Enter:
                self._hide_timer.stop()
            elif etype == QtCore.QEvent.Leave:
                self._leave_event_hide()
        return super().eventFilter(obj, event)

    def timerEvent(self, event):
        if False:
            return 10
        ' Reimplemented to hide the widget when the hide timer fires.\n        '
        if event.timerId() == self._hide_timer.timerId():
            self._hide_timer.stop()
            self.hide()

    def enterEvent(self, event):
        if False:
            print('Hello World!')
        ' Reimplemented to cancel the hide timer.\n        '
        super().enterEvent(event)
        self._hide_timer.stop()

    def hideEvent(self, event):
        if False:
            print('Hello World!')
        ' Reimplemented to disconnect signal handlers and event filter.\n        '
        super().hideEvent(event)
        try:
            self._text_edit.cursorPositionChanged.disconnect(self._cursor_position_changed)
        except TypeError:
            pass
        self._text_edit.removeEventFilter(self)

    def leaveEvent(self, event):
        if False:
            print('Hello World!')
        ' Reimplemented to start the hide timer.\n        '
        super().leaveEvent(event)
        self._leave_event_hide()

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        ' Reimplemented to paint the background panel.\n        '
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionFrame()
        option.initFrom(self)
        painter.drawPrimitive(QtWidgets.QStyle.PE_PanelTipLabel, option)
        painter.end()
        super().paintEvent(event)

    def setFont(self, font):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to allow use of this method as a slot.\n        '
        super().setFont(font)

    def showEvent(self, event):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to connect signal handlers and event filter.\n        '
        super().showEvent(event)
        self._text_edit.cursorPositionChanged.connect(self._cursor_position_changed)
        self._text_edit.installEventFilter(self)

    def deleteLater(self):
        if False:
            i = 10
            return i + 15
        ' Avoids an error when the widget has already been deleted.\n\n            Fixes jupyter/qtconsole#507.\n        '
        try:
            return super().deleteLater()
        except RuntimeError:
            pass

    def show_inspect_data(self, content, maxlines=20):
        if False:
            for i in range(10):
                print('nop')
        'Show inspection data as a tooltip'
        data = content.get('data', {})
        text = data.get('text/plain', '')
        match = re.match('(?:[^\n]*\n){%i}' % maxlines, text)
        if match:
            text = text[:match.end()] + '\n[Documentation continues...]'
        return self.show_tip(self._format_tooltip(text))

    def show_tip(self, tip):
        if False:
            for i in range(10):
                print('nop')
        ' Attempts to show the specified tip at the current cursor location.\n        '
        text_edit = self._text_edit
        document = text_edit.document()
        cursor = text_edit.textCursor()
        search_pos = cursor.position() - 1
        (self._start_position, _) = self._find_parenthesis(search_pos, forward=False)
        if self._start_position == -1:
            return False
        self.setText(tip)
        self.resize(self.sizeHint())
        padding = 3
        cursor_rect = text_edit.cursorRect(cursor)
        if QT6:
            screen_rect = text_edit.screen().geometry()
        else:
            screen_rect = QtWidgets.QApplication.instance().desktop().screenGeometry(text_edit)
        point = text_edit.mapToGlobal(cursor_rect.bottomRight())
        point.setY(point.y() + padding)
        tip_height = self.size().height()
        tip_width = self.size().width()
        vertical = 'bottom'
        horizontal = 'Right'
        if point.y() + tip_height > screen_rect.height() + screen_rect.y():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.y() - tip_height < padding:
                if 2 * point.y() < screen_rect.height():
                    vertical = 'bottom'
                else:
                    vertical = 'top'
            else:
                vertical = 'top'
        if point.x() + tip_width > screen_rect.width() + screen_rect.x():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.x() - tip_width < padding:
                if 2 * point.x() < screen_rect.width():
                    horizontal = 'Right'
                else:
                    horizontal = 'Left'
            else:
                horizontal = 'Left'
        pos = getattr(cursor_rect, '%s%s' % (vertical, horizontal))
        point = text_edit.mapToGlobal(pos())
        point.setY(point.y() + padding)
        if vertical == 'top':
            point.setY(point.y() - tip_height)
        if horizontal == 'Left':
            point.setX(point.x() - tip_width - padding)
        self.move(point)
        self.show()
        return True

    def _find_parenthesis(self, position, forward=True):
        if False:
            print('Hello World!')
        " If 'forward' is True (resp. False), proceed forwards\n            (resp. backwards) through the line that contains 'position' until an\n            unmatched closing (resp. opening) parenthesis is found. Returns a\n            tuple containing the position of this parenthesis (or -1 if it is\n            not found) and the number commas (at depth 0) found along the way.\n        "
        commas = depth = 0
        document = self._text_edit.document()
        char = document.characterAt(position)
        while category(char) != 'Cc' and position > 0:
            if char == ',' and depth == 0:
                commas += 1
            elif char == ')':
                if forward and depth == 0:
                    break
                depth += 1
            elif char == '(':
                if not forward and depth == 0:
                    break
                depth -= 1
            position += 1 if forward else -1
            char = document.characterAt(position)
        else:
            position = -1
        return (position, commas)

    def _leave_event_hide(self):
        if False:
            i = 10
            return i + 15
        ' Hides the tooltip after some time has passed (assuming the cursor is\n            not over the tooltip).\n        '
        if not self._hide_timer.isActive() and QtWidgets.QApplication.instance().topLevelAt(QtGui.QCursor.pos()) != self:
            self._hide_timer.start(300, self)

    def _format_tooltip(self, doc):
        if False:
            print('Hello World!')
        doc = re.sub('\\033\\[(\\d|;)+?m', '', doc)
        return doc

    def _cursor_position_changed(self):
        if False:
            while True:
                i = 10
        ' Updates the tip based on user cursor movement.\n        '
        cursor = self._text_edit.textCursor()
        if cursor.position() <= self._start_position:
            self.hide()
        else:
            (position, commas) = self._find_parenthesis(self._start_position + 1)
            if position != -1:
                self.hide()