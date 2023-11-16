"""A dropdown completer widget for the qtconsole."""
import os
import sys
from qtpy import QT6
from qtpy import QtCore, QtGui, QtWidgets

class CompletionWidget(QtWidgets.QListWidget):
    """ A widget for GUI tab completion.
    """

    def __init__(self, console_widget, height=0):
        if False:
            return 10
        ' Create a completion widget that is attached to the specified Qt\n            text edit widget.\n        '
        text_edit = console_widget._control
        assert isinstance(text_edit, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__(parent=console_widget)
        self._text_edit = text_edit
        self._height_max = height if height > 0 else self.sizeHint().height()
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setWindowFlags(QtCore.Qt.Popup)
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        original_policy = text_edit.focusPolicy()
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        text_edit.setFocusPolicy(original_policy)
        self.setFocusProxy(self._text_edit)
        self.setFrameShadow(QtWidgets.QFrame.Plain)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.itemActivated.connect(self._complete_current)

    def eventFilter(self, obj, event):
        if False:
            while True:
                i = 10
        ' Reimplemented to handle mouse input and to auto-hide when the\n            text edit loses focus.\n        '
        if obj is self:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                pos = self.mapToGlobal(event.pos())
                target = QtWidgets.QApplication.widgetAt(pos)
                if target and self.isAncestorOf(target) or target is self:
                    return False
                else:
                    self.cancel_completion()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if False:
            while True:
                i = 10
        key = event.key()
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Tab):
            self._complete_current()
        elif key == QtCore.Qt.Key_Escape:
            self.hide()
        elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown, QtCore.Qt.Key_Home, QtCore.Qt.Key_End):
            return super().keyPressEvent(event)
        else:
            QtWidgets.QApplication.sendEvent(self._text_edit, event)

    def hideEvent(self, event):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to disconnect signal handlers and event filter.\n        '
        super().hideEvent(event)
        try:
            self._text_edit.cursorPositionChanged.disconnect(self._update_current)
        except TypeError:
            pass
        self.removeEventFilter(self)

    def showEvent(self, event):
        if False:
            return 10
        ' Reimplemented to connect signal handlers and event filter.\n        '
        super().showEvent(event)
        self._text_edit.cursorPositionChanged.connect(self._update_current)
        self.installEventFilter(self)

    def show_items(self, cursor, items, prefix_length=0):
        if False:
            while True:
                i = 10
        " Shows the completion widget with 'items' at the position specified\n            by 'cursor'.\n        "
        point = self._get_top_left_position(cursor)
        self.clear()
        path_items = []
        for item in items:
            if os.path.isfile(os.path.abspath(item.replace('"', ''))) or os.path.isdir(os.path.abspath(item.replace('"', ''))):
                path_items.append(item.replace('"', ''))
            else:
                list_item = QtWidgets.QListWidgetItem()
                list_item.setData(QtCore.Qt.UserRole, item)
                list_item.setText(item.split('.')[-1])
                self.addItem(list_item)
        common_prefix = os.path.dirname(os.path.commonprefix(path_items))
        for path_item in path_items:
            list_item = QtWidgets.QListWidgetItem()
            list_item.setData(QtCore.Qt.UserRole, path_item)
            if common_prefix:
                text = path_item.split(common_prefix)[-1]
            else:
                text = path_item
            list_item.setText(text)
            self.addItem(list_item)
        if QT6:
            screen_rect = self.screen().availableGeometry()
        else:
            screen_rect = QtWidgets.QApplication.desktop().availableGeometry(self)
        screen_height = screen_rect.height()
        height = int(min(self._height_max, screen_height - 50))
        if screen_height - point.y() - height < 0:
            point = self._text_edit.mapToGlobal(self._text_edit.cursorRect().topRight())
            py = point.y()
            point.setY(int(py - min(height, py - 10)))
        w = self.sizeHintForColumn(0) + self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth()
        self.setGeometry(point.x(), point.y(), w, height)
        cursor.movePosition(QtGui.QTextCursor.Left, n=prefix_length)
        self._start_position = cursor.position()
        self.setCurrentRow(0)
        self.raise_()
        self.show()

    def _get_top_left_position(self, cursor):
        if False:
            i = 10
            return i + 15
        ' Get top left position for this widget.\n        '
        return self._text_edit.mapToGlobal(self._text_edit.cursorRect().bottomRight())

    def _complete_current(self):
        if False:
            while True:
                i = 10
        ' Perform the completion with the currently selected item.\n        '
        text = self.currentItem().data(QtCore.Qt.UserRole)
        self._current_text_cursor().insertText(text)
        self.hide()

    def _current_text_cursor(self):
        if False:
            i = 10
            return i + 15
        ' Returns a cursor with text between the start position and the\n            current position selected.\n        '
        cursor = self._text_edit.textCursor()
        if cursor.position() >= self._start_position:
            cursor.setPosition(self._start_position, QtGui.QTextCursor.KeepAnchor)
        return cursor

    def _update_current(self):
        if False:
            return 10
        ' Updates the current item based on the current text and the\n            position of the widget.\n        '
        cursor = self._text_edit.textCursor()
        point = self._get_top_left_position(cursor)
        point.setY(self.y())
        self.move(point)
        prefix = self._current_text_cursor().selection().toPlainText()
        if prefix:
            items = self.findItems(prefix, QtCore.Qt.MatchStartsWith | QtCore.Qt.MatchCaseSensitive)
            if items:
                self.setCurrentItem(items[0])
            else:
                self.hide()
        else:
            self.hide()

    def cancel_completion(self):
        if False:
            return 10
        self.hide()