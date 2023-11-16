"""A simple completer for the qtconsole"""
from qtpy import QtCore, QtGui, QtWidgets
from .util import columnize

class CompletionPlain(QtWidgets.QWidget):
    """ A widget for tab completion,  navigable by arrow keys """

    def __init__(self, console_widget):
        if False:
            while True:
                i = 10
        ' Create a completion widget that is attached to the specified Qt\n            text edit widget.\n        '
        assert isinstance(console_widget._control, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__()
        self._text_edit = console_widget._control
        self._console_widget = console_widget
        self._text_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if False:
            while True:
                i = 10
        ' Reimplemented to handle keyboard input and to auto-hide when the\n            text edit loses focus.\n        '
        if obj == self._text_edit:
            etype = event.type()
            if etype in (QtCore.QEvent.KeyPress, QtCore.QEvent.FocusOut):
                self.cancel_completion()
        return super().eventFilter(obj, event)

    def cancel_completion(self):
        if False:
            print('Hello World!')
        'Cancel the completion, reseting internal variable, clearing buffer '
        self._console_widget._clear_temporary_buffer()

    def show_items(self, cursor, items, prefix_length=0):
        if False:
            while True:
                i = 10
        " Shows the completion widget with 'items' at the position specified\n            by 'cursor'.\n        "
        if not items:
            return
        self.cancel_completion()
        strng = columnize(items)
        cursor.movePosition(QtGui.QTextCursor.Left, n=prefix_length)
        self._console_widget._fill_temporary_buffer(cursor, strng, html=False)