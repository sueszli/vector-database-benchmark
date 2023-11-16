"""
This module contains the text decoration API.

Adapted from pyqode/core/api/decoration.py of the
`PyQode project <https://github.com/pyQode/pyQode>`_.
Original file:
<https://github.com/pyQode/pyqode.core/blob/master/pyqode/core/api/decoration.py>
"""
from qtpy.QtWidgets import QTextEdit
from qtpy.QtCore import QObject, Signal, Qt
from qtpy.QtGui import QTextCursor, QFont, QPen, QColor, QTextFormat, QTextCharFormat
from spyder.utils.palette import QStylePalette, SpyderPalette
DRAW_ORDERS = {'on_bottom': 0, 'current_cell': 1, 'codefolding': 2, 'current_line': 3, 'on_top': 4}

class TextDecoration(QTextEdit.ExtraSelection):
    """
    Helper class to quickly create a text decoration.

    The text decoration is an utility class that adds a few utility methods to
    QTextEdit.ExtraSelection.

    In addition to the helper methods, a tooltip can be added to a decoration.
    (useful for errors markers and so on...)

    Text decoration expose a **clicked** signal stored in a separate QObject:
        :attr:`pyqode.core.api.TextDecoration.Signals`

    .. code-block:: python

        deco = TextDecoration()
        deco.signals.clicked.connect(a_slot)

        def a_slot(decoration):
            print(decoration)  # spyder: test-skip
    """

    class Signals(QObject):
        """
        Holds the signals for a TextDecoration (since we cannot make it a
        QObject, we need to store its signals in an external QObject).
        """
        clicked = Signal(object)

    def __init__(self, cursor_or_bloc_or_doc, start_pos=None, end_pos=None, start_line=None, end_line=None, draw_order=0, tooltip=None, full_width=False, font=None, kind=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a text decoration.\n\n        .. note:: start_pos/end_pos and start_line/end_line pairs let you\n            easily specify the selected text. You should use one pair or the\n            other or they will conflict between each others. If you don't\n            specify any values, the selection will be based on the cursor.\n\n        :param cursor_or_bloc_or_doc: Reference to a valid\n            QTextCursor/QTextBlock/QTextDocument\n        :param start_pos: Selection start position\n        :param end_pos: Selection end position\n        :param start_line: Selection start line.\n        :param end_line: Selection end line.\n        :param draw_order: The draw order of the selection, highest values will\n            appear on top of the lowest values.\n        :param tooltip: An optional tooltips that will be automatically shown\n            when the mouse cursor hover the decoration.\n        :param full_width: True to select the full line width.\n        :param font: Decoration font.\n        :param kind: Decoration kind, e.g. 'current_cell'.\n\n        .. note:: Use the cursor selection if startPos and endPos are none.\n        "
        super(TextDecoration, self).__init__()
        self.signals = self.Signals()
        self.draw_order = draw_order
        self.tooltip = tooltip
        self.cursor = QTextCursor(cursor_or_bloc_or_doc)
        self.kind = kind
        if full_width:
            self.set_full_width(full_width)
        if start_pos is not None:
            self.cursor.setPosition(start_pos)
        if end_pos is not None:
            self.cursor.setPosition(end_pos, QTextCursor.KeepAnchor)
        if start_line is not None:
            self.cursor.movePosition(self.cursor.Start, self.cursor.MoveAnchor)
            self.cursor.movePosition(self.cursor.Down, self.cursor.MoveAnchor, start_line)
        if end_line is not None:
            self.cursor.movePosition(self.cursor.Down, self.cursor.KeepAnchor, end_line - start_line)
        if font is not None:
            self.format.setFont(font)

    def contains_cursor(self, cursor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the textCursor is in the decoration.\n\n        :param cursor: The text cursor to test\n        :type cursor: QtGui.QTextCursor\n\n        :returns: True if the cursor is over the selection\n        '
        start = self.cursor.selectionStart()
        end = self.cursor.selectionEnd()
        if cursor.atBlockEnd():
            end -= 1
        return start <= cursor.position() <= end

    def set_as_bold(self):
        if False:
            for i in range(10):
                print('nop')
        'Uses bold text.'
        self.format.setFontWeight(QFont.Bold)

    def set_foreground(self, color):
        if False:
            return 10
        'Sets the foreground color.\n        :param color: Color\n        :type color: QtGui.QColor\n        '
        self.format.setForeground(color)

    def set_background(self, brush):
        if False:
            print('Hello World!')
        '\n        Sets the background brush.\n\n        :param brush: Brush\n        :type brush: QtGui.QBrush\n        '
        self.format.setBackground(brush)

    def set_outline(self, color):
        if False:
            return 10
        '\n        Uses an outline rectangle.\n\n        :param color: Color of the outline rect\n        :type color: QtGui.QColor\n        '
        self.format.setProperty(QTextFormat.OutlinePen, QPen(color))

    def select_line(self):
        if False:
            print('Hello World!')
        '\n        Select the entire line but starts at the first non whitespace character\n        and stops at the non-whitespace character.\n        :return:\n        '
        self.cursor.movePosition(self.cursor.StartOfBlock)
        text = self.cursor.block().text()
        lindent = len(text) - len(text.lstrip())
        self.cursor.setPosition(self.cursor.block().position() + lindent)
        self.cursor.movePosition(self.cursor.EndOfBlock, self.cursor.KeepAnchor)

    def set_full_width(self, flag=True, clear=True):
        if False:
            print('Hello World!')
        '\n        Enables FullWidthSelection (the selection does not stops at after the\n        character instead it goes up to the right side of the widget).\n\n        :param flag: True to use full width selection.\n        :type flag: bool\n\n        :param clear: True to clear any previous selection. Default is True.\n        :type clear: bool\n        '
        if clear:
            self.cursor.clearSelection()
        self.format.setProperty(QTextFormat.FullWidthSelection, flag)

    def set_as_underlined(self, color=Qt.blue):
        if False:
            while True:
                i = 10
        '\n        Underlines the text.\n\n        :param color: underline color.\n        '
        self.format.setUnderlineStyle(QTextCharFormat.SingleUnderline)
        self.format.setUnderlineColor(color)

    def set_as_spell_check(self, color=Qt.blue):
        if False:
            i = 10
            return i + 15
        '\n        Underlines text as a spellcheck error.\n\n        :param color: Underline color\n        :type color: QtGui.QColor\n        '
        self.format.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)
        self.format.setUnderlineColor(color)

    def set_as_error(self, color=SpyderPalette.COLOR_ERROR_2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Highlights text as a syntax error.\n\n        :param color: Underline color\n        :type color: QtGui.QColor\n        '
        self.format.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        self.format.setUnderlineColor(color)

    def set_as_warning(self, color=QColor(SpyderPalette.COLOR_WARN_1)):
        if False:
            print('Hello World!')
        '\n        Highlights text as a syntax warning.\n\n        :param color: Underline color\n        :type color: QtGui.QColor\n        '
        self.format.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        self.format.setUnderlineColor(color)