"""Array Builder Widget."""
import re
from qtpy.QtCore import QEvent, QPoint, Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem, QToolButton, QToolTip
from spyder.config.base import _
from spyder.utils.icon_manager import ima
from spyder.utils.palette import QStylePalette
from spyder.widgets.helperwidgets import HelperToolButton
SHORTCUT_TABLE = 'Ctrl+M'
SHORTCUT_INLINE = 'Ctrl+Alt+M'

class ArrayBuilderType:
    LANGUAGE = None
    ELEMENT_SEPARATOR = None
    ROW_SEPARATOR = None
    BRACES = None
    EXTRA_VALUES = None
    ARRAY_PREFIX = None
    MATRIX_PREFIX = None

    def check_values(self):
        if False:
            print('Hello World!')
        pass

class ArrayBuilderPython(ArrayBuilderType):
    ELEMENT_SEPARATOR = ', '
    ROW_SEPARATOR = ';'
    BRACES = '], ['
    EXTRA_VALUES = {'np.nan': ['nan', 'NAN', 'NaN', 'Na', 'NA', 'na'], 'np.inf': ['inf', 'INF']}
    ARRAY_PREFIX = 'np.array([['
    MATRIX_PREFIX = 'np.matrix([['
_REGISTERED_ARRAY_BUILDERS = {'python': ArrayBuilderPython}

class ArrayInline(QLineEdit):

    def __init__(self, parent, options=None):
        if False:
            for i in range(10):
                print('nop')
        super(ArrayInline, self).__init__(parent)
        self._parent = parent
        self._options = options

    def keyPressEvent(self, event):
        if False:
            print('Hello World!')
        'Override Qt method.'
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self._parent.process_text()
            if self._parent.is_valid():
                self._parent.keyPressEvent(event)
        else:
            super(ArrayInline, self).keyPressEvent(event)

    def event(self, event):
        if False:
            return 10
        '\n        Override Qt method.\n\n        This is needed to be able to intercept the Tab key press event.\n        '
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Tab or event.key() == Qt.Key_Space:
                text = self.text()
                cursor = self.cursorPosition()
                if cursor != 0 and text[cursor - 1] == ' ':
                    text = text[:cursor - 1] + self._options.ROW_SEPARATOR + ' ' + text[cursor:]
                else:
                    text = text[:cursor] + ' ' + text[cursor:]
                self.setCursorPosition(cursor)
                self.setText(text)
                self.setCursorPosition(cursor + 1)
                return False
        return super(ArrayInline, self).event(event)

class ArrayTable(QTableWidget):

    def __init__(self, parent, options=None):
        if False:
            while True:
                i = 10
        super(ArrayTable, self).__init__(parent)
        self._parent = parent
        self._options = options
        self.setRowCount(2)
        self.setColumnCount(2)
        self.reset_headers()
        self.cellChanged.connect(self.cell_changed)

    def keyPressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.'
        super(ArrayTable, self).keyPressEvent(event)
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.setDisabled(True)
            self.setDisabled(False)
            self._parent.keyPressEvent(event)

    def cell_changed(self, row, col):
        if False:
            i = 10
            return i + 15
        item = self.item(row, col)
        value = None
        if item:
            rows = self.rowCount()
            cols = self.columnCount()
            value = item.text()
        if value:
            if row == rows - 1:
                self.setRowCount(rows + 1)
            if col == cols - 1:
                self.setColumnCount(cols + 1)
        self.reset_headers()

    def reset_headers(self):
        if False:
            print('Hello World!')
        'Update the column and row numbering in the headers.'
        rows = self.rowCount()
        cols = self.columnCount()
        for r in range(rows):
            self.setVerticalHeaderItem(r, QTableWidgetItem(str(r)))
        for c in range(cols):
            self.setHorizontalHeaderItem(c, QTableWidgetItem(str(c)))
            self.setColumnWidth(c, 40)

    def text(self):
        if False:
            i = 10
            return i + 15
        'Return the entered array in a parseable form.'
        text = []
        rows = self.rowCount()
        cols = self.columnCount()
        if rows == 2 and cols == 2:
            item = self.item(0, 0)
            if item is None:
                return ''
        for r in range(rows - 1):
            for c in range(cols - 1):
                item = self.item(r, c)
                if item is not None:
                    value = item.text()
                else:
                    value = '0'
                if not value.strip():
                    value = '0'
                text.append(' ')
                text.append(value)
            text.append(self._options.ROW_SEPARATOR)
        return ''.join(text[:-1])

class ArrayBuilderDialog(QDialog):

    def __init__(self, parent=None, inline=True, offset=0, force_float=False, language='python'):
        if False:
            i = 10
            return i + 15
        super(ArrayBuilderDialog, self).__init__(parent=parent)
        self._language = language
        self._options = _REGISTERED_ARRAY_BUILDERS.get('python', None)
        self._parent = parent
        self._text = None
        self._valid = None
        self._offset = offset
        self._force_float = force_float
        self._help_inline = _("\n           <b>Numpy Array/Matrix Helper</b><br>\n           Type an array in Matlab    : <code>[1 2;3 4]</code><br>\n           or Spyder simplified syntax : <code>1 2;3 4</code>\n           <br><br>\n           Hit 'Enter' for array or 'Ctrl+Enter' for matrix.\n           <br><br>\n           <b>Hint:</b><br>\n           Use two spaces or two tabs to generate a ';'.\n           ")
        self._help_table = _("\n           <b>Numpy Array/Matrix Helper</b><br>\n           Enter an array in the table. <br>\n           Use Tab to move between cells.\n           <br><br>\n           Hit 'Enter' for array or 'Ctrl+Enter' for matrix.\n           <br><br>\n           <b>Hint:</b><br>\n           Use two tabs at the end of a row to move to the next row.\n           ")
        self._button_warning = QToolButton()
        self._button_help = HelperToolButton()
        self._button_help.setIcon(ima.icon('MessageBoxInformation'))
        style = '\n            QToolButton {{\n              border: 1px solid grey;\n              padding:0px;\n              border-radius: 2px;\n              background-color: qlineargradient(x1: 1, y1: 1, x2: 1, y2: 1,\n                  stop: 0 {stop_0}, stop: 1 {stop_1});\n            }}\n            '.format(stop_0=QStylePalette.COLOR_BACKGROUND_4, stop_1=QStylePalette.COLOR_BACKGROUND_2)
        self._button_help.setStyleSheet(style)
        if inline:
            self._button_help.setToolTip(self._help_inline)
            self._text = ArrayInline(self, options=self._options)
            self._widget = self._text
        else:
            self._button_help.setToolTip(self._help_table)
            self._table = ArrayTable(self, options=self._options)
            self._widget = self._table
        style = '\n            QDialog {\n              margin:0px;\n              border: 1px solid grey;\n              padding:0px;\n              border-radius: 2px;\n            }'
        self.setStyleSheet(style)
        style = '\n            QToolButton {\n              margin:1px;\n              border: 0px solid grey;\n              padding:0px;\n              border-radius: 0px;\n            }'
        self._button_warning.setStyleSheet(style)
        self.setWindowFlags(Qt.Window | Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setWindowOpacity(0.9)
        self._widget.setMinimumWidth(200)
        self._layout = QHBoxLayout()
        self._layout.addWidget(self._widget)
        self._layout.addWidget(self._button_warning, 1, Qt.AlignTop)
        self._layout.addWidget(self._button_help, 1, Qt.AlignTop)
        self.setLayout(self._layout)
        self._widget.setFocus()

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override Qt method.'
        QToolTip.hideText()
        ctrl = event.modifiers() & Qt.ControlModifier
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            if ctrl:
                self.process_text(array=False)
            else:
                self.process_text(array=True)
            self.accept()
        else:
            super(ArrayBuilderDialog, self).keyPressEvent(event)

    def event(self, event):
        if False:
            print('Hello World!')
        '\n        Override Qt method.\n\n        Useful when in line edit mode.\n        '
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Tab:
            return False
        return super(ArrayBuilderDialog, self).event(event)

    def process_text(self, array=True):
        if False:
            return 10
        '\n        Construct the text based on the entered content in the widget.\n        '
        if array:
            prefix = self._options.ARRAY_PREFIX
        else:
            prefix = self._options.MATRIX_PREFIX
        suffix = ']])'
        values = self._widget.text().strip()
        if values != '':
            exp = '(\\s*)' + self._options.ROW_SEPARATOR + '(\\s*)'
            values = re.sub(exp, self._options.ROW_SEPARATOR, values)
            values = re.sub('\\s+', ' ', values)
            values = re.sub(']$', '', values)
            values = re.sub('^\\[', '', values)
            values = re.sub(self._options.ROW_SEPARATOR + '*$', '', values)
            values = values.replace(' ', self._options.ELEMENT_SEPARATOR)
            new_values = []
            rows = values.split(self._options.ROW_SEPARATOR)
            nrows = len(rows)
            ncols = []
            for row in rows:
                new_row = []
                elements = row.split(self._options.ELEMENT_SEPARATOR)
                ncols.append(len(elements))
                for e in elements:
                    num = e
                    for (key, values) in self._options.EXTRA_VALUES.items():
                        if num in values:
                            num = key
                    if self._force_float:
                        try:
                            num = str(float(e))
                        except:
                            pass
                    new_row.append(num)
                new_values.append(self._options.ELEMENT_SEPARATOR.join(new_row))
            new_values = self._options.ROW_SEPARATOR.join(new_values)
            values = new_values
            if len(set(ncols)) == 1:
                self._valid = True
            else:
                self._valid = False
            if nrows == 1:
                prefix = prefix[:-1]
                suffix = suffix.replace(']])', '])')
            offset = self._offset
            braces = self._options.BRACES.replace(' ', '\n' + ' ' * (offset + len(prefix) - 1))
            values = values.replace(self._options.ROW_SEPARATOR, braces)
            text = '{0}{1}{2}'.format(prefix, values, suffix)
            self._text = text
        else:
            self._text = ''
        self.update_warning()

    def update_warning(self):
        if False:
            print('Hello World!')
        '\n        Updates the icon and tip based on the validity of the array content.\n        '
        widget = self._button_warning
        if not self.is_valid():
            tip = _('Array dimensions not valid')
            widget.setIcon(ima.icon('MessageBoxWarning'))
            widget.setToolTip(tip)
            QToolTip.showText(self._widget.mapToGlobal(QPoint(0, 5)), tip)
        else:
            self._button_warning.setToolTip('')

    def is_valid(self):
        if False:
            return 10
        'Return if the current array state is valid.'
        return self._valid

    def text(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the parsed array/matrix text.'
        return self._text

    @property
    def array_widget(self):
        if False:
            i = 10
            return i + 15
        'Return the array builder widget.'
        return self._widget

def test():
    if False:
        return 10
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg_table = ArrayBuilderDialog(None, inline=False)
    dlg_inline = ArrayBuilderDialog(None, inline=True)
    dlg_table.show()
    dlg_inline.show()
    app.exec_()
if __name__ == '__main__':
    test()