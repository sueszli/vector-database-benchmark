"""A navigable completer for the qtconsole"""
from qtpy import QtCore, QtGui, QtWidgets
from .util import compute_item_matrix

def html_tableify(item_matrix, select=None, header=None, footer=None):
    if False:
        return 10
    ' returnr a string for an html table'
    if not item_matrix:
        return ''
    html_cols = []
    tds = lambda text: '<td>' + text + '  </td>'
    trs = lambda text: '<tr>' + text + '</tr>'
    tds_items = [list(map(tds, row)) for row in item_matrix]
    if select:
        (row, col) = select
        tds_items[row][col] = '<td class="inverted">' + item_matrix[row][col] + '  </td>'
    html_cols = map(trs, (''.join(row) for row in tds_items))
    head = ''
    foot = ''
    if header:
        head = '<tr>' + ''.join(('<td>' + header + '</td>') * len(item_matrix[0])) + '</tr>'
    if footer:
        foot = '<tr>' + ''.join(('<td>' + footer + '</td>') * len(item_matrix[0])) + '</tr>'
    html = '<table class="completion" style="white-space:pre"cellspacing=0>' + head + ''.join(html_cols) + foot + '</table>'
    return html

class SlidingInterval(object):
    """a bound interval that follows a cursor

    internally used to scoll the completion view when the cursor
    try to go beyond the edges, and show '...' when rows are hidden
    """
    _min = 0
    _max = 1
    _current = 0

    def __init__(self, maximum=1, width=6, minimum=0, sticky_lenght=1):
        if False:
            i = 10
            return i + 15
        "Create a new bounded interval\n\n        any value return by this will be bound between maximum and\n        minimum. usual width will be 'width', and sticky_length\n        set when the return  interval should expand to max and min\n        "
        self._min = minimum
        self._max = maximum
        self._start = 0
        self._width = width
        self._stop = self._start + self._width + 1
        self._sticky_lenght = sticky_lenght

    @property
    def current(self):
        if False:
            i = 10
            return i + 15
        'current cursor position'
        return self._current

    @current.setter
    def current(self, value):
        if False:
            i = 10
            return i + 15
        'set current cursor position'
        current = min(max(self._min, value), self._max)
        self._current = current
        if current > self._stop:
            self._stop = current
            self._start = current - self._width
        elif current < self._start:
            self._start = current
            self._stop = current + self._width
        if abs(self._start - self._min) <= self._sticky_lenght:
            self._start = self._min
        if abs(self._stop - self._max) <= self._sticky_lenght:
            self._stop = self._max

    @property
    def start(self):
        if False:
            print('Hello World!')
        'begiiing of interval to show'
        return self._start

    @property
    def stop(self):
        if False:
            print('Hello World!')
        'end of interval to show'
        return self._stop

    @property
    def width(self):
        if False:
            while True:
                i = 10
        return self._stop - self._start

    @property
    def nth(self):
        if False:
            print('Hello World!')
        return self.current - self.start

class CompletionHtml(QtWidgets.QWidget):
    """ A widget for tab completion,  navigable by arrow keys """
    _items = ()
    _index = (0, 0)
    _consecutive_tab = 0
    _size = (1, 1)
    _old_cursor = None
    _start_position = 0
    _slice_start = 0
    _slice_len = 4

    def __init__(self, console_widget, rows=10):
        if False:
            i = 10
            return i + 15
        ' Create a completion widget that is attached to the specified Qt\n            text edit widget.\n        '
        assert isinstance(console_widget._control, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__()
        self._text_edit = console_widget._control
        self._console_widget = console_widget
        self._rows = rows if rows > 0 else 10
        self._text_edit.installEventFilter(self)
        self._sliding_interval = None
        self._justified_items = None
        self.setFocusProxy(self._text_edit)

    def eventFilter(self, obj, event):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to handle keyboard input and to auto-hide when the\n            text edit loses focus.\n        '
        if obj == self._text_edit:
            etype = event.type()
            if etype == QtCore.QEvent.KeyPress:
                key = event.key()
                if self._consecutive_tab == 0 and key in (QtCore.Qt.Key_Tab,):
                    return False
                elif self._consecutive_tab == 1 and key in (QtCore.Qt.Key_Tab,):
                    self._consecutive_tab = self._consecutive_tab + 1
                    self._update_list()
                    return True
                elif self._consecutive_tab == 2:
                    if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                        self._complete_current()
                        return True
                    if key in (QtCore.Qt.Key_Tab,):
                        self.select_right()
                        self._update_list()
                        return True
                    elif key in (QtCore.Qt.Key_Down,):
                        self.select_down()
                        self._update_list()
                        return True
                    elif key in (QtCore.Qt.Key_Right,):
                        self.select_right()
                        self._update_list()
                        return True
                    elif key in (QtCore.Qt.Key_Up,):
                        self.select_up()
                        self._update_list()
                        return True
                    elif key in (QtCore.Qt.Key_Left,):
                        self.select_left()
                        self._update_list()
                        return True
                    elif key in (QtCore.Qt.Key_Escape,):
                        self.cancel_completion()
                        return True
                    else:
                        self.cancel_completion()
                else:
                    self.cancel_completion()
            elif etype == QtCore.QEvent.FocusOut:
                self.cancel_completion()
        return super().eventFilter(obj, event)

    def cancel_completion(self):
        if False:
            return 10
        'Cancel the completion\n\n        should be called when the completer have to be dismissed\n\n        This reset internal variable, clearing the temporary buffer\n        of the console where the completion are shown.\n        '
        self._consecutive_tab = 0
        self._slice_start = 0
        self._console_widget._clear_temporary_buffer()
        self._index = (0, 0)
        if self._sliding_interval:
            self._sliding_interval = None

    def _select_index(self, row, col):
        if False:
            while True:
                i = 10
        'Change the selection index, and make sure it stays in the right range\n\n        A little more complicated than just dooing modulo the number of row columns\n        to be sure to cycle through all element.\n\n        horizontaly, the element are maped like this :\n        to r <-- a b c d e f --> to g\n        to f <-- g h i j k l --> to m\n        to l <-- m n o p q r --> to a\n\n        and vertically\n        a d g j m p\n        b e h k n q\n        c f i l o r\n        '
        (nr, nc) = self._size
        nr = nr - 1
        nc = nc - 1
        if row > nr and col >= nc or (row >= nr and col > nc):
            self._select_index(0, 0)
        elif row <= 0 and col < 0 or (row < 0 and col <= 0):
            self._select_index(nr, nc)
        elif row > nr:
            self._select_index(0, col + 1)
        elif row < 0:
            self._select_index(nr, col - 1)
        elif col > nc:
            self._select_index(row + 1, 0)
        elif col < 0:
            self._select_index(row - 1, nc)
        elif 0 <= row and row <= nr and (0 <= col) and (col <= nc):
            self._index = (row, col)
        else:
            raise NotImplementedError("you'r trying to go where no completion                           have gone before : %d:%d (%d:%d)" % (row, col, nr, nc))

    @property
    def _slice_end(self):
        if False:
            for i in range(10):
                print('nop')
        end = self._slice_start + self._slice_len
        if end > len(self._items):
            return None
        return end

    def select_up(self):
        if False:
            return 10
        'move cursor up'
        (r, c) = self._index
        self._select_index(r - 1, c)

    def select_down(self):
        if False:
            return 10
        'move cursor down'
        (r, c) = self._index
        self._select_index(r + 1, c)

    def select_left(self):
        if False:
            for i in range(10):
                print('nop')
        'move cursor left'
        (r, c) = self._index
        self._select_index(r, c - 1)

    def select_right(self):
        if False:
            i = 10
            return i + 15
        'move cursor right'
        (r, c) = self._index
        self._select_index(r, c + 1)

    def show_items(self, cursor, items, prefix_length=0):
        if False:
            i = 10
            return i + 15
        " Shows the completion widget with 'items' at the position specified\n            by 'cursor'.\n        "
        if not items:
            return
        cursor.movePosition(QtGui.QTextCursor.Left, n=prefix_length)
        self._start_position = cursor.position()
        self._consecutive_tab = 1
        width = self._text_edit.document().textWidth()
        char_width = self._console_widget._get_font_width()
        displaywidth = int(max(10, width / char_width - 1))
        (items_m, ci) = compute_item_matrix(items, empty=' ', displaywidth=displaywidth)
        self._sliding_interval = SlidingInterval(len(items_m) - 1, width=self._rows)
        self._items = items_m
        self._size = (ci['rows_numbers'], ci['columns_numbers'])
        self._old_cursor = cursor
        self._index = (0, 0)
        sjoin = lambda x: [y.ljust(w, ' ') for (y, w) in zip(x, ci['columns_width'])]
        self._justified_items = list(map(sjoin, items_m))
        self._update_list(hilight=False)

    def _update_list(self, hilight=True):
        if False:
            while True:
                i = 10
        ' update the list of completion and hilight the currently selected completion '
        self._sliding_interval.current = self._index[0]
        head = None
        foot = None
        if self._sliding_interval.start > 0:
            head = '...'
        if self._sliding_interval.stop < self._sliding_interval._max:
            foot = '...'
        items_m = self._justified_items[self._sliding_interval.start:self._sliding_interval.stop + 1]
        self._console_widget._clear_temporary_buffer()
        if hilight:
            sel = (self._sliding_interval.nth, self._index[1])
        else:
            sel = None
        strng = html_tableify(items_m, select=sel, header=head, footer=foot)
        self._console_widget._fill_temporary_buffer(self._old_cursor, strng, html=True)

    def _complete_current(self):
        if False:
            print('Hello World!')
        ' Perform the completion with the currently selected item.\n        '
        i = self._index
        item = self._items[i[0]][i[1]]
        item = item.strip()
        if item:
            self._current_text_cursor().insertText(item)
        self.cancel_completion()

    def _current_text_cursor(self):
        if False:
            i = 10
            return i + 15
        ' Returns a cursor with text between the start position and the\n            current position selected.\n        '
        cursor = self._text_edit.textCursor()
        if cursor.position() >= self._start_position:
            cursor.setPosition(self._start_position, QtGui.QTextCursor.KeepAnchor)
        return cursor