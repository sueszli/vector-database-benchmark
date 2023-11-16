"""
This module contains the edge line panel
"""
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QColor
from spyder.plugins.editor.api.panel import Panel

class EdgeLine(Panel):
    """Source code editor's edge line (default: 79 columns, PEP8)"""

    def __init__(self):
        if False:
            return 10
        Panel.__init__(self)
        self.columns = (79,)
        self.color = Qt.darkGray

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        'Override Qt method'
        painter = QPainter(self)
        size = self.size()
        color = QColor(self.color)
        color.setAlphaF(0.5)
        painter.setPen(color)
        for column in self.columns:
            x = self.editor.fontMetrics().width(column * '9') + 3
            painter.drawLine(x, 0, x, size.height())

    def sizeHint(self):
        if False:
            return 10
        'Override Qt method.'
        return self.size()

    def set_enabled(self, state):
        if False:
            while True:
                i = 10
        'Toggle edge line visibility'
        self._enabled = state
        self.setVisible(state)

    def set_columns(self, columns):
        if False:
            return 10
        'Set edge line columns values.'
        if isinstance(columns, tuple):
            self.columns = columns
        elif columns:
            columns = str(columns)
            self.columns = tuple((int(e) for e in columns.split(',')))
        self.update()

    def update_color(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set edgeline color using syntax highlighter color for comments\n        '
        self.color = self.editor.highlighter.get_color_name('comment')