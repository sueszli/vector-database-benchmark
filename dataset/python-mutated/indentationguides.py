"""
This module contains the indentation guide panel.
"""
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QColor
from spyder.plugins.editor.api.panel import Panel

class IndentationGuide(Panel):
    """Indentation guides to easy identify nested blocks."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize IndentationGuide panel.\n        i_width(int): identation width in characters.\n        '
        Panel.__init__(self)
        self.color = Qt.darkGray
        self.i_width = 4
        self.bar_offset = 0

    def on_install(self, editor):
        if False:
            for i in range(10):
                print('nop')
        'Manages install setup of the pane.'
        super().on_install(editor)
        horizontal_scrollbar = editor.horizontalScrollBar()
        horizontal_scrollbar.valueChanged.connect(self.update_bar_position)
        horizontal_scrollbar.sliderReleased.connect(self.update)

    def update_bar_position(self, value):
        if False:
            i = 10
            return i + 15
        self.bar_offset = value

    def sizeHint(self):
        if False:
            print('Hello World!')
        'Override Qt method.'
        return self.size()

    def paintEvent(self, event):
        if False:
            return 10
        '\n        Overriden Qt method.\n\n        Paint indent guides.\n        '
        painter = QPainter(self)
        color = QColor(self.color)
        color.setAlphaF(0.5)
        painter.setPen(color)
        offset = self.editor.document().documentMargin() + self.editor.contentOffset().x()
        folding_panel = self.editor.panels.get('FoldingPanel')
        folding_regions = folding_panel.folding_regions
        leading_whitespaces = self.editor.leading_whitespaces
        visible_blocks = self.editor.get_visible_block_numbers()
        for start_line in folding_regions:
            end_line = folding_regions[start_line]
            line_numbers = (start_line, end_line)
            if self.do_paint(visible_blocks, line_numbers):
                start_block = self.editor.document().findBlockByNumber(start_line)
                end_block = self.editor.document().findBlockByNumber(end_line - 1)
                content_offset = self.editor.contentOffset()
                top = int(self.editor.blockBoundingGeometry(start_block).translated(content_offset).top())
                bottom = int(self.editor.blockBoundingGeometry(end_block).translated(content_offset).bottom())
                total_whitespace = leading_whitespaces.get(max(start_line - 1, 0))
                end_whitespace = leading_whitespaces.get(end_line - 1)
                if end_whitespace and end_whitespace != total_whitespace:
                    font_metrics = self.editor.fontMetrics()
                    x = int(font_metrics.width(total_whitespace * '9') + self.bar_offset + offset)
                    painter.drawLine(x, top, x, bottom)

    def set_enabled(self, state):
        if False:
            print('Hello World!')
        'Toggle edge line visibility.'
        self._enabled = state
        self.setVisible(state)
        self.editor.request_folding()

    def update_color(self):
        if False:
            print('Hello World!')
        'Set color using syntax highlighter color for comments.'
        self.color = self.editor.highlighter.get_color_name('comment')

    def set_indentation_width(self, indentation_width):
        if False:
            print('Hello World!')
        'Set indentation width to be used to draw indent guides.'
        self.i_width = indentation_width

    def do_paint(self, visible_blocks, line_numbers):
        if False:
            return 10
        '\n        Decide if we need to paint an indent guide according to the\n        visible region.\n        '
        first_visible_line = visible_blocks[0] + 1
        last_visible_line = visible_blocks[1] + 1
        start_line = line_numbers[0]
        end_line = line_numbers[1]
        if start_line < first_visible_line and first_visible_line <= end_line <= last_visible_line:
            return True
        if start_line <= first_visible_line and end_line >= last_visible_line:
            return True
        if first_visible_line <= start_line <= last_visible_line and end_line > last_visible_line:
            return True
        if first_visible_line <= start_line <= last_visible_line and first_visible_line <= end_line <= last_visible_line:
            return True
        return False