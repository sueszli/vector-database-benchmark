"""
Tests for editor panels.
"""
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QPainter, QColor, QFontMetrics
import pytest
from spyder.plugins.editor.api.panel import Panel

class EmojiPanel(Panel):
    """Example external panel."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize panel.'
        Panel.__init__(self)
        self.setMouseTracking(True)
        self.scrollable = True

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.\n        Returns the widget size hint (based on the editor font size).\n        '
        fm = QFontMetrics(self.editor.font())
        size_hint = QSize(fm.height(), fm.height())
        if size_hint.width() > 16:
            size_hint.setWidth(16)
        return size_hint

    def _draw_red(self, top, painter):
        if False:
            return 10
        'Draw emojis.\n\n        Arguments\n        ---------\n        top: int\n            top of the line to draw the emoji\n        painter: QPainter\n            QPainter instance\n        '
        painter.setPen(QColor('white'))
        font_height = self.editor.fontMetrics().height()
        painter.drawText(0, top, self.sizeHint().width(), font_height, int(Qt.AlignRight | Qt.AlignBottom), '👀')

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.\n        Paint emojis.\n        '
        super(EmojiPanel, self).paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.editor.sideareas_color)
        for (top, __, __) in self.editor.visible_blocks:
            self._draw_red(top, painter)

@pytest.mark.parametrize('position', [Panel.Position.LEFT, Panel.Position.RIGHT, Panel.Position.TOP, Panel.Position.BOTTOM, Panel.Position.FLOATING])
def test_register_panel(setup_editor, position):
    if False:
        print('Hello World!')
    'Test registering an example external panel in the editor.'
    (editor_stack, editor) = setup_editor
    editor_stack.register_panel(EmojiPanel, position=position)
    new_panel = editor.panels.get(EmojiPanel)
    assert new_panel is not None
    assert (EmojiPanel, (), {}, position) in editor_stack.external_panels
    finfo = editor_stack.new('foo.py', 'utf-8', 'hola = 3\n')
    editor2 = finfo.editor
    new_panel = editor2.panels.get(EmojiPanel)
    assert new_panel is not None
    editor_stack.external_panels = []
    editor.panels.remove(EmojiPanel)
    editor2.panels.remove(EmojiPanel)
if __name__ == '__main__':
    pytest.main()