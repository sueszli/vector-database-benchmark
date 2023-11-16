"""
This module contains the DebuggerPanel panel
"""
from qtpy.QtCore import QSize, Qt, QRect, Slot
from qtpy.QtGui import QPainter, QFontMetrics
from spyder.config.base import debug_print
from spyder.plugins.editor.api.panel import Panel
from spyder.utils.icon_manager import ima

class DebuggerPanel(Panel):
    """Debugger panel for show information about the debugging in process."""

    def __init__(self, breakpoints_manager):
        if False:
            return 10
        'Initialize panel.'
        Panel.__init__(self)
        self.breakpoints_manager = breakpoints_manager
        self.setMouseTracking(True)
        self.scrollable = True
        self.line_number_hint = None
        self._current_line_arrow = None
        self.stop = False
        self.icons = {'breakpoint': ima.icon('breakpoint_big'), 'transparent': ima.icon('breakpoint_transparent'), 'condition': ima.icon('breakpoint_cond_big'), 'arrow': ima.icon('arrow_debugger')}

    def set_current_line_arrow(self, n):
        if False:
            while True:
                i = 10
        self._current_line_arrow = n

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.\n\n        Returns the widget size hint (based on the editor font size).\n        '
        fm = QFontMetrics(self.editor.font())
        size_hint = QSize(fm.height(), fm.height())
        if size_hint.width() > 16:
            size_hint.setWidth(16)
        return size_hint

    def _draw_breakpoint_icon(self, top, painter, icon_name):
        if False:
            for i in range(10):
                print('nop')
        'Draw the given breakpoint pixmap.\n\n        Args:\n            top (int): top of the line to draw the breakpoint icon.\n            painter (QPainter)\n            icon_name (srt): key of icon to draw (see: self.icons)\n        '
        rect = QRect(0, top, self.sizeHint().width(), self.sizeHint().height())
        try:
            icon = self.icons[icon_name]
        except KeyError as e:
            debug_print("Breakpoint icon doesn't exist, {}".format(e))
        else:
            icon.paint(painter, rect)

    @Slot()
    def stop_clean(self):
        if False:
            return 10
        'Handle debugging state. The debugging is not running.'
        self.stop = True
        self.update()

    @Slot()
    def start_clean(self):
        if False:
            print('Hello World!')
        'Handle debugging state. The debugging is running.'
        self.stop = False
        self.update()

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        'Override Qt method.\n\n        Paint breakpoints icons.\n        '
        super(DebuggerPanel, self).paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.editor.sideareas_color)
        self.paint_cell(painter)
        for (top, line_number, block) in self.editor.visible_blocks:
            if self.line_number_hint == line_number:
                self._draw_breakpoint_icon(top, painter, 'transparent')
            if self._current_line_arrow == line_number and (not self.stop):
                self._draw_breakpoint_icon(top, painter, 'arrow')
            data = block.userData()
            if data is None or not data.breakpoint:
                continue
            if data.breakpoint_condition is None:
                self._draw_breakpoint_icon(top, painter, 'breakpoint')
            else:
                self._draw_breakpoint_icon(top, painter, 'condition')

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override Qt method\n\n        Add/remove breakpoints by single click.\n        '
        line_number = self.editor.get_linenumber_from_mouse_event(event)
        shift = event.modifiers() & Qt.ShiftModifier
        self.breakpoints_manager.toogle_breakpoint(line_number, edit_condition=shift)

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override Qt method.\n\n        Draw semitransparent breakpoint hint.\n        '
        self.line_number_hint = self.editor.get_linenumber_from_mouse_event(event)
        self.update()

    def leaveEvent(self, event):
        if False:
            print('Hello World!')
        'Override Qt method.\n\n        Remove semitransparent breakpoint hint\n        '
        self.line_number_hint = None
        self.update()

    def wheelEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.\n\n        Needed for scroll down the editor when scrolling over the panel.\n        '
        self.editor.wheelEvent(event)

    def on_state_changed(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Change visibility and connect/disconnect signal.\n\n        Args:\n            state (bool): Activate/deactivate.\n        '
        if state:
            self.breakpoints_manager.sig_repaint_breakpoints.connect(self.repaint)
        else:
            self.breakpoints_manager.sig_repaint_breakpoints.disconnect(self.repaint)