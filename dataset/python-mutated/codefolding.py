"""
This module contains the marker panel.

Adapted from pyqode/core/panels/folding.py of the
`PyQode project <https://github.com/pyQode/pyQode>`_.
Original file:
<https://github.com/pyQode/pyqode.core/blob/master/pyqode/core/panels/folding.py>
"""
from math import ceil
import sys
from intervaltree import IntervalTree
from qtpy.QtCore import Signal, QSize, QPointF, QRectF, QRect, Qt
from qtpy.QtWidgets import QApplication, QStyleOptionViewItem, QStyle
from qtpy.QtGui import QTextBlock, QColor, QFontMetricsF, QPainter, QLinearGradient, QPen, QPalette, QResizeEvent, QCursor
from spyder.plugins.editor.panels.utils import FoldingRegion
from spyder.plugins.editor.api.decoration import TextDecoration, DRAW_ORDERS
from spyder.plugins.editor.api.panel import Panel
from spyder.plugins.editor.utils.editor import TextHelper, DelayJobRunner, drift_color
from spyder.utils.icon_manager import ima
from spyder.utils.palette import QStylePalette

class FoldingPanel(Panel):
    """
    Displays the document outline and lets the user collapse/expand blocks.

    The data represented by the panel come from the Language Server Protocol
    invoked via the CodeEditor. This panel stores information about both
    folding regions and their folding state.
    """
    trigger_state_changed = Signal(QTextBlock, bool)
    collapse_all_triggered = Signal()
    expand_all_triggered = Signal()

    @property
    def native_icons(self):
        if False:
            while True:
                i = 10
        '\n        Defines whether the panel will use native indicator icons or\n        use custom ones.\n\n        If you want to use custom indicator icons, you must first\n        set this flag to False.\n        '
        return self.native_icons

    @native_icons.setter
    def native_icons(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._native_icons = value
        if self.editor:
            for clone in self.editor.clones:
                try:
                    clone.modes.get(self.__class__).native_icons = value
                except KeyError:
                    pass

    @property
    def indicators_icons(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets/sets the icons for the fold indicators.\n\n        The list of indicators is interpreted as follow::\n\n            (COLLAPSED_OFF, COLLAPSED_ON, EXPANDED_OFF, EXPANDED_ON)\n\n        To use this property you must first set `native_icons` to False.\n\n        :returns: tuple(str, str, str, str)\n        '
        return self._indicators_icons

    @indicators_icons.setter
    def indicators_icons(self, value):
        if False:
            return 10
        if len(value) != 4:
            raise ValueError('The list of custom indicators must contains 4 strings')
        self._indicators_icons = value
        if self.editor:
            for clone in self.editor.clones:
                try:
                    clone.modes.get(self.__class__).indicators_icons = value
                except KeyError:
                    pass

    @property
    def highlight_caret_scope(self):
        if False:
            print('Hello World!')
        '\n        True to highlight the caret scope automatically.\n\n        (Similar to the ``Highlight blocks in Qt Creator``.\n\n        Default is False.\n        '
        return self._highlight_caret

    @highlight_caret_scope.setter
    def highlight_caret_scope(self, value):
        if False:
            print('Hello World!')
        if value != self._highlight_caret:
            self._highlight_caret = value
            if self.editor:
                if value:
                    self._block_nbr = -1
                    self.editor.cursorPositionChanged.connect(self._highlight_caret_scope)
                else:
                    self._block_nbr = -1
                    self.editor.cursorPositionChanged.disconnect(self._highlight_caret_scope)
                for clone in self.editor.clones:
                    try:
                        clone.modes.get(self.__class__).highlight_caret_scope = value
                    except KeyError:
                        pass

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        Panel.__init__(self)
        self._native_icons = False
        self._indicators_icons = ('folding.arrow_right_off', 'folding.arrow_right_on', 'folding.arrow_down_off', 'folding.arrow_down_on')
        self._block_nbr = -1
        self._highlight_caret = False
        self.highlight_caret_scope = False
        self._indic_size = 16
        self._scope_decos = []
        self._block_decos = {}
        self.setMouseTracking(True)
        self.scrollable = True
        self._mouse_over_line = None
        self._current_scope = None
        self._prev_cursor = None
        self.context_menu = None
        self.action_collapse = None
        self.action_expand = None
        self.action_collapse_all = None
        self.action_expand_all = None
        self._original_background = None
        self._display_folding = False
        self._key_pressed = False
        self._highlight_runner = DelayJobRunner(delay=250)
        self.current_tree = IntervalTree()
        self.root = FoldingRegion(None, None)
        self.folding_regions = {}
        self.folding_status = {}
        self.folding_levels = {}
        self.folding_nesting = {}

    def update_folding(self, folding_info):
        if False:
            print('Hello World!')
        'Update folding panel folding ranges.'
        if folding_info is None:
            return
        (self.current_tree, self.root, self.folding_regions, self.folding_nesting, self.folding_levels, self.folding_status) = folding_info
        self.update()

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        'Returns the widget size hint (based on the editor font size) '
        fm = QFontMetricsF(self.editor.font())
        size_hint = QSize(ceil(fm.height()), ceil(fm.height()))
        if size_hint.width() > 16:
            size_hint.setWidth(16)
        return size_hint

    def _draw_collapsed_indicator(self, line_number, top_position, block, painter, mouse_hover=False):
        if False:
            return 10
        if line_number in self.folding_regions:
            collapsed = self.folding_status[line_number]
            line_end = self.folding_regions[line_number]
            mouse_over = self._mouse_over_line == line_number
            if not mouse_hover:
                self._draw_fold_indicator(top_position, mouse_over, collapsed, painter)
            if collapsed:
                if mouse_hover:
                    self._draw_fold_indicator(top_position, mouse_over, collapsed, painter)
                for deco_line in self._block_decos:
                    deco = self._block_decos[deco_line]
                    if deco.block == block:
                        break
                else:
                    self._add_fold_decoration(block, line_end)
            elif not mouse_hover:
                for deco_line in list(self._block_decos.keys()):
                    deco = self._block_decos[deco_line]
                    if deco.block == block:
                        self._block_decos.pop(deco_line)
                        self.editor.decorations.remove(deco)
                        del deco
                        break

    def paintEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(FoldingPanel, self).paintEvent(event)
        painter = QPainter(self)
        self.paint_cell(painter)
        if not self._display_folding and (not self._key_pressed):
            if any(self.folding_status.values()):
                for info in self.editor.visible_blocks:
                    (top_position, line_number, block) = info
                    self._draw_collapsed_indicator(line_number, top_position, block, painter, mouse_hover=True)
            return
        if self._mouse_over_line is not None:
            block = self.editor.document().findBlockByNumber(self._mouse_over_line)
            try:
                self._draw_fold_region_background(block, painter)
            except (ValueError, KeyError):
                pass
        for (top_position, line_number, block) in self.editor.visible_blocks:
            self._draw_collapsed_indicator(line_number, top_position, block, painter, mouse_hover=False)

    def _draw_fold_region_background(self, block, painter):
        if False:
            return 10
        '\n        Draw the fold region when the mouse is over and non collapsed\n        indicator.\n\n        :param top: Top position\n        :param block: Current block.\n        :param painter: QPainter\n        '
        th = TextHelper(self.editor)
        start = block.blockNumber()
        end = self.folding_regions[start]
        if start > 0:
            top = th.line_pos_from_number(start)
        else:
            top = 0
        bottom = th.line_pos_from_number(end)
        h = bottom - top
        if h == 0:
            h = self.sizeHint().height()
        w = self.sizeHint().width()
        self._draw_rect(QRectF(0, top, w, h), painter)

    def _draw_rect(self, rect, painter):
        if False:
            while True:
                i = 10
        "\n        Draw the background rectangle using the current style primitive color.\n\n        :param rect: The fold zone rect to draw\n\n        :param painter: The widget's painter.\n        "
        c = self.editor.sideareas_color
        grad = QLinearGradient(rect.topLeft(), rect.topRight())
        if sys.platform == 'darwin':
            grad.setColorAt(0, c.lighter(100))
            grad.setColorAt(1, c.lighter(110))
            outline = c.darker(110)
        else:
            grad.setColorAt(0, c.lighter(110))
            grad.setColorAt(1, c.lighter(130))
            outline = c.darker(100)
        painter.fillRect(rect, grad)
        painter.setPen(QPen(outline))
        painter.drawLine(rect.topLeft() + QPointF(1, 0), rect.topRight() - QPointF(1, 0))
        painter.drawLine(rect.bottomLeft() + QPointF(1, 0), rect.bottomRight() - QPointF(1, 0))
        painter.drawLine(rect.topRight() + QPointF(0, 1), rect.bottomRight() - QPointF(0, 1))
        painter.drawLine(rect.topLeft() + QPointF(0, 1), rect.bottomLeft() - QPointF(0, 1))

    def _draw_fold_indicator(self, top, mouse_over, collapsed, painter):
        if False:
            i = 10
            return i + 15
        '\n        Draw the fold indicator/trigger (arrow).\n\n        :param top: Top position\n        :param mouse_over: Whether the mouse is over the indicator\n        :param collapsed: Whether the trigger is collapsed or not.\n        :param painter: QPainter\n        '
        rect = QRect(0, top, self.sizeHint().width(), self.sizeHint().height())
        if self._native_icons:
            opt = QStyleOptionViewItem()
            opt.rect = rect
            opt.state = QStyle.State_Active | QStyle.State_Item | QStyle.State_Children
            if not collapsed:
                opt.state |= QStyle.State_Open
            if mouse_over:
                opt.state |= QStyle.State_MouseOver | QStyle.State_Enabled | QStyle.State_Selected
                opt.palette.setBrush(QPalette.Window, self.palette().highlight())
            opt.rect.translate(-2, 0)
            self.style().drawPrimitive(QStyle.PE_IndicatorBranch, opt, painter, self)
        else:
            index = 0
            if not collapsed:
                index = 2
            if mouse_over:
                index += 1
            ima.icon(self._indicators_icons[index]).paint(painter, rect)

    def find_parent_scope(self, block):
        if False:
            i = 10
            return i + 15
        'Find parent scope, if the block is not a fold trigger.'
        block_line = block.blockNumber()
        if block_line not in self.folding_regions:
            for start_line in self.folding_regions:
                end_line = self.folding_regions[start_line]
                if end_line > block_line:
                    if start_line < block_line:
                        block = self.editor.document().findBlockByNumber(start_line)
                        break
        return block

    def _clear_scope_decos(self):
        if False:
            print('Hello World!')
        'Clear scope decorations (on the editor)'
        for deco in self._scope_decos:
            self.editor.decorations.remove(deco)
        self._scope_decos[:] = []

    def _get_scope_highlight_color(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the base scope highlight color (derivated from the editor\n        background)\n\n        For lighter themes will be a darker color,\n        and for darker ones will be a lighter color\n        '
        color = self.editor.sideareas_color
        if color.lightness() < 128:
            color = drift_color(color, 130)
        else:
            color = drift_color(color, 105)
        return color

    def _decorate_block(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a decoration and add it to the editor.\n\n        Args:\n            start (int) start line of the decoration\n            end (int) end line of the decoration\n        '
        color = self._get_scope_highlight_color()
        draw_order = DRAW_ORDERS.get('codefolding')
        d = TextDecoration(self.editor.document(), start_line=max(0, start - 1), end_line=end, draw_order=draw_order)
        d.set_background(color)
        d.set_full_width(True, clear=False)
        self.editor.decorations.add(d)
        self._scope_decos.append(d)

    def _highlight_block(self, block):
        if False:
            return 10
        '\n        Highlights the current fold scope.\n\n        :param block: Block that starts the current fold scope.\n        '
        block_line = block.blockNumber()
        end_line = self.folding_regions[block_line]
        scope = (block_line, end_line)
        if self._current_scope is None or self._current_scope != scope:
            self._current_scope = scope
            self._clear_scope_decos()
            (start, end) = scope
            if not self.folding_status[start]:
                self._decorate_block(start, end)

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detect mouser over indicator and highlight the current scope in the\n        editor (up and down decoration arround the foldable text when the mouse\n        is over an indicator).\n\n        :param event: event\n        '
        super(FoldingPanel, self).mouseMoveEvent(event)
        th = TextHelper(self.editor)
        line = th.line_nbr_from_position(event.pos().y())
        if line >= 0:
            block = self.editor.document().findBlockByNumber(line)
            block = self.find_parent_scope(block)
            line_number = block.blockNumber()
            if line_number in self.folding_regions:
                if self._mouse_over_line is None:
                    QApplication.setOverrideCursor(QCursor(Qt.PointingHandCursor))
                if self._mouse_over_line != block.blockNumber() and self._mouse_over_line is not None:
                    self._mouse_over_line = block.blockNumber()
                    try:
                        self._highlight_block(block)
                    except KeyError:
                        pass
                else:
                    self._mouse_over_line = block.blockNumber()
                    try:
                        self._highlight_runner.request_job(self._highlight_block, block)
                    except KeyError:
                        pass
                self._highight_block = block
            else:
                self._highlight_runner.cancel_requests()
                self._mouse_over_line = None
                QApplication.restoreOverrideCursor()
            self.repaint()

    def enterEvent(self, event):
        if False:
            i = 10
            return i + 15
        self._display_folding = True
        self.repaint()

    def leaveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes scope decorations and background from the editor and the panel\n        if highlight_caret_scope, else simply update the scope decorations to\n        match the caret scope.\n        '
        super(FoldingPanel, self).leaveEvent(event)
        QApplication.restoreOverrideCursor()
        self._highlight_runner.cancel_requests()
        if not self.highlight_caret_scope:
            self._clear_scope_decos()
            self._mouse_over_line = None
            self._current_scope = None
        else:
            self._block_nbr = -1
            self._highlight_caret_scope()
        self.editor.repaint()
        self._display_folding = False

    def _add_fold_decoration(self, block, end_line):
        if False:
            while True:
                i = 10
        '\n        Add fold decorations (boxes arround a folded block in the editor\n        widget).\n        '
        start_line = block.blockNumber()
        text = self.editor.get_text_region(start_line + 1, end_line)
        draw_order = DRAW_ORDERS.get('codefolding')
        deco = TextDecoration(block, draw_order=draw_order)
        deco.signals.clicked.connect(self._on_fold_deco_clicked)
        deco.tooltip = text
        deco.block = block
        deco.select_line()
        deco.set_outline(drift_color(self._get_scope_highlight_color(), 110))
        deco.set_background(self._get_scope_highlight_color())
        deco.set_foreground(QColor(QStylePalette.COLOR_TEXT_4))
        self._block_decos[start_line] = deco
        self.editor.decorations.add(deco)

    def _get_block_until_line(self, block, end_line):
        if False:
            for i in range(10):
                print('nop')
        while block.blockNumber() <= end_line and block.isValid():
            block.setVisible(False)
            block = block.next()
        return block

    def fold_region(self, block, start_line, end_line):
        if False:
            for i in range(10):
                print('nop')
        'Fold region spanned by *start_line* and *end_line*.'
        while block.blockNumber() < end_line and block.isValid():
            block.setVisible(False)
            block = block.next()
        return block

    def unfold_region(self, block, start_line, end_line):
        if False:
            for i in range(10):
                print('nop')
        'Unfold region spanned by *start_line* and *end_line*.'
        if start_line - 1 in self._block_decos:
            deco = self._block_decos[start_line - 1]
            self._block_decos.pop(start_line - 1)
            self.editor.decorations.remove(deco)
        while block.blockNumber() < end_line and block.isValid():
            current_line = block.blockNumber()
            block.setVisible(True)
            get_next = True
            if current_line in self.folding_regions and current_line != start_line:
                block_end = self.folding_regions[current_line]
                if self.folding_status[current_line]:
                    get_next = False
                    block = self._get_block_until_line(block, block_end - 1)
            if get_next:
                block = block.next()

    def toggle_fold_trigger(self, block):
        if False:
            return 10
        '\n        Toggle a fold trigger block (expand or collapse it).\n\n        :param block: The QTextBlock to expand/collapse\n        '
        start_line = block.blockNumber()
        if start_line not in self.folding_regions:
            return
        end_line = self.folding_regions[start_line]
        if self.folding_status[start_line]:
            self.unfold_region(block, start_line, end_line)
            self.folding_status[start_line] = False
            if self._mouse_over_line is not None:
                self._decorate_block(start_line, end_line)
        else:
            self.fold_region(block, start_line, end_line)
            self.folding_status[start_line] = True
            self._clear_scope_decos()
        self._refresh_editor_and_scrollbars()

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Folds/unfolds the pressed indicator if any.'
        if self._mouse_over_line is not None:
            block = self.editor.document().findBlockByNumber(self._mouse_over_line)
            self.toggle_fold_trigger(block)

    def _on_fold_deco_clicked(self, deco):
        if False:
            while True:
                i = 10
        'Unfold a folded block that has just been clicked by the user'
        self.toggle_fold_trigger(deco.block)

    def on_state_changed(self, state):
        if False:
            while True:
                i = 10
        '\n        On state changed we (dis)connect to the cursorPositionChanged signal\n        '
        if state:
            self.editor.sig_key_pressed.connect(self._on_key_pressed)
            if self._highlight_caret:
                self.editor.cursorPositionChanged.connect(self._highlight_caret_scope)
                self._block_nbr = -1
            self.editor.new_text_set.connect(self._clear_block_deco)
        else:
            self.editor.sig_key_pressed.disconnect(self._on_key_pressed)
            if self._highlight_caret:
                self.editor.cursorPositionChanged.disconnect(self._highlight_caret_scope)
                self._block_nbr = -1
            self.editor.new_text_set.disconnect(self._clear_block_deco)

    def _on_key_pressed(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override key press to select the current scope if the user wants\n        to deleted a folded scope (without selecting it).\n        '
        delete_request = event.key() in {Qt.Key_Delete, Qt.Key_Backspace}
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            if event.key() == Qt.Key_Return:
                delete_request = True
        if event.text() or delete_request:
            self._key_pressed = True
            if cursor.hasSelection():
                positions_to_check = (cursor.selectionStart(), cursor.selectionEnd())
            else:
                positions_to_check = (cursor.position(),)
            for pos in positions_to_check:
                block = self.editor.document().findBlock(pos)
                start_line = block.blockNumber() + 2
                if start_line in self.folding_regions and self.folding_status[start_line]:
                    end_line = self.folding_regions[start_line]
                    if delete_request and cursor.hasSelection():
                        tc = TextHelper(self.editor).select_lines(start_line, end_line)
                        if tc.selectionStart() > cursor.selectionStart():
                            start = cursor.selectionStart()
                        else:
                            start = tc.selectionStart()
                        if tc.selectionEnd() < cursor.selectionEnd():
                            end = cursor.selectionEnd()
                        else:
                            end = tc.selectionEnd()
                        tc.setPosition(start)
                        tc.setPosition(end, tc.KeepAnchor)
                        self.editor.setTextCursor(tc)
            self._key_pressed = False

    def _refresh_editor_and_scrollbars(self):
        if False:
            i = 10
            return i + 15
        "\n        Refrehes editor content and scollbars.\n\n        We generate a fake resize event to refresh scroll bar.\n\n        We have the same problem as described here:\n        http://www.qtcentre.org/threads/44803 and we apply the same solution\n        (don't worry, there is no visual effect, the editor does not grow up\n        at all, even with a value = 500)\n        "
        TextHelper(self.editor).mark_whole_doc_dirty()
        self.editor.repaint()
        s = self.editor.size()
        s.setWidth(s.width() + 1)
        self.editor.resizeEvent(QResizeEvent(self.editor.size(), s))

    def collapse_all(self):
        if False:
            return 10
        '\n        Collapses all triggers and makes all blocks with fold level > 0\n        invisible.\n        '
        self._clear_block_deco()
        block = self.editor.document().firstBlock()
        while block.isValid():
            line_number = block.blockNumber()
            if line_number in self.folding_regions:
                end_line = self.folding_regions[line_number]
                self.fold_region(block, line_number, end_line)
            block = block.next()
        self._refresh_editor_and_scrollbars()
        tc = self.editor.textCursor()
        tc.movePosition(tc.Start)
        self.editor.setTextCursor(tc)
        self.collapse_all_triggered.emit()

    def _clear_block_deco(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear the folded block decorations.'
        for deco_line in self._block_decos:
            deco = self._block_decos[deco_line]
            self.editor.decorations.remove(deco)
        self._block_decos = {}

    def expand_all(self):
        if False:
            return 10
        'Expands all fold triggers.'
        block = self.editor.document().firstBlock()
        while block.isValid():
            line_number = block.BlockNumber()
            if line_number in self.folding_regions:
                end_line = self.folding_regions[line_number]
                self.unfold_region(block, line_number, end_line)
            block = block.next()
        self._clear_block_deco()
        self._refresh_editor_and_scrollbars()
        self.expand_all_triggered.emit()

    def _highlight_caret_scope(self):
        if False:
            return 10
        '\n        Highlight the scope of the current caret position.\n\n        This get called only if :attr:`\n        spyder.widgets.panels.FoldingPanel.highlight_care_scope` is True.\n        '
        cursor = self.editor.textCursor()
        block_nbr = cursor.blockNumber()
        if self._block_nbr != block_nbr:
            block = self.find_parent_scope(cursor.block())
            line_number = block.blockNumber()
            if line_number in self.folding_regions:
                self._mouse_over_line = block.blockNumber()
                try:
                    self._highlight_block(block)
                except KeyError:
                    pass
            else:
                self._clear_scope_decos()
        self._block_nbr = block_nbr