from PyQt5.QtCore import Qt, QRect, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDragMoveEvent, QDragEnterEvent, QPainter, QBrush, QColor, QPen, QDropEvent, QDragLeaveEvent, QContextMenuEvent, QIcon
from PyQt5.QtWidgets import QActionGroup, QInputDialog
from PyQt5.QtWidgets import QHeaderView, QAbstractItemView, QStyleOption, QMenu
from urh.models.GeneratorTableModel import GeneratorTableModel
from urh.ui.views.TableView import TableView

class GeneratorTableView(TableView):
    encodings_updated = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.drop_indicator_rect = QRect()
        self.drag_active = False
        self.show_pause_active = False
        self.pause_row = -1

    def model(self) -> GeneratorTableModel:
        if False:
            for i in range(10):
                print('nop')
        return super().model()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if False:
            i = 10
            return i + 15
        event.acceptProposedAction()
        self.drag_active = True

    def dragMoveEvent(self, event: QDragMoveEvent):
        if False:
            print('Hello World!')
        pos = event.pos()
        row = self.rowAt(pos.y())
        index = self.model().createIndex(row, 0)
        rect = self.visualRect(index)
        rect_left = self.visualRect(index.sibling(index.row(), 0))
        rect_right = self.visualRect(index.sibling(index.row(), self.horizontalHeader().logicalIndex(self.model().columnCount() - 1)))
        self.drop_indicator_position = self.position(event.pos(), rect)
        if self.drop_indicator_position == self.AboveItem:
            self.drop_indicator_rect = QRect(rect_left.left(), rect_left.top(), rect_right.right() - rect_left.left(), 0)
            event.accept()
        elif self.drop_indicator_position == self.BelowItem:
            self.drop_indicator_rect = QRect(rect_left.left(), rect_left.bottom(), rect_right.right() - rect_left.left(), 0)
            event.accept()
        elif self.drop_indicator_position == self.OnItem:
            self.drop_indicator_rect = QRect(rect_left.left(), rect_left.bottom(), rect_right.right() - rect_left.left(), 0)
            event.accept()
        else:
            self.drop_indicator_rect = QRect()
        self.viewport().update()

    def __rect_for_row(self, row):
        if False:
            return 10
        index = self.model().createIndex(row, 0)
        rect_left = self.visualRect(index.sibling(index.row(), 0))
        rect_right = self.visualRect(index.sibling(index.row(), self.horizontalHeader().logicalIndex(self.model().columnCount() - 1)))
        return QRect(rect_left.left(), rect_left.bottom(), rect_right.right() - rect_left.left(), 0)

    def dropEvent(self, event: QDropEvent):
        if False:
            print('Hello World!')
        self.drag_active = False
        row = self.rowAt(event.pos().y())
        index = self.model().createIndex(row, 0)
        rect = self.visualRect(index)
        drop_indicator_position = self.position(event.pos(), rect)
        if row == -1:
            row = self.model().row_count - 1
        elif drop_indicator_position == self.BelowItem or drop_indicator_position == self.OnItem:
            row += 1
        self.model().dropped_row = row
        super().dropEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        if False:
            return 10
        self.drag_active = False
        self.viewport().update()
        super().dragLeaveEvent(event)

    @staticmethod
    def position(pos, rect):
        if False:
            print('Hello World!')
        r = QAbstractItemView.OnViewport
        margin = 5
        if pos.y() - rect.top() < margin:
            r = QAbstractItemView.AboveItem
        elif rect.bottom() - pos.y() < margin:
            r = QAbstractItemView.BelowItem
        elif pos.y() - rect.top() > margin and rect.bottom() - pos.y() > margin:
            r = QAbstractItemView.OnItem
        return r

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        self.paint_drop_indicator(painter)
        self.paint_pause_indicator(painter)

    def paint_drop_indicator(self, painter):
        if False:
            return 10
        if self.drag_active:
            opt = QStyleOption()
            opt.initFrom(self)
            opt.rect = self.drop_indicator_rect
            rect = opt.rect
            brush = QBrush(QColor(Qt.darkRed))
            if rect.height() == 0:
                pen = QPen(brush, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(rect.topLeft(), rect.topRight())
            else:
                pen = QPen(brush, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(rect)

    def paint_pause_indicator(self, painter):
        if False:
            print('Hello World!')
        if self.show_pause_active:
            rect = self.__rect_for_row(self.pause_row)
            brush = QBrush(QColor(Qt.darkGreen))
            pen = QPen(brush, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(rect.topLeft(), rect.topRight())

    def create_context_menu(self) -> QMenu:
        if False:
            print('Hello World!')
        menu = super().create_context_menu()
        add_message_action = menu.addAction('Add empty message...')
        add_message_action.setIcon(QIcon.fromTheme('edit-table-insert-row-below'))
        add_message_action.triggered.connect(self.on_add_message_action_triggered)
        if not self.selection_is_empty:
            menu.addAction(self.copy_action)
        if self.model().row_count > 0:
            duplicate_action = menu.addAction('Duplicate selected lines')
            duplicate_action.setIcon(QIcon.fromTheme('edit-table-insert-row-under'))
            duplicate_action.triggered.connect(self.on_duplicate_action_triggered)
            self._add_insert_column_menu(menu)
            menu.addSeparator()
            clear_action = menu.addAction('Clear table')
            clear_action.triggered.connect(self.on_clear_action_triggered)
            clear_action.setIcon(QIcon.fromTheme('edit-clear'))
        self.encoding_actions = {}
        if not self.selection_is_empty:
            selected_encoding = self.model().protocol.messages[self.selected_rows[0]].decoder
            for i in self.selected_rows:
                if self.model().protocol.messages[i].decoder != selected_encoding:
                    selected_encoding = None
                    break
            menu.addSeparator()
            encoding_group = QActionGroup(self)
            encoding_menu = menu.addMenu('Enforce encoding')
            for decoding in self.model().decodings:
                ea = encoding_menu.addAction(decoding.name)
                ea.setCheckable(True)
                ea.setActionGroup(encoding_group)
                if selected_encoding == decoding:
                    ea.setChecked(True)
                self.encoding_actions[ea] = decoding
                ea.triggered.connect(self.on_encoding_action_triggered)
            menu.addSeparator()
            de_bruijn_action = menu.addAction('Generate De Bruijn Sequence from Selection')
            de_bruijn_action.triggered.connect(self.on_de_bruijn_action_triggered)
        return menu

    @pyqtSlot()
    def on_duplicate_action_triggered(self):
        if False:
            print('Hello World!')
        self.model().duplicate_rows(self.selected_rows)

    @pyqtSlot()
    def on_clear_action_triggered(self):
        if False:
            while True:
                i = 10
        self.model().clear()

    @pyqtSlot()
    def on_encoding_action_triggered(self):
        if False:
            for i in range(10):
                print('nop')
        for row in self.selected_rows:
            self.model().protocol.messages[row].decoder = self.encoding_actions[self.sender()]
        self.encodings_updated.emit()

    @pyqtSlot()
    def on_de_bruijn_action_triggered(self):
        if False:
            print('Hello World!')
        self.setCursor(Qt.WaitCursor)
        row = self.rowAt(self.context_menu_pos.y())
        (_, _, start, end) = self.selection_range()
        self.model().generate_de_bruijn(row, start, end)
        self.unsetCursor()

    @pyqtSlot()
    def on_add_message_action_triggered(self):
        if False:
            return 10
        row = self.rowAt(self.context_menu_pos.y())
        (num_bits, ok) = QInputDialog.getInt(self, self.tr('How many bits shall the new message have?'), self.tr('Number of bits:'), 42, 1)
        if ok:
            self.model().add_empty_row_behind(row, num_bits)