from collections import defaultdict
from PyQt5.QtCore import pyqtSignal, QModelIndex, Qt
from urh import settings
from urh.models.TableModel import TableModel
from urh.signalprocessing.ProtocolAnalyzer import ProtocolAnalyzer
from urh.ui.actions.DeleteBitsAndPauses import DeleteBitsAndPauses

class ProtocolTableModel(TableModel):
    ref_index_changed = pyqtSignal(int)

    def __init__(self, proto_analyzer: ProtocolAnalyzer, participants, controller, parent=None):
        if False:
            print('Hello World!')
        super().__init__(participants=participants, parent=parent)
        self.controller = controller
        self.protocol = proto_analyzer
        self.active_group_ids = [0]

    @property
    def diff_columns(self) -> defaultdict(set):
        if False:
            print('Hello World!')
        return self._diffs

    @property
    def refindex(self):
        if False:
            return 10
        return self._refindex

    @refindex.setter
    def refindex(self, refindex):
        if False:
            for i in range(10):
                print('nop')
        if refindex != self._refindex:
            self._refindex = refindex
            self.update()
            self.ref_index_changed.emit(self._refindex)

    def refresh_fonts(self):
        if False:
            i = 10
            return i + 15
        self.bold_fonts.clear()
        self.text_colors.clear()
        for i in self._diffs.keys():
            for j in self._diffs[i]:
                self.bold_fonts[i, j] = True
                self.text_colors[i, j] = settings.DIFFERENCE_CELL_COLOR
        if self._refindex >= 0:
            for j in range(self.col_count):
                self.text_colors[self._refindex, j] = settings.SELECTED_ROW_COLOR

    def delete_range(self, min_row: int, max_row: int, start: int, end: int):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_writeable:
            return
        del_action = DeleteBitsAndPauses(proto_analyzer=self.protocol, start_message=min_row, end_message=max_row, start=start, end=end, view=self.proto_view, decoded=True, subprotos=self.controller.protocol_list, update_label_ranges=False)
        self.undo_stack.push(del_action)

    def flags(self, index: QModelIndex):
        if False:
            i = 10
            return i + 15
        if index.isValid():
            alignment_offset = self.get_alignment_offset_at(index.row())
            if index.column() < alignment_offset:
                return Qt.ItemIsSelectable | Qt.ItemIsEnabled
            if self.is_writeable:
                return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        else:
            return Qt.NoItemFlags