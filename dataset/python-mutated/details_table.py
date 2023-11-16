from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtWidgets import QHeaderView, QTableView
from PyQt5.QtGui import QFont, QBrush
from hscommon.trans import trget
tr = trget('ui')
HEADER = [tr('Selected'), tr('Reference')]

class DetailsModel(QAbstractTableModel):

    def __init__(self, model, app, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.model = model
        self.prefs = app.prefs

    def columnCount(self, parent):
        if False:
            i = 10
            return i + 15
        return len(HEADER)

    def data(self, index, role):
        if False:
            while True:
                i = 10
        if not index.isValid():
            return None
        column = index.column() + 1
        row = index.row()
        ignored_fields = ['Dupe Count']
        if self.model.row(row)[0] in ignored_fields or self.model.row(row)[1] == '---' or self.model.row(row)[2] == '---':
            if role != Qt.DisplayRole:
                return None
            return self.model.row(row)[column]
        if role == Qt.DisplayRole:
            return self.model.row(row)[column]
        if role == Qt.ForegroundRole and self.model.row(row)[1] != self.model.row(row)[2]:
            return QBrush(self.prefs.details_table_delta_foreground_color)
        if role == Qt.FontRole and self.model.row(row)[1] != self.model.row(row)[2]:
            font = QFont(self.model.view.font())
            font.setBold(True)
            return font
        return None

    def headerData(self, section, orientation, role):
        if False:
            while True:
                i = 10
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and (section < len(HEADER)):
            return HEADER[section]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole and (section < self.model.row_count()):
            return self.model.row(section)[0]
        return None

    def rowCount(self, parent):
        if False:
            print('Hello World!')
        return self.model.row_count()

class DetailsTable(QTableView):

    def __init__(self, *args):
        if False:
            return 10
        QTableView.__init__(self, *args)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.NoSelection)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)

    def setModel(self, model):
        if False:
            for i in range(10):
                print('nop')
        QTableView.setModel(self, model)
        hheader = self.horizontalHeader()
        hheader.setHighlightSections(False)
        hheader.setSectionResizeMode(0, QHeaderView.Stretch)
        hheader.setSectionResizeMode(1, QHeaderView.Stretch)
        vheader = self.verticalHeader()
        vheader.setVisible(True)
        vheader.setDefaultSectionSize(18)
        vheader.setSectionResizeMode(QHeaderView.Fixed)
        vheader.setSectionsMovable(True)