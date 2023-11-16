from PyQt5.QtCore import QAbstractListModel, Qt, QModelIndex
from PyQt5.QtGui import QColor
from urh import settings

class ParticipantLegendListModel(QAbstractListModel):

    def __init__(self, participants, parent=None):
        if False:
            i = 10
            return i + 15
        '\n\n        :type participants: list of Participant\n        '
        super().__init__(parent)
        self.participants = participants

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        if False:
            print('Hello World!')
        return len(self.participants) + 1

    def data(self, index: QModelIndex, role=None):
        if False:
            return 10
        row = index.row()
        if role == Qt.DisplayRole:
            if row == 0:
                return 'not assigned'
            else:
                try:
                    return str(self.participants[row - 1])
                except IndexError:
                    return None
        elif role == Qt.BackgroundColorRole:
            if row > 0:
                try:
                    return settings.PARTICIPANT_COLORS[self.participants[row - 1].color_index]
                except IndexError:
                    return None
        elif role == Qt.TextColorRole:
            if row > 0:
                try:
                    bgcolor = settings.PARTICIPANT_COLORS[self.participants[row - 1].color_index]
                    (red, green, blue) = (bgcolor.red(), bgcolor.green(), bgcolor.blue())
                    return QColor('black') if red * 0.299 + green * 0.587 + blue * 0.114 > 186 else QColor('white')
                except IndexError:
                    return None

    def flags(self, index):
        if False:
            return 10
        return Qt.ItemIsEnabled

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        self.beginResetModel()
        self.endResetModel()