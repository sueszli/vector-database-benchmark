import html
from qtpy.QtCore import Qt
from qtpy import QtWidgets
from .escapable_qlist_widget import EscapableQListWidget

class UniqueLabelQListWidget(EscapableQListWidget):

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        if False:
            return 10
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item

    def createItemFromLabel(self, label):
        if False:
            print('Hello World!')
        if self.findItemByLabel(label):
            raise ValueError("Item for label '{}' already exists".format(label))
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None):
        if False:
            print('Hello World!')
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText('{}'.format(label))
        else:
            qlabel.setText('{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(html.escape(label), *color))
        qlabel.setAlignment(Qt.AlignBottom)
        item.setSizeHint(qlabel.sizeHint())
        self.setItemWidget(item, qlabel)