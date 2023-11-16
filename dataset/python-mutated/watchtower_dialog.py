import enum
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from electrum.i18n import _
from .util import Buttons
from .my_treeview import MyTreeView

class WatcherList(MyTreeView):

    class Columns(MyTreeView.BaseColumnsEnum):
        OUTPOINT = enum.auto()
        TX_COUNT = enum.auto()
        STATUS = enum.auto()
    headers = {Columns.OUTPOINT: _('Outpoint'), Columns.TX_COUNT: _('Tx'), Columns.STATUS: _('Status')}

    def __init__(self, parent: 'WatchtowerDialog'):
        if False:
            print('Hello World!')
        super().__init__(parent=parent, stretch_column=self.Columns.OUTPOINT)
        self.parent = parent
        self.setModel(QStandardItemModel(self))
        self.setSortingEnabled(True)
        self.update()

    def update(self):
        if False:
            while True:
                i = 10
        if self.parent.lnwatcher is None:
            return
        self.model().clear()
        self.update_headers(self.__class__.headers)
        lnwatcher = self.parent.lnwatcher
        l = lnwatcher.list_sweep_tx()
        for outpoint in l:
            n = lnwatcher.get_num_tx(outpoint)
            status = lnwatcher.get_channel_status(outpoint)
            labels = [''] * len(self.Columns)
            labels[self.Columns.OUTPOINT] = outpoint
            labels[self.Columns.TX_COUNT] = str(n)
            labels[self.Columns.STATUS] = status
            items = [QStandardItem(e) for e in labels]
            self.set_editability(items)
            self.model().insertRow(self.model().rowCount(), items)
        size = lnwatcher.sweepstore.filesize()
        self.parent.size_label.setText('Database size: %.2f Mb' % (size / 1024 / 1024.0))

class WatchtowerDialog(QDialog):

    def __init__(self, gui_object):
        if False:
            i = 10
            return i + 15
        QDialog.__init__(self)
        self.gui_object = gui_object
        self.config = gui_object.config
        self.network = gui_object.daemon.network
        assert self.network
        self.lnwatcher = self.network.local_watchtower
        self.setWindowTitle(_('Watchtower'))
        self.setMinimumSize(600, 200)
        self.size_label = QLabel()
        self.watcher_list = WatcherList(self)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.size_label)
        vbox.addWidget(self.watcher_list)
        b = QPushButton(_('Close'))
        b.clicked.connect(self.close)
        vbox.addLayout(Buttons(b))
        self.watcher_list.update()

    def is_hidden(self):
        if False:
            for i in range(10):
                print('nop')
        return self.isMinimized() or self.isHidden()

    def show_or_hide(self):
        if False:
            while True:
                i = 10
        if self.is_hidden():
            self.bring_to_top()
        else:
            self.hide()

    def bring_to_top(self):
        if False:
            return 10
        self.show()
        self.raise_()

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        self.gui_object.watchtower_dialog = None
        event.accept()