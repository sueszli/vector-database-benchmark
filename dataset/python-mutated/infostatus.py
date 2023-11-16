import time
from PyQt6 import QtCore, QtGui, QtWidgets
from picard.util import icontheme
from picard.util.time import get_timestamp
from picard.ui.ui_infostatus import Ui_InfoStatus

class InfoStatus(QtWidgets.QWidget, Ui_InfoStatus):

    def __init__(self, parent):
        if False:
            print('Hello World!')
        QtWidgets.QWidget.__init__(self, parent)
        Ui_InfoStatus.__init__(self)
        self.setupUi(self)
        self._size = QtCore.QSize(16, 16)
        self._create_icons()
        self._init_labels()
        self.reset_counters()

    def _init_labels(self):
        if False:
            return 10
        size = self._size
        self.label1.setPixmap(self.icon_eta.pixmap(size))
        self.label1.hide()
        self.label2.setPixmap(self.icon_file.pixmap(size))
        self.label3.setPixmap(self.icon_cd.pixmap(size))
        self.label4.setPixmap(self.icon_file_pending.pixmap(size))
        self.label5.setPixmap(self.icon_download.pixmap(size, QtGui.QIcon.Mode.Disabled))
        self._init_tooltips()

    def _create_icons(self):
        if False:
            return 10
        self.icon_eta = QtGui.QIcon(':/images/22x22/hourglass.png')
        self.icon_cd = icontheme.lookup('media-optical')
        self.icon_file = QtGui.QIcon(':/images/file.png')
        self.icon_file_pending = QtGui.QIcon(':/images/file-pending.png')
        self.icon_download = QtGui.QIcon(':/images/16x16/action-go-down-16.png')

    def _init_tooltips(self):
        if False:
            i = 10
            return i + 15
        t1 = _('Estimated Time')
        t2 = _('Files')
        t3 = _('Albums')
        t4 = _('Pending files')
        t5 = _('Pending requests')
        self.val1.setToolTip(t1)
        self.label1.setToolTip(t1)
        self.val2.setToolTip(t2)
        self.label2.setToolTip(t2)
        self.val3.setToolTip(t3)
        self.label3.setToolTip(t3)
        self.val4.setToolTip(t4)
        self.label4.setToolTip(t4)
        self.val5.setToolTip(t5)
        self.label5.setToolTip(t5)

    def update(self, files=0, albums=0, pending_files=0, pending_requests=0, progress=0):
        if False:
            i = 10
            return i + 15
        self.set_files(files)
        self.set_albums(albums)
        self.set_pending_files(pending_files)
        self.set_pending_requests(pending_requests)
        total_pending = pending_files + pending_requests
        last_pending = self._last_pending_files + self._last_pending_requests
        if total_pending > 0 and last_pending == 0:
            self.reset_counters()
        previous_done_files = max(0, self._max_pending_files - self._last_pending_files)
        previous_done_requests = max(0, self._max_pending_requests - self._last_pending_requests)
        self._max_pending_files = max(self._max_pending_files, previous_done_files + pending_files)
        self._max_pending_requests = max(self._max_pending_requests, previous_done_requests + pending_requests)
        self._last_pending_files = pending_files
        self._last_pending_requests = pending_requests
        if total_pending == 0 or self._max_pending_files + self._max_pending_requests <= 1:
            self.reset_counters()
            self.hide_eta()
            return
        if total_pending != last_pending:
            current_time = time.time()
            diff_time = max(0.1, current_time - self._prev_time)
            previous_done_files = max(1, previous_done_files)
            file_eta_seconds = diff_time / previous_done_files * pending_files + pending_requests
            network_eta_seconds = pending_requests + previous_done_requests / previous_done_files * pending_files
            eta_seconds = max(network_eta_seconds, file_eta_seconds)
            self._last_progress = diff_time / (diff_time + eta_seconds)
            self.set_eta(eta_seconds)

    def reset_counters(self):
        if False:
            while True:
                i = 10
        self._last_progress = 0
        self._max_pending_requests = 0
        self._last_pending_requests = 0
        self._max_pending_files = 0
        self._last_pending_files = 0
        self._prev_time = time.time()

    def get_progress(self):
        if False:
            while True:
                i = 10
        return self._last_progress

    def set_eta(self, eta_seconds):
        if False:
            return 10
        if eta_seconds > 0:
            self.val1.setText(get_timestamp(eta_seconds))
            self.val1.show()
            self.label1.show()
        else:
            self.hide_eta()

    def hide_eta(self):
        if False:
            print('Hello World!')
        self.val1.hide()
        self.label1.hide()

    def set_files(self, num):
        if False:
            return 10
        self.val2.setText(str(num))

    def set_albums(self, num):
        if False:
            for i in range(10):
                print('nop')
        self.val3.setText(str(num))

    def set_pending_files(self, num):
        if False:
            while True:
                i = 10
        self.val4.setText(str(num))

    def set_pending_requests(self, num):
        if False:
            print('Hello World!')
        if num <= 0:
            enabled = QtGui.QIcon.Mode.Disabled
        else:
            enabled = QtGui.QIcon.Mode.Normal
        self.label5.setPixmap(self.icon_download.pixmap(self._size, enabled))
        self.val5.setText(str(num))