from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QProgressBar
from tribler.gui.utilities import connect

class TimeoutProgressBar(QProgressBar):
    timeout = pyqtSignal()

    def __init__(self, parent=None, timeout=10000):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.timeout_interval = timeout
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.setInterval(100)
        connect(self.timer.timeout, self._update)
        self.setMaximum(self.timeout_interval)

    def _update(self):
        if False:
            for i in range(10):
                print('nop')
        self.setValue(self.value() + self.timer.interval())
        if self.value() >= self.maximum():
            self.timer.stop()
            self.timeout.emit()

    def start(self):
        if False:
            i = 10
            return i + 15
        self.setValue(0)
        self.timer.start()

    def stop(self):
        if False:
            i = 10
            return i + 15
        self.setValue(0)
        self.timer.stop()