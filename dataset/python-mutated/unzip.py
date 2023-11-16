from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from MainWindow import Ui_MainWindow
import os
import types
import random
import sys
import traceback
import zipfile
PROGRESS_ON = '\nQLabel {\n    background-color: rgb(233,30,99);\n    border: 2px solid rgb(194,24,91);\n    color: rgb(136,14,79);\n}\n'
PROGRESS_OFF = '\nQLabel {\n    color: rgba(0,0,0,0);\n}\n'
EXCLUDE_PATHS = ['__MACOSX/']

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    progress = pyqtSignal(float)

class UnzipWorker(QRunnable):
    """
    Worker thread for unzipping.
    """
    signals = WorkerSignals()

    def __init__(self, path):
        if False:
            while True:
                i = 10
        super(UnzipWorker, self).__init__()
        os.chdir(os.path.dirname(path))
        self.zipfile = zipfile.ZipFile(path)

    @pyqtSlot()
    def run(self):
        if False:
            return 10
        try:
            items = self.zipfile.infolist()
            total_n = len(items)
            for (n, item) in enumerate(items, 1):
                if not any((item.filename.startswith(p) for p in EXCLUDE_PATHS)):
                    self.zipfile.extract(item)
                self.signals.progress.emit(n / total_n)
        except Exception as e:
            (exctype, value) = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            return
        self.signals.finished.emit()

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAcceptDrops(True)
        self.prev_pos = None
        self.threadpool = QThreadPool()
        self.head.raise_()

        def patch_mousePressEvent(self_, e):
            if False:
                return 10
            if e.button() == Qt.LeftButton and self.worker is not None:
                self_.current_rotation = random.randint(-15, +15)
                self_.current_y = 30
                self.update()
                self.threadpool.start(self.worker)
                self.worker = None
            elif e.button() == Qt.RightButton:
                pass

        def patch_paintEvent(self, event):
            if False:
                while True:
                    i = 10
            p = QPainter(self)
            rect = event.rect()
            transform = QTransform()
            transform.translate(rect.width() / 2, rect.height() / 2)
            transform.rotate(self.current_rotation)
            transform.translate(-rect.width() / 2, -rect.height() / 2)
            p.setTransform(transform)
            prect = self.pixmap().rect()
            rect.adjust((rect.width() - prect.width()) / 2, self.current_y + (rect.height() - prect.height()) / 2, -(rect.width() - prect.width()) / 2, self.current_y + -(rect.height() - prect.height()) / 2)
            p.drawPixmap(rect, self.pixmap())
        self.head.mousePressEvent = types.MethodType(patch_mousePressEvent, self.head)
        self.head.paintEvent = types.MethodType(patch_paintEvent, self.head)
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_triggered)
        self.timer.start(5)
        self.head.current_rotation = 0
        self.head.current_y = 0
        self.head.locked = True
        self.worker = None
        self.update_progress(1)
        self.show()

    def timer_triggered(self):
        if False:
            print('Hello World!')
        if self.head.current_y > 0:
            self.head.current_y -= 1
        if self.head.current_rotation > 0:
            self.head.current_rotation -= 1
        elif self.head.current_rotation < 0:
            self.head.current_rotation += 1
        self.head.update()
        if self.head.current_y == 0 and self.head.current_rotation == 0:
            self.head.locked = False

    def dragEnterEvent(self, e):
        if False:
            i = 10
            return i + 15
        data = e.mimeData()
        if data.hasUrls():
            url = data.urls()[0].toLocalFile()
            if os.path.splitext(url)[1].lower() == '.zip':
                e.accept()

    def dropEvent(self, e):
        if False:
            return 10
        data = e.mimeData()
        path = data.urls()[0].toLocalFile()
        self.worker = UnzipWorker(path)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.finished.connect(self.unzip_finished)
        self.worker.signals.error.connect(self.unzip_error)
        self.update_progress(0)

    def mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.prev_pos = e.globalPos()

    def mouseMoveEvent(self, e):
        if False:
            return 10
        if self.prev_pos:
            delta = e.globalPos() - self.prev_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.prev_pos = e.globalPos()

    def update_progress(self, pc):
        if False:
            print('Hello World!')
        '\n        Accepts progress as float in\n        :param pc: float 0-1 of completion.\n        :return:\n        '
        current_n = int(pc * 10)
        for n in range(1, 11):
            getattr(self, 'progress_%d' % n).setStyleSheet(PROGRESS_ON if n > current_n else PROGRESS_OFF)

    def unzip_finished(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def unzip_error(self, err):
        if False:
            for i in range(10):
                print('nop')
        (exctype, value, traceback) = err
        self.update_progress(1)
        dlg = QMessageBox(self)
        dlg.setText(traceback)
        dlg.setIcon(QMessageBox.Critical)
        dlg.show()
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()