"""
Created on 2018年3月9日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: InheritQThread
@description: 继承QThread
"""
try:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton
except ImportError:
    from PySide2.QtCore import QThread, Signal as pyqtSignal
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton

class Worker(QThread):
    valueChanged = pyqtSignal(int)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print('thread id', QThread.currentThread())
        for i in range(1, 101):
            if self.isInterruptionRequested():
                break
            print('value', i)
            self.valueChanged.emit(i)
            QThread.sleep(1)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 100)
        layout.addWidget(self.progressBar)
        layout.addWidget(QPushButton('开启线程', self, clicked=self.onStart))
        print('main id', QThread.currentThread())
        self._thread = Worker(self)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.valueChanged.connect(self.progressBar.setValue)

    def onStart(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._thread.isRunning():
            print('main id', QThread.currentThread())
            self._thread.start()

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        if self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait()
        self._thread.deleteLater()
        super(Window, self).closeEvent(event)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())