"""
Created on 2023/02/23
@author: Irony
@site: https://pyqt.site https://github.com/PyQt5
@email: 892768447@qq.com
@file: CallInThread.py
@description:
"""
import time
from datetime import datetime
from threading import Thread
from PyQt5.QtCore import Q_ARG, Q_RETURN_ARG, QMetaObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QTextBrowser, QWidget

class ThreadQt(QThread):

    def __init__(self, textBrowser, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ThreadQt, self).__init__(*args, **kwargs)
        self._textBrowser = textBrowser

    def stop(self):
        if False:
            while True:
                i = 10
        self.requestInterruption()

    def run(self):
        if False:
            i = 10
            return i + 15
        while not self.isInterruptionRequested():
            text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            retValue = QMetaObject.invokeMethod(self.parent(), 'isReadOnly1', Qt.DirectConnection, Q_RETURN_ARG(bool))
            argValue = Q_ARG(str, text + ' readOnly: ' + str(retValue))
            QMetaObject.invokeMethod(self._textBrowser, 'append', Qt.QueuedConnection, argValue)
            self.sleep(1)

class ThreadPy(Thread):

    def __init__(self, textBrowser, parent, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ThreadPy, self).__init__(*args, **kwargs)
        self._running = True
        self._textBrowser = textBrowser
        self._parent = parent

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._running = False

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while self._running:
            text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            QMetaObject.invokeMethod(self._parent, 'appendText', Qt.QueuedConnection, Q_ARG(str, text + ' from Signal'))
            QMetaObject.invokeMethod(self._textBrowser, 'append', Qt.QueuedConnection, Q_ARG(str, text + ' to Slot'))
            time.sleep(1)

class Window(QWidget):
    appendText = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.textBrowser1 = QTextBrowser(self)
        self.textBrowser2 = QTextBrowser(self)
        layout.addWidget(self.textBrowser1)
        layout.addWidget(self.textBrowser2)
        self.appendText.connect(self.textBrowser2.append)
        self.thread1 = ThreadQt(self.textBrowser1, self)
        self.thread1.start()
        self.thread2 = ThreadPy(self.textBrowser2, self)
        self.thread2.start()

    @pyqtSlot(result=bool)
    def isReadOnly1(self):
        if False:
            i = 10
            return i + 15
        return self.textBrowser1.isReadOnly()

    def closeEvent(self, event):
        if False:
            return 10
        self.thread1.stop()
        self.thread2.stop()
        self.thread1.wait()
        self.thread2.join()
        super(Window, self).closeEvent(event)
if __name__ == '__main__':
    import cgitb
    import sys
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())