"""
Created on 2023/02/01
@author: Irony
@site: https://pyqt.site https://github.com/PyQt5
@email: 892768447@qq.com
@file: GetCmdResult.py
@description:
"""
import sys
try:
    from PyQt5.QtCore import QProcess
    from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget
except ImportError:
    from PySide2.QtCore import QProcess
    from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        self.setWindowTitle('执行命令得到结果')
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('点击执行 ping www.baidu.com', self))
        self.buttonRunSync = QPushButton('同步执行', self)
        layout.addWidget(self.buttonRunSync)
        self.buttonRunSync.clicked.connect(self.run_ping)
        self.buttonRunASync = QPushButton('异步执行', self)
        layout.addWidget(self.buttonRunASync)
        self.buttonRunASync.clicked.connect(self.run_ping)
        self.resultView = QTextBrowser(self)
        layout.addWidget(self.resultView)
        self._pingProcess = None

    def run_ping(self):
        if False:
            while True:
                i = 10
        sender = self.sender()
        self.buttonRunSync.setEnabled(False)
        self.buttonRunASync.setEnabled(False)
        if self._pingProcess:
            self._pingProcess.terminate()
        self._pingProcess = QProcess(self)
        self._pingProcess.setProgram('ping')
        if sys.platform.startswith('win'):
            self._pingProcess.setArguments(['-n', '5', 'www.baidu.com'])
            self._pingProcess.setArguments(['-n', '5', 'www.baidu.com'])
        elif sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
            self._pingProcess.setArguments(['-c', '5', 'www.baidu.com'])
        self._pingProcess.setProcessChannelMode(QProcess.MergedChannels)
        self._pingProcess.started.connect(self.on_started)
        if sender == self.buttonRunASync:
            self._pingProcess.finished.connect(self.on_finished)
            self._pingProcess.errorOccurred.connect(self.on_error)
            self._pingProcess.start()
        elif sender == self.buttonRunSync:
            self._pingProcess.start()
            if self._pingProcess.waitForFinished():
                self.on_finished(self._pingProcess.exitCode(), self._pingProcess.exitStatus())
            else:
                self.resultView.append('ping process read timeout')
                self.on_error(self._pingProcess.error())

    def on_started(self):
        if False:
            while True:
                i = 10
        self.resultView.append('ping process started')

    def on_finished(self, exitCode, exitStatus):
        if False:
            for i in range(10):
                print('nop')
        self.resultView.append('ping process finished, exitCode: %s, exitStatus: %s' % (exitCode, exitStatus))
        result = self._pingProcess.readAll().data()
        try:
            import chardet
            encoding = chardet.detect(result)
            self.resultView.append(result.decode(encoding['encoding']))
        except Exception:
            self.resultView.append(result.decode('utf-8', errors='ignore'))
        self._pingProcess.kill()
        self._pingProcess = None
        self.buttonRunSync.setEnabled(True)
        self.buttonRunASync.setEnabled(True)

    def on_error(self, error):
        if False:
            for i in range(10):
                print('nop')
        self.resultView.append('ping process error: %s, message: %s' % (error, self._pingProcess.errorString()))
        self._pingProcess.kill()
        self._pingProcess = None
        self.buttonRunSync.setEnabled(True)
        self.buttonRunASync.setEnabled(True)
if __name__ == '__main__':
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())