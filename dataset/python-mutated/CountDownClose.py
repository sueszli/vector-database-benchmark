"""
Created on 2018年6月22日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: MessageBox
@description: 
"""
from random import randrange
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton
except ImportError:
    from PySide2.QtCore import QTimer
    from PySide2.QtWidgets import QApplication, QMessageBox, QPushButton

class MessageBox(QMessageBox):

    def __init__(self, *args, count=5, time=1000, auto=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MessageBox, self).__init__(*args, **kwargs)
        self._count = count
        self._time = time
        self._auto = auto
        assert count > 0
        assert time >= 500
        self.setStandardButtons(self.Close)
        self.closeBtn = self.button(self.Close)
        self.closeBtn.setText('关闭(%s)' % count)
        self.closeBtn.setEnabled(False)
        self._timer = QTimer(self, timeout=self.doCountDown)
        self._timer.start(self._time)
        print('是否自动关闭', auto)

    def doCountDown(self):
        if False:
            while True:
                i = 10
        self.closeBtn.setText('关闭(%s)' % self._count)
        self._count -= 1
        if self._count <= 0:
            self.closeBtn.setText('关闭')
            self.closeBtn.setEnabled(True)
            self._timer.stop()
            if self._auto:
                self.accept()
                self.close()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = QPushButton('点击弹出对话框')
    w.resize(200, 200)
    w.show()
    w.clicked.connect(lambda : MessageBox(w, text='倒计时关闭对话框', auto=randrange(0, 2)).exec_())
    sys.exit(app.exec_())