"""
Created on 2020/7/1
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: TaskbarProgress
@description: 
"""
import cgitb
import sys
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QSpinBox, QPushButton, QLabel
    from PyQt5.QtWinExtras import QWinTaskbarButton
except ImportError:
    from PySide2.QtCore import QTimer
    from PySide2.QtWidgets import QWidget, QApplication, QGridLayout, QSpinBox, QPushButton, QLabel
    from PySide2.QtWinExtras import QWinTaskbarButton

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        self.taskButton = QWinTaskbarButton(self)
        self.taskProgress = self.taskButton.progress()
        self.timerProgress = QTimer(self)
        self.timerProgress.timeout.connect(self.update_progress)
        self.setup_ui()

    def showEvent(self, event):
        if False:
            print('Hello World!')
        super(Window, self).showEvent(event)
        if not self.taskButton.window():
            self.taskButton.setWindow(self.windowHandle())
            self.taskProgress.show()

    def closeEvent(self, event):
        if False:
            return 10
        self.timerProgress.stop()
        super(Window, self).closeEvent(event)

    def setup_ui(self):
        if False:
            return 10
        layout = QGridLayout(self)
        self.spinBoxMin = QSpinBox(self)
        self.spinBoxMax = QSpinBox(self)
        self.spinBoxMax.setMaximum(100)
        self.spinBoxMax.setValue(100)
        layout.addWidget(self.spinBoxMin, 0, 0)
        layout.addWidget(self.spinBoxMax, 0, 1)
        layout.addWidget(QPushButton('设置范围值', self, clicked=self.set_range), 0, 2)
        self.spinBoxCur = QSpinBox(self)
        self.spinBoxCur.setMaximum(100)
        self.spinBoxCur.setValue(50)
        layout.addWidget(self.spinBoxCur, 0, 3)
        layout.addWidget(QPushButton('设置当前值', self, clicked=self.set_current_value), 0, 4)
        layout.addWidget(QPushButton('隐藏', self, clicked=self.set_show_hide), 1, 0)
        layout.addWidget(QPushButton('暂停', self, clicked=self.set_pause_resume), 1, 1)
        layout.addWidget(QPushButton('重置', self, clicked=self.set_reset), 1, 2)
        layout.addWidget(QPushButton('停止', self, clicked=self.set_stop), 1, 3)
        layout.addWidget(QPushButton('不可见', self, clicked=self.set_visible), 1, 4)
        layout.addWidget(QPushButton('模拟进度动画', self, clicked=self.start_progress), 2, 0, 1, 5)
        layout.addWidget(QLabel('暂停信号\u3000：', self), 3, 0)
        self.labelPause = QLabel(self)
        layout.addWidget(self.labelPause, 3, 1)
        self.taskProgress.pausedChanged.connect(lambda v: self.labelPause.setText(str(v)))
        layout.addWidget(QLabel('停止信号\u3000：', self), 4, 0)
        self.labelStop = QLabel(self)
        layout.addWidget(self.labelStop, 4, 1)
        self.taskProgress.stoppedChanged.connect(lambda v: self.labelStop.setText(str(v)))
        layout.addWidget(QLabel('值改变信号：', self), 5, 0)
        self.labelValue = QLabel(self)
        layout.addWidget(self.labelValue, 5, 1)
        self.taskProgress.valueChanged.connect(lambda v: self.labelValue.setText(str(v)))
        layout.addWidget(QLabel('可见度信号：', self), 6, 0)
        self.labelVisible = QLabel(self)
        layout.addWidget(self.labelVisible, 6, 1)
        self.taskProgress.visibilityChanged.connect(lambda v: self.labelVisible.setText(str(v)))

    def set_range(self):
        if False:
            while True:
                i = 10
        vmin = min(self.spinBoxMin.value(), self.spinBoxMax.value())
        vmax = max(self.spinBoxMin.value(), self.spinBoxMax.value())
        self.taskProgress.setRange(vmin, vmax)

    def set_current_value(self):
        if False:
            i = 10
            return i + 15
        self.taskProgress.setValue(self.spinBoxCur.value())

    def set_show_hide(self):
        if False:
            print('Hello World!')
        visible = self.taskProgress.isVisible()
        if visible:
            self.taskProgress.hide()
            self.sender().setText('显示')
        else:
            self.taskProgress.show()
            self.sender().setText('隐藏')

    def set_pause_resume(self):
        if False:
            for i in range(10):
                print('nop')
        paused = self.taskProgress.isPaused()
        if paused:
            self.taskProgress.resume()
            self.timerProgress.start(100)
            self.sender().setText('暂停')
        else:
            self.taskProgress.pause()
            self.timerProgress.stop()
            self.sender().setText('恢复')

    def set_reset(self):
        if False:
            return 10
        self.taskProgress.reset()
        paused = self.taskProgress.isPaused()
        if not paused:
            self.timerProgress.stop()
            self.timerProgress.start(100)

    def set_stop(self):
        if False:
            print('Hello World!')
        self.timerProgress.stop()
        self.taskProgress.stop()
        self.setEnabled(False)

    def set_visible(self):
        if False:
            while True:
                i = 10
        visible = self.taskProgress.isVisible()
        self.taskProgress.setVisible(not visible)
        self.sender().setText('可见' if visible else '不可见')

    def start_progress(self):
        if False:
            i = 10
            return i + 15
        self.timerProgress.start(100)
        self.sender().setEnabled(False)

    def update_progress(self):
        if False:
            i = 10
            return i + 15
        value = self.taskProgress.value()
        value += 1
        if value > self.taskProgress.maximum():
            value = 0
        self.taskProgress.setValue(value)
if __name__ == '__main__':
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())