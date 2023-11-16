"""
Created on 2018年1月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: BubbleTips
@description: 
"""
import sys
try:
    from PyQt5.QtCore import QRectF, Qt, QPropertyAnimation, pyqtProperty, QPoint, QParallelAnimationGroup, QEasingCurve
    from PyQt5.QtGui import QPainter, QPainterPath, QColor, QPen
    from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QApplication, QLineEdit, QPushButton
except ImportError:
    from PySide2.QtCore import QRectF, Qt, QPropertyAnimation, Property as pyqtProperty, QPoint, QParallelAnimationGroup, QEasingCurve
    from PySide2.QtGui import QPainter, QPainterPath, QColor, QPen
    from PySide2.QtWidgets import QLabel, QWidget, QVBoxLayout, QApplication, QLineEdit, QPushButton

class BubbleLabel(QWidget):
    BackgroundColor = QColor(195, 195, 195)
    BorderColor = QColor(150, 150, 150)

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        text = kwargs.pop('text', '')
        super(BubbleLabel, self).__init__(*args, **kwargs)
        self.setWindowFlags(Qt.Window | Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint)
        self.setMinimumWidth(200)
        self.setMinimumHeight(48)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 16)
        self.label = QLabel(self)
        layout.addWidget(self.label)
        self.setText(text)
        self._desktop = QApplication.instance().desktop()

    def setText(self, text):
        if False:
            i = 10
            return i + 15
        self.label.setText(text)

    def text(self):
        if False:
            i = 10
            return i + 15
        return self.label.text()

    def stop(self):
        if False:
            return 10
        self.hide()
        self.animationGroup.stop()
        self.close()

    def show(self):
        if False:
            i = 10
            return i + 15
        super(BubbleLabel, self).show()
        startPos = QPoint(self._desktop.screenGeometry().width() - self.width() - 100, self._desktop.availableGeometry().height() - self.height())
        endPos = QPoint(self._desktop.screenGeometry().width() - self.width() - 100, self._desktop.availableGeometry().height() - self.height() * 3 - 5)
        print(startPos, endPos)
        self.move(startPos)
        self.initAnimation(startPos, endPos)

    def initAnimation(self, startPos, endPos):
        if False:
            i = 10
            return i + 15
        opacityAnimation = QPropertyAnimation(self, b'opacity')
        opacityAnimation.setStartValue(1.0)
        opacityAnimation.setEndValue(0.0)
        opacityAnimation.setEasingCurve(QEasingCurve.InQuad)
        opacityAnimation.setDuration(4000)
        moveAnimation = QPropertyAnimation(self, b'pos')
        moveAnimation.setStartValue(startPos)
        moveAnimation.setEndValue(endPos)
        moveAnimation.setEasingCurve(QEasingCurve.InQuad)
        moveAnimation.setDuration(5000)
        self.animationGroup = QParallelAnimationGroup(self)
        self.animationGroup.addAnimation(opacityAnimation)
        self.animationGroup.addAnimation(moveAnimation)
        self.animationGroup.finished.connect(self.close)
        self.animationGroup.start()

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(BubbleLabel, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rectPath = QPainterPath()
        triPath = QPainterPath()
        height = self.height() - 8
        rectPath.addRoundedRect(QRectF(0, 0, self.width(), height), 5, 5)
        x = self.width() / 5 * 4
        triPath.moveTo(x, height)
        triPath.lineTo(x + 6, height + 8)
        triPath.lineTo(x + 12, height)
        rectPath.addPath(triPath)
        painter.setPen(QPen(self.BorderColor, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(self.BackgroundColor)
        painter.drawPath(rectPath)
        painter.setPen(QPen(self.BackgroundColor, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(x, height, x + 12, height)

    def windowOpacity(self):
        if False:
            i = 10
            return i + 15
        return super(BubbleLabel, self).windowOpacity()

    def setWindowOpacity(self, opacity):
        if False:
            for i in range(10):
                print('nop')
        super(BubbleLabel, self).setWindowOpacity(opacity)
    opacity = pyqtProperty(float, windowOpacity, setWindowOpacity)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.msgEdit = QLineEdit(self, returnPressed=self.onMsgShow)
        self.msgButton = QPushButton('显示内容', self, clicked=self.onMsgShow)
        layout.addWidget(self.msgEdit)
        layout.addWidget(self.msgButton)

    def onMsgShow(self):
        if False:
            i = 10
            return i + 15
        msg = self.msgEdit.text().strip()
        if not msg:
            return
        if hasattr(self, '_blabel'):
            self._blabel.stop()
            self._blabel.deleteLater()
            del self._blabel
        self._blabel = BubbleLabel()
        self._blabel.setText(msg)
        self._blabel.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())