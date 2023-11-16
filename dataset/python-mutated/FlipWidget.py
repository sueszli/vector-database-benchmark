"""
Created on 2019年5月15日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FlipWidget
@description: 动画翻转窗口
"""
try:
    from PyQt5.QtCore import pyqtSignal, pyqtProperty, Qt, QPropertyAnimation, QEasingCurve, QPointF
    from PyQt5.QtGui import QPainter, QTransform
    from PyQt5.QtWidgets import QWidget
except ImportError:
    from PySide2.QtCore import Signal as pyqtSignal, Property as pyqtProperty, Qt, QPropertyAnimation, QEasingCurve, QPointF
    from PySide2.QtGui import QPainter, QTransform
    from PySide2.QtWidgets import QWidget

class FlipWidget(QWidget):
    Left = 0
    Right = 1
    Scale = 3
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(FlipWidget, self).__init__(*args, **kwargs)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._angle = 0
        self._animation = QPropertyAnimation(self, b'angle', self)
        self._animation.setDuration(550)
        self._animation.setEasingCurve(QEasingCurve.OutInQuad)
        self._animation.finished.connect(self.finished.emit)

    @pyqtProperty(int)
    def angle(self):
        if False:
            i = 10
            return i + 15
        return self._angle

    @angle.setter
    def angle(self, angle):
        if False:
            for i in range(10):
                print('nop')
        self._angle = angle
        self.update()

    def updateImages(self, direction, image1, image2):
        if False:
            while True:
                i = 10
        '设置两张切换图\n        :param direction:        方向\n        :param image1:           图片1\n        :param image2:           图片2\n        '
        self.image1 = image1
        self.image2 = image2
        self.show()
        self._angle = 0
        if direction == self.Right:
            self._animation.setStartValue(1)
            self._animation.setEndValue(-180)
        elif direction == self.Left:
            self._animation.setStartValue(1)
            self._animation.setEndValue(180)
        self._animation.start()

    def paintEvent(self, event):
        if False:
            return 10
        super(FlipWidget, self).paintEvent(event)
        if hasattr(self, 'image1') and hasattr(self, 'image2') and self.isVisible():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            transform = QTransform()
            transform.translate(self.width() / 2, self.height() / 2)
            if self._angle >= -90 and self._angle <= 90:
                painter.save()
                transform.rotate(self._angle, Qt.YAxis)
                painter.setTransform(transform)
                width = self.image1.width() / 2
                height = int(self.image1.height() * (1 - abs(self._angle / self.Scale) / 100))
                image = self.image1.scaled(self.image1.width(), height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(QPointF(-width, -height / 2), image)
                painter.restore()
            else:
                painter.save()
                if self._angle > 0:
                    angle = 180 + self._angle
                else:
                    angle = self._angle - 180
                transform.rotate(angle, Qt.YAxis)
                painter.setTransform(transform)
                width = self.image2.width() / 2
                height = int(self.image2.height() * (1 - (360 - abs(angle)) / self.Scale / 100))
                image = self.image2.scaled(self.image2.width(), height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(QPointF(-width, -height / 2), image)
                painter.restore()