"""
Created on 2018年9月日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: MetroCircleProgress
@description: 
"""
try:
    from PyQt5.QtCore import QSequentialAnimationGroup, QPauseAnimation, QPropertyAnimation, QParallelAnimationGroup, QObject, QSize, Qt, QRectF, pyqtSignal, pyqtProperty
    from PyQt5.QtGui import QPainter, QColor
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
except ImportError:
    from PySide2.QtCore import QSequentialAnimationGroup, QPauseAnimation, QPropertyAnimation, QParallelAnimationGroup, QObject, QSize, Qt, QRectF, Signal as pyqtSignal, Property as pyqtProperty
    from PySide2.QtGui import QPainter, QColor
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout

class CircleItem(QObject):
    X = 0
    Opacity = 1
    valueChanged = pyqtSignal()

    @pyqtProperty(float)
    def x(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.X

    @x.setter
    def x(self, x: float):
        if False:
            return 10
        self.X = x
        self.valueChanged.emit()

    @pyqtProperty(float)
    def opacity(self) -> float:
        if False:
            while True:
                i = 10
        return self.Opacity

    @opacity.setter
    def opacity(self, opacity: float):
        if False:
            while True:
                i = 10
        self.Opacity = opacity

def qBound(miv, cv, mxv):
    if False:
        print('Hello World!')
    return max(min(cv, mxv), miv)

class MetroCircleProgress(QWidget):
    Radius = 5
    Color = QColor(24, 189, 155)
    BackgroundColor = QColor(Qt.transparent)

    def __init__(self, *args, radius=5, color=QColor(24, 189, 155), backgroundColor=QColor(Qt.transparent), **kwargs):
        if False:
            return 10
        super(MetroCircleProgress, self).__init__(*args, **kwargs)
        self.Radius = radius
        self.Color = color
        self.BackgroundColor = backgroundColor
        self._items = []
        self._initAnimations()

    @pyqtProperty(int)
    def radius(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.Radius

    @radius.setter
    def radius(self, radius: int):
        if False:
            for i in range(10):
                print('nop')
        if self.Radius != radius:
            self.Radius = radius
            self.update()

    @pyqtProperty(QColor)
    def color(self) -> QColor:
        if False:
            return 10
        return self.Color

    @color.setter
    def color(self, color: QColor):
        if False:
            i = 10
            return i + 15
        if self.Color != color:
            self.Color = color
            self.update()

    @pyqtProperty(QColor)
    def backgroundColor(self) -> QColor:
        if False:
            print('Hello World!')
        return self.BackgroundColor

    @backgroundColor.setter
    def backgroundColor(self, backgroundColor: QColor):
        if False:
            while True:
                i = 10
        if self.BackgroundColor != backgroundColor:
            self.BackgroundColor = backgroundColor
            self.update()

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        super(MetroCircleProgress, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.BackgroundColor)
        painter.setPen(Qt.NoPen)
        for (item, _) in self._items:
            painter.save()
            color = self.Color.toRgb()
            color.setAlphaF(item.opacity)
            painter.setBrush(color)
            radius = qBound(self.Radius, self.Radius / 200 * self.height(), 2 * self.Radius)
            diameter = 2 * radius
            painter.drawRoundedRect(QRectF(item.x / 100 * self.width() - diameter, (self.height() - radius) / 2, diameter, diameter), radius, radius)
            painter.restore()

    def _initAnimations(self):
        if False:
            print('Hello World!')
        for index in range(5):
            item = CircleItem(self)
            item.valueChanged.connect(self.update)
            seqAnimation = QSequentialAnimationGroup(self)
            seqAnimation.setLoopCount(-1)
            self._items.append((item, seqAnimation))
            seqAnimation.addAnimation(QPauseAnimation(150 * index, self))
            parAnimation1 = QParallelAnimationGroup(self)
            parAnimation1.addAnimation(QPropertyAnimation(item, b'opacity', self, duration=400, startValue=0, endValue=1.0))
            parAnimation1.addAnimation(QPropertyAnimation(item, b'x', self, duration=400, startValue=0, endValue=25.0))
            seqAnimation.addAnimation(parAnimation1)
            seqAnimation.addAnimation(QPropertyAnimation(item, b'x', self, duration=2000, startValue=25.0, endValue=75.0))
            parAnimation2 = QParallelAnimationGroup(self)
            parAnimation2.addAnimation(QPropertyAnimation(item, b'opacity', self, duration=400, startValue=1.0, endValue=0))
            parAnimation2.addAnimation(QPropertyAnimation(item, b'x', self, duration=400, startValue=75.0, endValue=100.0))
            seqAnimation.addAnimation(parAnimation2)
            seqAnimation.addAnimation(QPauseAnimation((5 - index - 1) * 150, self))
        for (_, animation) in self._items:
            animation.start()

    def sizeHint(self):
        if False:
            return 10
        return QSize(100, self.Radius * 2)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(MetroCircleProgress(self))
        layout.addWidget(MetroCircleProgress(self, radius=10))
        layout.addWidget(MetroCircleProgress(self, styleSheet='\n            qproperty-color: rgb(255, 0, 0);\n        '))
        layout.addWidget(MetroCircleProgress(self, styleSheet='\n            qproperty-color: rgb(0, 0, 255);\n            qproperty-backgroundColor: rgba(180, 180, 180, 180);\n        '))
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())