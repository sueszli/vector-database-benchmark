"""
Created on 2018年11月22日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: RlatticeEffect
@description: 
"""
from random import random
from time import time
try:
    from PyQt5.QtCore import QPropertyAnimation, QObject, QEasingCurve, Qt, QRectF, pyqtSignal, pyqtProperty
    from PyQt5.QtGui import QColor, QPainterPath, QPainter
    from PyQt5.QtWidgets import QApplication, QWidget
except ImportError:
    from PySide2.QtCore import QPropertyAnimation, QObject, QEasingCurve, Qt, QRectF, Signal as pyqtSignal, Property as pyqtProperty
    from PySide2.QtGui import QColor, QPainterPath, QPainter
    from PySide2.QtWidgets import QApplication, QWidget
try:
    from Lib import pointtool
    getDistance = pointtool.getDistance
    findClose = pointtool.findClose
except:
    import math

    def getDistance(p1, p2):
        if False:
            for i in range(10):
                print('nop')
        return math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2)

    def findClose(points):
        if False:
            for i in range(10):
                print('nop')
        plen = len(points)
        for i in range(plen):
            closest = [None, None, None, None, None]
            p1 = points[i]
            for j in range(plen):
                p2 = points[j]
                dte1 = getDistance(p1, p2)
                if p1 != p2:
                    placed = False
                    for k in range(5):
                        if not placed:
                            if not closest[k]:
                                closest[k] = p2
                                placed = True
                    for k in range(5):
                        if not placed:
                            if dte1 < getDistance(p1, closest[k]):
                                closest[k] = p2
                                placed = True
            p1.closest = closest

class Target:

    def __init__(self, x, y):
        if False:
            print('Hello World!')
        self.x = x
        self.y = y

class Point(QObject):
    valueChanged = pyqtSignal(int)

    def __init__(self, x, ox, y, oy, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Point, self).__init__(*args, **kwargs)
        self.__x = x
        self._x = x
        self.originX = ox
        self._y = y
        self.__y = y
        self.originY = oy
        self.closest = [0, 0, 0, 0, 0]
        self.radius = 2 + random() * 2
        self.lineColor = QColor(156, 217, 249)
        self.circleColor = QColor(156, 217, 249)

    def initAnimation(self):
        if False:
            print('Hello World!')
        if not hasattr(self, 'xanimation'):
            self.xanimation = QPropertyAnimation(self, b'x', self, easingCurve=QEasingCurve.InOutSine)
            self.xanimation.valueChanged.connect(self.valueChanged.emit)
            self.yanimation = QPropertyAnimation(self, b'y', self, easingCurve=QEasingCurve.InOutSine)
            self.yanimation.valueChanged.connect(self.valueChanged.emit)
            self.yanimation.finished.connect(self.updateAnimation)
            self.updateAnimation()

    def updateAnimation(self):
        if False:
            for i in range(10):
                print('nop')
        self.xanimation.stop()
        self.yanimation.stop()
        duration = (1 + random()) * 1000
        self.xanimation.setDuration(duration)
        self.yanimation.setDuration(duration)
        self.xanimation.setStartValue(self.__x)
        self.xanimation.setEndValue(self.originX - 50 + random() * 100)
        self.yanimation.setStartValue(self.__y)
        self.yanimation.setEndValue(self.originY - 50 + random() * 100)
        self.xanimation.start()
        self.yanimation.start()

    @pyqtProperty(float)
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        return self._x

    @x.setter
    def x(self, x):
        if False:
            while True:
                i = 10
        self._x = x

    @pyqtProperty(float)
    def y(self):
        if False:
            i = 10
            return i + 15
        return self._y

    @y.setter
    def y(self, y):
        if False:
            for i in range(10):
                print('nop')
        self._y = y

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.resize(800, 600)
        self.points = []
        self.target = Target(self.width() / 2, self.height() / 2)
        self.initPoints()

    def update(self, *args):
        if False:
            i = 10
            return i + 15
        super(Window, self).update()

    def paintEvent(self, event):
        if False:
            return 10
        super(Window, self).paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)
        self.animate(painter)
        painter.end()

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).mouseMoveEvent(event)
        self.target.x = event.x()
        self.target.y = event.y()
        self.update()

    def initPoints(self):
        if False:
            for i in range(10):
                print('nop')
        t = time()
        self.points.clear()
        stepX = self.width() / 20
        stepY = self.height() / 20
        for x in range(0, self.width(), int(stepX)):
            for y in range(0, self.height(), int(stepY)):
                ox = x + random() * stepX
                oy = y + random() * stepY
                point = Point(ox, ox, oy, oy)
                point.valueChanged.connect(self.update)
                self.points.append(point)
        print(time() - t)
        t = time()
        findClose(self.points)
        print(time() - t)

    def animate(self, painter):
        if False:
            print('Hello World!')
        for p in self.points:
            value = abs(getDistance(self.target, p))
            if value < 4000:
                p.lineColor.setAlphaF(0.3)
                p.circleColor.setAlphaF(0.6)
            elif value < 20000:
                p.lineColor.setAlphaF(0.1)
                p.circleColor.setAlphaF(0.3)
            elif value < 40000:
                p.lineColor.setAlphaF(0.02)
                p.circleColor.setAlphaF(0.1)
            else:
                p.lineColor.setAlphaF(0)
                p.circleColor.setAlphaF(0)
            if p.lineColor.alpha():
                for pc in p.closest:
                    if not pc:
                        continue
                    path = QPainterPath()
                    path.moveTo(p.x, p.y)
                    path.lineTo(pc.x, pc.y)
                    painter.save()
                    painter.setPen(p.lineColor)
                    painter.drawPath(path)
                    painter.restore()
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(p.circleColor)
            painter.drawRoundedRect(QRectF(p.x - p.radius, p.y - p.radius, 2 * p.radius, 2 * p.radius), p.radius, p.radius)
            painter.restore()
            p.initAnimation()
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())