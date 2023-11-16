"""
Created on 2019年3月19日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CircleLine
@description: 
"""
from math import floor, pi, cos, sin
from random import random, randint
from time import time
try:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import QWidget, QApplication
except ImportError:
    from PySide2.QtCore import QTimer, Qt
    from PySide2.QtGui import QColor, QPainter, QPainterPath, QPen
    from PySide2.QtWidgets import QWidget, QApplication
radMin = 10
radMax = 80
filledCircle = 30
concentricCircle = 60
radThreshold = 25
speedMin = 0.3
speedMax = 0.6
maxOpacity = 0.6
colors = [QColor(52, 168, 83), QColor(117, 95, 147), QColor(199, 108, 23), QColor(194, 62, 55), QColor(0, 172, 212), QColor(120, 120, 120)]
circleBorder = 10
backgroundLine = colors[0]
backgroundColor = QColor(38, 43, 46)
backgroundMlt = 0.85
lineBorder = 2.5
maxCircles = 8
points = []
circleExp = 1
circleExpMax = 1.003
circleExpMin = 0.997
circleExpSp = 4e-05
circlePulse = False

def randint(a, b):
    if False:
        while True:
            i = 10
    return floor(random() * (b - a + 1) + a)

def randRange(a, b):
    if False:
        return 10
    return random() * (b - a) + a

def hyperRange(a, b):
    if False:
        while True:
            i = 10
    return random() * random() * random() * (b - a) + a

class Circle:

    def __init__(self, background, width, height):
        if False:
            return 10
        self.background = background
        self.x = randRange(-width / 2, width / 2)
        self.y = randRange(-height / 2, height / 2)
        self.radius = hyperRange(radMin, radMax)
        self.filled = (False if randint(0, 100) > concentricCircle else 'full') if self.radius < radThreshold else False if randint(0, 100) > concentricCircle else 'concentric'
        self.color = colors[randint(0, len(colors) - 1)]
        self.borderColor = colors[randint(0, len(colors) - 1)]
        self.opacity = 0.05
        self.speed = randRange(speedMin, speedMax)
        self.speedAngle = random() * 2 * pi
        self.speedx = cos(self.speedAngle) * self.speed
        self.speedy = sin(self.speedAngle) * self.speed
        spacex = abs((self.x - (-1 if self.speedx < 0 else 1) * (width / 2 + self.radius)) / self.speedx)
        spacey = abs((self.y - (-1 if self.speedy < 0 else 1) * (height / 2 + self.radius)) / self.speedy)
        self.ttl = min(spacex, spacey)

class CircleLineWindow(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(CircleLineWindow, self).__init__(*args, **kwargs)
        palette = self.palette()
        palette.setColor(palette.Background, backgroundColor)
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        geometry = QApplication.instance().desktop().availableGeometry()
        self.screenWidth = geometry.width()
        self.screenHeight = geometry.height()
        self._canDraw = True
        self._firstDraw = True
        self._timer = QTimer(self, timeout=self.update)
        self.init()

    def init(self):
        if False:
            while True:
                i = 10
        points.clear()
        self.linkDist = min(self.screenWidth, self.screenHeight) / 2.4
        for _ in range(maxCircles * 3):
            points.append(Circle('', self.screenWidth, self.screenHeight))
        self.update()

    def showEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(CircleLineWindow, self).showEvent(event)
        self._canDraw = True

    def hideEvent(self, event):
        if False:
            while True:
                i = 10
        super(CircleLineWindow, self).hideEvent(event)
        self._canDraw = False

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        super(CircleLineWindow, self).paintEvent(event)
        if not self._canDraw:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        self.draw(painter)

    def draw(self, painter):
        if False:
            return 10
        if circlePulse:
            if circleExp < circleExpMin or circleExp > circleExpMax:
                circleExpSp *= -1
            circleExp += circleExpSp
        painter.translate(self.screenWidth / 2, self.screenHeight / 2)
        if self._firstDraw:
            t = time()
        self.renderPoints(painter, points)
        if self._firstDraw:
            self._firstDraw = False
            t = (time() - t) * 1000 * 2
            t = int(min(2.4, self.screenHeight / self.height()) * t) - 1
            t = t if t > 15 else 15
            print('start timer(%d msec)' % t)
            self._timer.start(t)

    def drawCircle(self, painter, circle):
        if False:
            print('Hello World!')
        if circle.background:
            circle.radius *= circleExp
        else:
            circle.radius /= circleExp
        radius = circle.radius
        r = radius * circleExp
        c = QColor(circle.borderColor)
        c.setAlphaF(circle.opacity)
        painter.save()
        if circle.filled == 'full':
            painter.setBrush(c)
            painter.setPen(Qt.NoPen)
        else:
            painter.setPen(QPen(c, max(1, circleBorder * (radMin - circle.radius) / (radMin - radMax))))
        painter.drawEllipse(circle.x - r, circle.y - r, 2 * r, 2 * r)
        painter.restore()
        if circle.filled == 'concentric':
            r = radius / 2
            painter.save()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(c, max(1, circleBorder * (radMin - circle.radius) / (radMin - radMax))))
            painter.drawEllipse(circle.x - r, circle.y - r, 2 * r, 2 * r)
            painter.restore()
        circle.x += circle.speedx
        circle.y += circle.speedy
        if circle.opacity < maxOpacity:
            circle.opacity += 0.01
        circle.ttl -= 1

    def renderPoints(self, painter, circles):
        if False:
            print('Hello World!')
        for (i, circle) in enumerate(circles):
            if circle.ttl < -20:
                circle = Circle('', self.screenWidth, self.screenHeight)
                circles[i] = circle
            self.drawCircle(painter, circle)
        circles_len = len(circles)
        for i in range(circles_len - 1):
            for j in range(i + 1, circles_len):
                deltax = circles[i].x - circles[j].x
                deltay = circles[i].y - circles[j].y
                dist = pow(pow(deltax, 2) + pow(deltay, 2), 0.5)
                if dist <= circles[i].radius + circles[j].radius:
                    continue
                if dist < self.linkDist:
                    xi = (1 if circles[i].x < circles[j].x else -1) * abs(circles[i].radius * deltax / dist)
                    yi = (1 if circles[i].y < circles[j].y else -1) * abs(circles[i].radius * deltay / dist)
                    xj = (-1 if circles[i].x < circles[j].x else 1) * abs(circles[j].radius * deltax / dist)
                    yj = (-1 if circles[i].y < circles[j].y else 1) * abs(circles[j].radius * deltay / dist)
                    path = QPainterPath()
                    path.moveTo(circles[i].x + xi, circles[i].y + yi)
                    path.lineTo(circles[j].x + xj, circles[j].y + yj)
                    c = QColor(circles[i].borderColor)
                    c.setAlphaF(min(circles[i].opacity, circles[j].opacity) * ((self.linkDist - dist) / self.linkDist))
                    painter.setPen(QPen(c, (lineBorder * backgroundMlt if circles[i].background else lineBorder) * ((self.linkDist - dist) / self.linkDist)))
                    painter.drawPath(path)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = CircleLineWindow()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())