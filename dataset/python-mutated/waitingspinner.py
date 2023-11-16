"""
The MIT License (MIT)

Copyright (c) 2012-2014 Alexander Turkin
Copyright (c) 2014 William Hallatt
Copyright (c) 2015 Jacob Dawid
Copyright (c) 2016 Luca Weiss
Copyright (c) 2017- Spyder Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

See NOTICE.txt in the Spyder repository root for more detailed information.

Minimally adapted from waitingspinnerwidget.py of the
`QtWaitingSpinner Python Fork <https://github.com/z3ntu/QtWaitingSpinner>`_.
A port of `QtWaitingSpinner <https://github.com/snowwlex/QtWaitingSpinner>`_.
"""
import math
from qtpy.QtCore import QRect, Qt, QTimer
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QWidget

class QWaitingSpinner(QWidget):

    def __init__(self, parent, centerOnParent=True, disableParentWhenSpinning=False, modality=Qt.NonModal):
        if False:
            return 10
        QWidget.__init__(self, parent)
        self._centerOnParent = centerOnParent
        self._disableParentWhenSpinning = disableParentWhenSpinning
        self._color = QColor(Qt.black)
        self._roundness = 100.0
        self._minimumTrailOpacity = 3.141592653589793
        self._trailFadePercentage = 80.0
        self._trailSizeDecreasing = False
        self._revolutionsPerSecond = 1.5707963267948966
        self._numberOfLines = 20
        self._lineLength = 10
        self._lineWidth = 2
        self._innerRadius = 10
        self._currentCounter = 0
        self._isSpinning = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()
        self.setWindowModality(modality)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()

    def paintEvent(self, QPaintEvent):
        if False:
            while True:
                i = 10
        if not self._isSpinning:
            return
        self.updatePosition()
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0
        painter.setPen(Qt.NoPen)
        for i in range(0, self._numberOfLines):
            painter.save()
            painter.translate(self._innerRadius + self._lineLength, self._innerRadius + self._lineLength)
            rotateAngle = float(360 * i) / float(self._numberOfLines)
            painter.rotate(rotateAngle)
            painter.translate(self._innerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i, self._currentCounter, self._numberOfLines)
            color = self.currentLineColor(distance, self._numberOfLines, self._trailFadePercentage, self._minimumTrailOpacity, self._color)
            if self._trailSizeDecreasing:
                sf = (self._numberOfLines - distance) / self._numberOfLines
            else:
                sf = 1
            painter.setBrush(color)
            rect = QRect(0, round(-self._lineWidth / 2), round(sf * self._lineLength), round(sf * self._lineWidth))
            painter.drawRoundedRect(rect, self._roundness, self._roundness, Qt.RelativeSize)
            painter.restore()

    def start(self):
        if False:
            return 10
        self.updatePosition()
        self._isSpinning = True
        if self.parentWidget and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(False)
        if not self._timer.isActive():
            self._timer.start()
            self._currentCounter = 0
        self.show()

    def stop(self):
        if False:
            print('Hello World!')
        if not self._isSpinning:
            return
        self._isSpinning = False
        if self.parentWidget() and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(True)
        if self._timer.isActive():
            self._timer.stop()
            self._currentCounter = 0
        self.show()
        self.repaint()

    def setNumberOfLines(self, lines):
        if False:
            i = 10
            return i + 15
        self._numberOfLines = lines
        self._currentCounter = 0
        self.updateTimer()

    def setLineLength(self, length):
        if False:
            print('Hello World!')
        self._lineLength = length
        self.updateSize()

    def setLineWidth(self, width):
        if False:
            print('Hello World!')
        self._lineWidth = width
        self.updateSize()

    def setInnerRadius(self, radius):
        if False:
            print('Hello World!')
        self._innerRadius = radius
        self.updateSize()

    def color(self):
        if False:
            return 10
        return self._color

    def roundness(self):
        if False:
            return 10
        return self._roundness

    def minimumTrailOpacity(self):
        if False:
            return 10
        return self._minimumTrailOpacity

    def trailFadePercentage(self):
        if False:
            i = 10
            return i + 15
        return self._trailFadePercentage

    def revolutionsPersSecond(self):
        if False:
            while True:
                i = 10
        return self._revolutionsPerSecond

    def numberOfLines(self):
        if False:
            for i in range(10):
                print('nop')
        return self._numberOfLines

    def lineLength(self):
        if False:
            i = 10
            return i + 15
        return self._lineLength

    def isTrailSizeDecreasing(self):
        if False:
            print('Hello World!')
        '\n        Return whether the length and thickness of the trailing lines\n        are decreasing.\n        '
        return self._trailSizeDecreasing

    def lineWidth(self):
        if False:
            print('Hello World!')
        return self._lineWidth

    def innerRadius(self):
        if False:
            while True:
                i = 10
        return self._innerRadius

    def isSpinning(self):
        if False:
            i = 10
            return i + 15
        return self._isSpinning

    def setRoundness(self, roundness):
        if False:
            while True:
                i = 10
        self._roundness = max(0.0, min(100.0, roundness))

    def setColor(self, color=Qt.black):
        if False:
            i = 10
            return i + 15
        self._color = QColor(color)

    def setRevolutionsPerSecond(self, revolutionsPerSecond):
        if False:
            while True:
                i = 10
        self._revolutionsPerSecond = revolutionsPerSecond
        self.updateTimer()

    def setTrailFadePercentage(self, trail):
        if False:
            i = 10
            return i + 15
        self._trailFadePercentage = trail

    def setTrailSizeDecreasing(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set whether the length and thickness of the trailing lines\n        are decreasing.\n        '
        self._trailSizeDecreasing = value

    def setMinimumTrailOpacity(self, minimumTrailOpacity):
        if False:
            while True:
                i = 10
        self._minimumTrailOpacity = minimumTrailOpacity

    def rotate(self):
        if False:
            while True:
                i = 10
        self._currentCounter += 1
        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0
        self.update()

    def updateSize(self):
        if False:
            for i in range(10):
                print('nop')
        size = int((self._innerRadius + self._lineLength) * 2)
        self.setFixedSize(size, size)

    def updateTimer(self):
        if False:
            return 10
        self._timer.setInterval(int(1000 / (self._numberOfLines * self._revolutionsPerSecond)))

    def updatePosition(self):
        if False:
            while True:
                i = 10
        if self.parentWidget() and self._centerOnParent:
            self.move(int(self.parentWidget().width() / 2 - self.width() / 2), int(self.parentWidget().height() / 2 - self.height() / 2))

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        if False:
            i = 10
            return i + 15
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc, minOpacity, colorinput):
        if False:
            return 10
        color = QColor(colorinput)
        if countDistance == 0:
            return color
        minAlphaF = minOpacity / 100.0
        distanceThreshold = int(math.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0))
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)
        else:
            alphaDiff = color.alphaF() - minAlphaF
            gradient = alphaDiff / float(distanceThreshold + 1)
            resultAlpha = color.alphaF() - gradient * countDistance
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color