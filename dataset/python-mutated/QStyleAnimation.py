"""
Created on 2022/02/26
@author: Irony
@site: https://pyqt.site, https://github.com/PyQt5
@email: 892768447@qq.com
@file: QStyleAnimation.py
@description:
"""
from enum import IntEnum
try:
    from PyQt5.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QTime
except ImportError:
    from PySide2.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QTime
ScrollBarFadeOutDuration = 200.0
ScrollBarFadeOutDelay = 450.0
StyleAnimationUpdate = 213

class QStyleAnimation(QAbstractAnimation):
    FrameRate = IntEnum('FrameRate', ['DefaultFps', 'SixtyFps', 'ThirtyFps', 'TwentyFps', 'FifteenFps'])

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(QStyleAnimation, self).__init__(*args, **kwargs)
        self._delay = 0
        self._duration = -1
        self._startTime = QTime.currentTime()
        self._fps = self.FrameRate.ThirtyFps
        self._skip = 0

    def target(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent()

    def duration(self):
        if False:
            return 10
        return self._duration

    def setDuration(self, duration):
        if False:
            i = 10
            return i + 15
        self._duration = duration

    def delay(self):
        if False:
            for i in range(10):
                print('nop')
        return self._delay

    def setDelay(self, delay):
        if False:
            return 10
        self._delay = delay

    def startTime(self):
        if False:
            i = 10
            return i + 15
        return self._startTime

    def setStartTime(self, time):
        if False:
            i = 10
            return i + 15
        self._startTime = time

    def frameRate(self):
        if False:
            for i in range(10):
                print('nop')
        return self._fps

    def setFrameRate(self, fps):
        if False:
            print('Hello World!')
        self._fps = fps

    def updateTarget(self):
        if False:
            while True:
                i = 10
        event = QEvent(QEvent.Type(StyleAnimationUpdate))
        event.setAccepted(False)
        QCoreApplication.sendEvent(self.target(), event)
        if not event.isAccepted():
            self.stop()

    def start(self):
        if False:
            print('Hello World!')
        self._skip = 0
        super(QStyleAnimation, self).start(QAbstractAnimation.KeepWhenStopped)

    def isUpdateNeeded(self):
        if False:
            i = 10
            return i + 15
        return self.currentTime() > self._delay

    def updateCurrentTime(self, _):
        if False:
            while True:
                i = 10
        self._skip += 1
        if self._skip >= self._fps:
            self._skip = 0
            if self.parent() and self.isUpdateNeeded():
                self.updateTarget()

class QProgressStyleAnimation(QStyleAnimation):

    def __init__(self, speed, *args, **kwargs):
        if False:
            print('Hello World!')
        super(QProgressStyleAnimation, self).__init__(*args, **kwargs)
        self._speed = speed
        self._step = -1

    def animationStep(self):
        if False:
            while True:
                i = 10
        return self.currentTime() / (1000.0 / self._speed)

    def progressStep(self, width):
        if False:
            i = 10
            return i + 15
        step = self.animationStep()
        progress = step * width / self._speed % width
        if step * width / self._speed % (2 * width) >= width:
            progress = width - progress
        return progress

    def speed(self):
        if False:
            print('Hello World!')
        return self._speed

    def setSpeed(self, speed):
        if False:
            for i in range(10):
                print('nop')
        self._speed = speed

    def isUpdateNeeded(self):
        if False:
            i = 10
            return i + 15
        if super(QProgressStyleAnimation, self).isUpdateNeeded():
            current = self.animationStep()
            if self._step == -1 or self._step != current:
                self._step = current
                return True
        return False