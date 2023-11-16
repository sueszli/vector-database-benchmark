"""
Created on 2018年11月24日
author: Irony
site: https://pyqt.site , https://github.com/PyQt5
email: 892768447@qq.com
file:
description: 参考 http://qt.shoutwiki.com/wiki/Extending_QStackedWidget_for_sliding_page_animations_in_Qt
"""
try:
    from PyQt5.QtCore import Qt, pyqtProperty, QEasingCurve, QPoint, QPropertyAnimation, QParallelAnimationGroup, QTimer
    from PyQt5.QtWidgets import QStackedWidget
except ImportError:
    from PySide2.QtCore import Qt, Property as pyqtProperty, QEasingCurve, QPoint, QPropertyAnimation, QParallelAnimationGroup, QTimer
    from PySide2.QtWidgets import QStackedWidget

class SlidingStackedWidget(QStackedWidget):
    (LEFT2RIGHT, RIGHT2LEFT, TOP2BOTTOM, BOTTOM2TOP, AUTOMATIC) = range(5)

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SlidingStackedWidget, self).__init__(*args, **kwargs)
        self._pnow = QPoint(0, 0)
        self._speed = 500
        self._now = 0
        self._current = 0
        self._next = 0
        self._active = 0
        self._orientation = Qt.Horizontal
        self._easing = QEasingCurve.Linear
        self._initAnimation()

    def setSpeed(self, speed=500):
        if False:
            return 10
        '设置动画速度\n        :param speed:       速度值,默认值为500\n        :type speed:        int\n        '
        self._speed = speed

    @pyqtProperty(int, fset=setSpeed)
    def speed(self):
        if False:
            i = 10
            return i + 15
        return self._speed

    def setOrientation(self, orientation=Qt.Horizontal):
        if False:
            print('Hello World!')
        '设置动画的方向(横向和纵向)\n        :param orientation:    方向(Qt.Horizontal或Qt.Vertical)\n        :type orientation:     http://doc.qt.io/qt-5/qt.html#Orientation-enum\n        '
        self._orientation = orientation

    @pyqtProperty(int, fset=setOrientation)
    def orientation(self):
        if False:
            return 10
        return self._orientation

    def setEasing(self, easing=QEasingCurve.OutBack):
        if False:
            i = 10
            return i + 15
        '设置动画的曲线类型\n        :param easing:    默认为QEasingCurve.OutBack\n        :type easing:     http://doc.qt.io/qt-5/qeasingcurve.html#Type-enum\n        '
        self._easing = easing

    @pyqtProperty(int, fset=setEasing)
    def easing(self):
        if False:
            print('Hello World!')
        return self._easing

    def slideInNext(self):
        if False:
            return 10
        '滑动到下一页'
        now = self.currentIndex()
        if now < self.count() - 1:
            self.slideInIdx(now + 1)
            self._current = now + 1

    def slideInPrev(self):
        if False:
            print('Hello World!')
        '滑动到上一页'
        now = self.currentIndex()
        if now > 0:
            self.slideInIdx(now - 1)
            self._current = now - 1

    def slideInIdx(self, idx, direction=4):
        if False:
            i = 10
            return i + 15
        '滑动到指定序号\n        :param idx:               序号\n        :type idx:                int\n        :param direction:         方向,默认是自动AUTOMATIC=4\n        :type direction:          int\n        '
        if idx > self.count() - 1:
            direction = self.TOP2BOTTOM if self._orientation == Qt.Vertical else self.RIGHT2LEFT
            idx = idx % self.count()
        elif idx < 0:
            direction = self.BOTTOM2TOP if self._orientation == Qt.Vertical else self.LEFT2RIGHT
            idx = (idx + self.count()) % self.count()
        self.slideInWgt(self.widget(idx), direction)

    def slideInWgt(self, widget, direction):
        if False:
            while True:
                i = 10
        '滑动到指定的widget\n        :param widget:        QWidget, QLabel, etc...\n        :type widget:         QWidget Base Class\n        :param direction:     方向\n        :type direction:      int\n        '
        if self._active:
            return
        self._active = 1
        _now = self.currentIndex()
        _next = self.indexOf(widget)
        if _now == _next:
            self._active = 0
            return
        w_now = self.widget(_now)
        w_next = self.widget(_next)
        if _now < _next:
            directionhint = self.TOP2BOTTOM if self._orientation == Qt.Vertical else self.RIGHT2LEFT
        else:
            directionhint = self.BOTTOM2TOP if self._orientation == Qt.Vertical else self.LEFT2RIGHT
        if direction == self.AUTOMATIC:
            direction = directionhint
        offsetX = self.frameRect().width()
        offsetY = self.frameRect().height()
        w_next.setGeometry(0, 0, offsetX, offsetY)
        if direction == self.BOTTOM2TOP:
            offsetX = 0
            offsetY = -offsetY
        elif direction == self.TOP2BOTTOM:
            offsetX = 0
        elif direction == self.RIGHT2LEFT:
            offsetX = -offsetX
            offsetY = 0
        elif direction == self.LEFT2RIGHT:
            offsetY = 0
        pnext = w_next.pos()
        pnow = w_now.pos()
        self._pnow = pnow
        w_next.move(pnext.x() - offsetX, pnext.y() - offsetY)
        w_next.show()
        w_next.raise_()
        self._animnow.setTargetObject(w_now)
        self._animnow.setDuration(self._speed)
        self._animnow.setEasingCurve(self._easing)
        self._animnow.setStartValue(QPoint(pnow.x(), pnow.y()))
        self._animnow.setEndValue(QPoint(offsetX + pnow.x(), offsetY + pnow.y()))
        self._animnext.setTargetObject(w_next)
        self._animnext.setDuration(self._speed)
        self._animnext.setEasingCurve(self._easing)
        self._animnext.setStartValue(QPoint(-offsetX + pnext.x(), offsetY + pnext.y()))
        self._animnext.setEndValue(QPoint(pnext.x(), pnext.y()))
        self._next = _next
        self._now = _now
        self._active = 1
        self._animgroup.start()

    def _initAnimation(self):
        if False:
            print('Hello World!')
        '初始化当前页和下一页的动画变量'
        self._animnow = QPropertyAnimation(self, propertyName=b'pos', duration=self._speed, easingCurve=self._easing)
        self._animnext = QPropertyAnimation(self, propertyName=b'pos', duration=self._speed, easingCurve=self._easing)
        self._animgroup = QParallelAnimationGroup(self, finished=self.animationDoneSlot)
        self._animgroup.addAnimation(self._animnow)
        self._animgroup.addAnimation(self._animnext)

    def setCurrentIndex(self, index):
        if False:
            return 10
        self.slideInIdx(index)

    def setCurrentWidget(self, widget):
        if False:
            i = 10
            return i + 15
        super(SlidingStackedWidget, self).setCurrentWidget(widget)
        self.setCurrentIndex(self.indexOf(widget))

    def animationDoneSlot(self):
        if False:
            print('Hello World!')
        '动画结束处理函数'
        QStackedWidget.setCurrentIndex(self, self._next)
        w = self.widget(self._now)
        w.hide()
        w.move(self._pnow)
        self._active = 0

    def autoStop(self):
        if False:
            print('Hello World!')
        '停止自动播放'
        if hasattr(self, '_autoTimer'):
            self._autoTimer.stop()

    def autoStart(self, msec=3000):
        if False:
            print('Hello World!')
        '自动轮播\n        :param time: 时间, 默认3000, 3秒\n        '
        if not hasattr(self, '_autoTimer'):
            self._autoTimer = QTimer(self, timeout=self._autoStart)
        self._autoTimer.stop()
        self._autoTimer.start(msec)

    def _autoStart(self):
        if False:
            while True:
                i = 10
        if self._current == self.count():
            self._current = 0
        self._current += 1
        self.setCurrentIndex(self._current)