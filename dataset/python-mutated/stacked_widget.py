from typing import List
from PyQt5.QtCore import QAbstractAnimation, QEasingCurve, QPoint, QPropertyAnimation, pyqtSignal
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QStackedWidget, QWidget

class OpacityAniStackedWidget(QStackedWidget):
    """ Stacked widget with fade in and fade out animation """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self.__nextIndex = 0
        self.__effects = []
        self.__anis = []

    def addWidget(self, w: QWidget):
        if False:
            return 10
        super().addWidget(w)
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(1)
        ani = QPropertyAnimation(effect, b'opacity', self)
        ani.setDuration(220)
        ani.finished.connect(self.__onAniFinished)
        self.__anis.append(ani)
        self.__effects.append(effect)
        w.setGraphicsEffect(effect)

    def setCurrentIndex(self, index: int):
        if False:
            return 10
        index_ = self.currentIndex()
        if index == index_:
            return
        if index > index_:
            ani = self.__anis[index]
            ani.setStartValue(0)
            ani.setEndValue(1)
            super().setCurrentIndex(index)
        else:
            ani = self.__anis[index_]
            ani.setStartValue(1)
            ani.setEndValue(0)
        self.widget(index_).show()
        self.__nextIndex = index
        ani.start()

    def setCurrentWidget(self, w: QWidget):
        if False:
            i = 10
            return i + 15
        self.setCurrentIndex(self.indexOf(w))

    def __onAniFinished(self):
        if False:
            while True:
                i = 10
        super().setCurrentIndex(self.__nextIndex)

class PopUpAniInfo:
    """ Pop up ani info """

    def __init__(self, widget: QWidget, deltaX: int, deltaY, ani: QPropertyAnimation):
        if False:
            for i in range(10):
                print('nop')
        self.widget = widget
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.ani = ani

class PopUpAniStackedWidget(QStackedWidget):
    """ Stacked widget with pop up animation """
    aniFinished = pyqtSignal()
    aniStart = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.aniInfos = []
        self._nextIndex = None
        self._ani = None

    def addWidget(self, widget, deltaX=0, deltaY=76):
        if False:
            print('Hello World!')
        ' add widget to window\n\n        Parameters\n        -----------\n        widget:\n            widget to be added\n\n        deltaX: int\n            the x-axis offset from the beginning to the end of animation\n\n        deltaY: int\n            the y-axis offset from the beginning to the end of animation\n        '
        super().addWidget(widget)
        self.aniInfos.append(PopUpAniInfo(widget=widget, deltaX=deltaX, deltaY=deltaY, ani=QPropertyAnimation(widget, b'pos')))

    def setCurrentIndex(self, index: int, needPopOut: bool=False, showNextWidgetDirectly: bool=True, duration: int=250, easingCurve=QEasingCurve.OutQuad):
        if False:
            print('Hello World!')
        ' set current window to display\n\n        Parameters\n        ----------\n        index: int\n            the index of widget to display\n\n        isNeedPopOut: bool\n            need pop up animation or not\n\n        showNextWidgetDirectly: bool\n            whether to show next widget directly when animation started\n\n        duration: int\n            animation duration\n\n        easingCurve: QEasingCurve\n            the interpolation mode of animation\n        '
        if index < 0 or index >= self.count():
            raise Exception(f'The index `{index}` is illegal')
        if index == self.currentIndex():
            return
        if self._ani and self._ani.state() == QAbstractAnimation.Running:
            self._ani.stop()
            self.__onAniFinished()
        self._nextIndex = index
        nextAniInfo = self.aniInfos[index]
        currentAniInfo = self.aniInfos[self.currentIndex()]
        currentWidget = self.currentWidget()
        nextWidget = nextAniInfo.widget
        ani = currentAniInfo.ani if needPopOut else nextAniInfo.ani
        self._ani = ani
        if needPopOut:
            (deltaX, deltaY) = (currentAniInfo.deltaX, currentAniInfo.deltaY)
            pos = currentWidget.pos() + QPoint(deltaX, deltaY)
            self.__setAnimation(ani, currentWidget.pos(), pos, duration, easingCurve)
            nextWidget.setVisible(showNextWidgetDirectly)
        else:
            (deltaX, deltaY) = (nextAniInfo.deltaX, nextAniInfo.deltaY)
            pos = nextWidget.pos() + QPoint(deltaX, deltaY)
            self.__setAnimation(ani, pos, QPoint(nextWidget.x(), 0), duration, easingCurve)
            super().setCurrentIndex(index)
        ani.finished.connect(self.__onAniFinished)
        ani.start()
        self.aniStart.emit()

    def setCurrentWidget(self, widget, needPopOut: bool=False, showNextWidgetDirectly: bool=True, duration: int=250, easingCurve=QEasingCurve.OutQuad):
        if False:
            print('Hello World!')
        ' set currect widget\n\n        Parameters\n        ----------\n        widget:\n            the widget to be displayed\n\n        isNeedPopOut: bool\n            need pop up animation or not\n\n        showNextWidgetDirectly: bool\n            whether to show next widget directly when animation started\n\n        duration: int\n            animation duration\n\n        easingCurve: QEasingCurve\n            the interpolation mode of animation\n        '
        self.setCurrentIndex(self.indexOf(widget), needPopOut, showNextWidgetDirectly, duration, easingCurve)

    def __setAnimation(self, ani, startValue, endValue, duration, easingCurve=QEasingCurve.Linear):
        if False:
            return 10
        ' set the config of animation '
        ani.setEasingCurve(easingCurve)
        ani.setStartValue(startValue)
        ani.setEndValue(endValue)
        ani.setDuration(duration)

    def __onAniFinished(self):
        if False:
            return 10
        ' animation finished slot '
        self._ani.disconnect()
        super().setCurrentIndex(self._nextIndex)
        self.aniFinished.emit()