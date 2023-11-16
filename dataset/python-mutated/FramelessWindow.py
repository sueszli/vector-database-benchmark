"""
Created on 2018年4月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FramelessWindow
@description:
"""
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QPoint
    from PyQt5.QtGui import QFont, QEnterEvent, QPainter, QColor, QPen
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QPushButton
except ImportError:
    from PySide2.QtCore import Qt, Signal as pyqtSignal, QPoint
    from PySide2.QtGui import QFont, QEnterEvent, QPainter, QColor, QPen
    from PySide2.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QPushButton

class TitleBar(QWidget):
    windowMinimumed = pyqtSignal()
    windowMaximumed = pyqtSignal()
    windowNormaled = pyqtSignal()
    windowClosed = pyqtSignal()
    windowMoved = pyqtSignal(QPoint)

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(TitleBar, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.mPos = None
        self.iconSize = 20
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(palette.Window, QColor(240, 240, 240))
        self.setPalette(palette)
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.iconLabel = QLabel(self)
        layout.addWidget(self.iconLabel)
        self.titleLabel = QLabel(self)
        self.titleLabel.setMargin(2)
        layout.addWidget(self.titleLabel)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        font = self.font() or QFont()
        font.setFamily('Webdings')
        self.buttonMinimum = QPushButton('0', self, clicked=self.windowMinimumed.emit, font=font, objectName='buttonMinimum')
        layout.addWidget(self.buttonMinimum)
        self.buttonMaximum = QPushButton('1', self, clicked=self.showMaximized, font=font, objectName='buttonMaximum')
        layout.addWidget(self.buttonMaximum)
        self.buttonClose = QPushButton('r', self, clicked=self.windowClosed.emit, font=font, objectName='buttonClose')
        layout.addWidget(self.buttonClose)
        self.setHeight()

    def showMaximized(self):
        if False:
            print('Hello World!')
        if self.buttonMaximum.text() == '1':
            self.buttonMaximum.setText('2')
            self.windowMaximumed.emit()
        else:
            self.buttonMaximum.setText('1')
            self.windowNormaled.emit()

    def setHeight(self, height=38):
        if False:
            return 10
        '设置标题栏高度'
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.buttonMinimum.setMinimumSize(height, height)
        self.buttonMinimum.setMaximumSize(height, height)
        self.buttonMaximum.setMinimumSize(height, height)
        self.buttonMaximum.setMaximumSize(height, height)
        self.buttonClose.setMinimumSize(height, height)
        self.buttonClose.setMaximumSize(height, height)

    def setTitle(self, title):
        if False:
            for i in range(10):
                print('nop')
        '设置标题'
        self.titleLabel.setText(title)

    def setIcon(self, icon):
        if False:
            while True:
                i = 10
        '设置图标'
        self.iconLabel.setPixmap(icon.pixmap(self.iconSize, self.iconSize))

    def setIconSize(self, size):
        if False:
            for i in range(10):
                print('nop')
        '设置图标大小'
        self.iconSize = size

    def enterEvent(self, event):
        if False:
            while True:
                i = 10
        self.setCursor(Qt.ArrowCursor)
        super(TitleBar, self).enterEvent(event)

    def mouseDoubleClickEvent(self, event):
        if False:
            print('Hello World!')
        super(TitleBar, self).mouseDoubleClickEvent(event)
        self.showMaximized()

    def mousePressEvent(self, event):
        if False:
            while True:
                i = 10
        '鼠标点击事件'
        if event.button() == Qt.LeftButton:
            self.mPos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        if False:
            while True:
                i = 10
        '鼠标弹起事件'
        self.mPos = None
        event.accept()

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if event.buttons() == Qt.LeftButton and self.mPos:
            self.windowMoved.emit(self.mapToGlobal(event.pos() - self.mPos))
        event.accept()
(Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom) = range(8)

class FramelessWindow(QWidget):
    Margins = 5

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(FramelessWindow, self).__init__(*args, **kwargs)
        self._pressed = False
        self.Direction = None
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setMouseTracking(True)
        layout = QVBoxLayout(self, spacing=0)
        layout.setContentsMargins(self.Margins, self.Margins, self.Margins, self.Margins)
        self.titleBar = TitleBar(self)
        layout.addWidget(self.titleBar)
        self.titleBar.windowMinimumed.connect(self.showMinimized)
        self.titleBar.windowMaximumed.connect(self.showMaximized)
        self.titleBar.windowNormaled.connect(self.showNormal)
        self.titleBar.windowClosed.connect(self.close)
        self.titleBar.windowMoved.connect(self.move)
        self.windowTitleChanged.connect(self.titleBar.setTitle)
        self.windowIconChanged.connect(self.titleBar.setIcon)

    def setTitleBarHeight(self, height=38):
        if False:
            print('Hello World!')
        '设置标题栏高度'
        self.titleBar.setHeight(height)

    def setIconSize(self, size):
        if False:
            while True:
                i = 10
        '设置图标的大小'
        self.titleBar.setIconSize(size)

    def setWidget(self, widget):
        if False:
            while True:
                i = 10
        '设置自己的控件'
        if hasattr(self, '_widget'):
            return
        self._widget = widget
        self._widget.setAutoFillBackground(True)
        palette = self._widget.palette()
        palette.setColor(palette.Window, QColor(240, 240, 240))
        self._widget.setPalette(palette)
        self._widget.installEventFilter(self)
        self.layout().addWidget(self._widget)

    def move(self, pos):
        if False:
            return 10
        if self.windowState() == Qt.WindowMaximized or self.windowState() == Qt.WindowFullScreen:
            return
        super(FramelessWindow, self).move(pos)

    def showMaximized(self):
        if False:
            return 10
        '最大化,要去除上下左右边界,如果不去除则边框地方会有空隙'
        super(FramelessWindow, self).showMaximized()
        self.layout().setContentsMargins(0, 0, 0, 0)

    def showNormal(self):
        if False:
            return 10
        '还原,要保留上下左右边界,否则没有边框无法调整'
        super(FramelessWindow, self).showNormal()
        self.layout().setContentsMargins(self.Margins, self.Margins, self.Margins, self.Margins)

    def eventFilter(self, obj, event):
        if False:
            for i in range(10):
                print('nop')
        '事件过滤器,用于解决鼠标进入其它控件后还原为标准鼠标样式'
        if isinstance(event, QEnterEvent):
            self.setCursor(Qt.ArrowCursor)
        return super(FramelessWindow, self).eventFilter(obj, event)

    def paintEvent(self, event):
        if False:
            return 10
        '由于是全透明背景窗口,重绘事件中绘制透明度为1的难以发现的边框,用于调整窗口大小'
        super(FramelessWindow, self).paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 255, 255, 1), 2 * self.Margins))
        painter.drawRect(self.rect())

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        '鼠标点击事件'
        super(FramelessWindow, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self._mpos = event.pos()
            self._pressed = True

    def mouseReleaseEvent(self, event):
        if False:
            i = 10
            return i + 15
        '鼠标弹起事件'
        super(FramelessWindow, self).mouseReleaseEvent(event)
        self._pressed = False
        self.Direction = None

    def mouseMoveEvent(self, event):
        if False:
            while True:
                i = 10
        '鼠标移动事件'
        super(FramelessWindow, self).mouseMoveEvent(event)
        pos = event.pos()
        (xPos, yPos) = (pos.x(), pos.y())
        (wm, hm) = (self.width() - self.Margins, self.height() - self.Margins)
        if self.isMaximized() or self.isFullScreen():
            self.Direction = None
            self.setCursor(Qt.ArrowCursor)
            return
        if event.buttons() == Qt.LeftButton and self._pressed:
            self._resizeWidget(pos)
            return
        if xPos <= self.Margins and yPos <= self.Margins:
            self.Direction = LeftTop
            self.setCursor(Qt.SizeFDiagCursor)
        elif wm <= xPos <= self.width() and hm <= yPos <= self.height():
            self.Direction = RightBottom
            self.setCursor(Qt.SizeFDiagCursor)
        elif wm <= xPos and yPos <= self.Margins:
            self.Direction = RightTop
            self.setCursor(Qt.SizeBDiagCursor)
        elif xPos <= self.Margins and hm <= yPos:
            self.Direction = LeftBottom
            self.setCursor(Qt.SizeBDiagCursor)
        elif 0 <= xPos <= self.Margins and self.Margins <= yPos <= hm:
            self.Direction = Left
            self.setCursor(Qt.SizeHorCursor)
        elif wm <= xPos <= self.width() and self.Margins <= yPos <= hm:
            self.Direction = Right
            self.setCursor(Qt.SizeHorCursor)
        elif self.Margins <= xPos <= wm and 0 <= yPos <= self.Margins:
            self.Direction = Top
            self.setCursor(Qt.SizeVerCursor)
        elif self.Margins <= xPos <= wm and hm <= yPos <= self.height():
            self.Direction = Bottom
            self.setCursor(Qt.SizeVerCursor)

    def _resizeWidget(self, pos):
        if False:
            print('Hello World!')
        '调整窗口大小'
        if self.Direction == None:
            return
        mpos = pos - self._mpos
        (xPos, yPos) = (mpos.x(), mpos.y())
        geometry = self.geometry()
        (x, y, w, h) = (geometry.x(), geometry.y(), geometry.width(), geometry.height())
        if self.Direction == LeftTop:
            if w - xPos > self.minimumWidth():
                x += xPos
                w -= xPos
            if h - yPos > self.minimumHeight():
                y += yPos
                h -= yPos
        elif self.Direction == RightBottom:
            if w + xPos > self.minimumWidth():
                w += xPos
                self._mpos = pos
            if h + yPos > self.minimumHeight():
                h += yPos
                self._mpos = pos
        elif self.Direction == RightTop:
            if h - yPos > self.minimumHeight():
                y += yPos
                h -= yPos
            if w + xPos > self.minimumWidth():
                w += xPos
                self._mpos.setX(pos.x())
        elif self.Direction == LeftBottom:
            if w - xPos > self.minimumWidth():
                x += xPos
                w -= xPos
            if h + yPos > self.minimumHeight():
                h += yPos
                self._mpos.setY(pos.y())
        elif self.Direction == Left:
            if w - xPos > self.minimumWidth():
                x += xPos
                w -= xPos
            else:
                return
        elif self.Direction == Right:
            if w + xPos > self.minimumWidth():
                w += xPos
                self._mpos = pos
            else:
                return
        elif self.Direction == Top:
            if h - yPos > self.minimumHeight():
                y += yPos
                h -= yPos
            else:
                return
        elif self.Direction == Bottom:
            if h + yPos > self.minimumHeight():
                h += yPos
                self._mpos = pos
            else:
                return
        self.setGeometry(x, y, w, h)