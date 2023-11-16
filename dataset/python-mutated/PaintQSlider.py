"""
Created on 2018年5月15日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: PaintQSlider
@description: 
"""
from PyQt5.QtCore import Qt, QRect, QPointF
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QSlider, QWidget, QVBoxLayout, QProxyStyle, QStyle, QStyleOptionSlider

class SliderStyle(QProxyStyle):

    def subControlRect(self, control, option, subControl, widget=None):
        if False:
            return 10
        rect = super(SliderStyle, self).subControlRect(control, option, subControl, widget)
        if subControl == QStyle.SC_SliderHandle:
            if option.orientation == Qt.Horizontal:
                radius = int(widget.height() / 3)
                offset = int(radius / 3)
                if option.state & QStyle.State_MouseOver:
                    x = min(rect.x() - offset, widget.width() - radius)
                    x = x if x >= 0 else 0
                else:
                    radius = offset
                    x = min(rect.x(), widget.width() - radius)
                rect = QRect(x, int((rect.height() - radius) / 2), radius, radius)
            else:
                radius = int(widget.width() / 3)
                offset = int(radius / 3)
                if option.state & QStyle.State_MouseOver:
                    y = min(rect.y() - offset, widget.height() - radius)
                    y = y if y >= 0 else 0
                else:
                    radius = offset
                    y = min(rect.y(), widget.height() - radius)
                rect = QRect(int((rect.width() - radius) / 2), y, radius, radius)
            return rect
        return rect

class PaintQSlider(QSlider):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(PaintQSlider, self).__init__(*args, **kwargs)
        self.setStyle(SliderStyle())

    def paintEvent(self, _):
        if False:
            print('Hello World!')
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
        painter.setPen(Qt.white)
        painter.setBrush(Qt.white)
        if self.orientation() == Qt.Horizontal:
            y = self.height() / 2
            painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
        else:
            x = self.width() / 2
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.setPen(Qt.NoPen)
        if option.state & QStyle.State_MouseOver:
            r = rect.height() / 2
            painter.setBrush(QColor(255, 255, 255, 100))
            painter.drawRoundedRect(rect, r, r)
            rect = rect.adjusted(4, 4, -4, -4)
            r = rect.height() / 2
            painter.setBrush(QColor(255, 255, 255, 255))
            painter.drawRoundedRect(rect, r, r)
            painter.setPen(Qt.white)
            if self.orientation() == Qt.Horizontal:
                (x, y) = (rect.x(), rect.y() - rect.height() - 2)
            else:
                (x, y) = (rect.x() - rect.width() - 2, rect.y())
            painter.drawText(x, y, rect.width(), rect.height(), Qt.AlignCenter, str(self.value()))
        else:
            r = rect.height() / 2
            painter.setBrush(Qt.white)
            painter.drawRoundedRect(rect, r, r)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.addWidget(PaintQSlider(Qt.Vertical, self, minimumWidth=90))
        layout.addWidget(PaintQSlider(Qt.Horizontal, self, minimumHeight=90))
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = Window()
    w.setStyleSheet('QWidget {background: gray;}')
    w.show()
    sys.exit(app.exec_())