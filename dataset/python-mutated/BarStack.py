"""
Created on 2017年12月28日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: charts.bar.BarStack
@description: like http://echarts.baidu.com/demo.html#bar-stack
"""
import sys
from random import randint
try:
    from PyQt5.QtChart import QChartView, QChart, QBarSeries, QBarSet, QBarCategoryAxis
    from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint
    from PyQt5.QtGui import QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QGraphicsLineItem, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QGraphicsProxyWidget
except ImportError:
    from PySide2.QtCore import Qt, QPointF, QRectF, QPoint
    from PySide2.QtGui import QPainter, QPen
    from PySide2.QtWidgets import QApplication, QGraphicsLineItem, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QGraphicsProxyWidget
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QBarSeries = QtCharts.QBarSeries
    QBarSet = QtCharts.QBarSet
    QBarCategoryAxis = QtCharts.QBarCategoryAxis

class ToolTipItem(QWidget):

    def __init__(self, color, text, parent=None):
        if False:
            while True:
                i = 10
        super(ToolTipItem, self).__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        clabel = QLabel(self)
        clabel.setMinimumSize(12, 12)
        clabel.setMaximumSize(12, 12)
        clabel.setStyleSheet('border-radius:6px;background: rgba(%s,%s,%s,%s);' % (color.red(), color.green(), color.blue(), color.alpha()))
        layout.addWidget(clabel)
        self.textLabel = QLabel(text, self, styleSheet='color:white;')
        layout.addWidget(self.textLabel)

    def setText(self, text):
        if False:
            i = 10
            return i + 15
        self.textLabel.setText(text)

class ToolTipWidget(QWidget):
    Cache = {}

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ToolTipWidget, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('ToolTipWidget{background: rgba(50, 50, 50, 100);}')
        layout = QVBoxLayout(self)
        self.titleLabel = QLabel(self, styleSheet='color:white;')
        layout.addWidget(self.titleLabel)

    def updateUi(self, title, bars):
        if False:
            i = 10
            return i + 15
        self.titleLabel.setText(title)
        for (bar, value) in bars:
            if bar not in self.Cache:
                item = ToolTipItem(bar.color(), (bar.label() or '-') + ':' + str(value), self)
                self.layout().addWidget(item)
                self.Cache[bar] = item
            else:
                self.Cache[bar].setText((bar.label() or '-') + ':' + str(value))
            brush = bar.brush()
            color = brush.color()
            self.Cache[bar].setVisible(color.alphaF() == 1.0)
        self.adjustSize()

class GraphicsProxyWidget(QGraphicsProxyWidget):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(GraphicsProxyWidget, self).__init__(*args, **kwargs)
        self.setZValue(999)
        self.tipWidget = ToolTipWidget()
        self.setWidget(self.tipWidget)
        self.hide()

    def width(self):
        if False:
            return 10
        return self.size().width()

    def height(self):
        if False:
            while True:
                i = 10
        return self.size().height()

    def show(self, title, bars, pos):
        if False:
            return 10
        self.setGeometry(QRectF(pos, self.size()))
        self.tipWidget.updateUi(title, bars)
        super(GraphicsProxyWidget, self).show()

class ChartView(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ChartView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        self.setRenderHint(QPainter.Antialiasing)
        self.initChart()
        self.toolTipWidget = GraphicsProxyWidget(self._chart)
        self.lineItem = QGraphicsLineItem(self._chart)
        pen = QPen(Qt.gray)
        self.lineItem.setPen(pen)
        self.lineItem.setZValue(998)
        self.lineItem.hide()
        (axisX, axisY) = (self._chart.axisX(), self._chart.axisY())
        self.category_len = len(axisX.categories())
        (self.min_x, self.max_x) = (-0.5, self.category_len - 0.5)
        (self.min_y, self.max_y) = (axisY.min(), axisY.max())
        self.point_top = self._chart.mapToPosition(QPointF(self.min_x, self.max_y))

    def mouseMoveEvent(self, event):
        if False:
            while True:
                i = 10
        super(ChartView, self).mouseMoveEvent(event)
        pos = event.pos()
        x = self._chart.mapToValue(pos).x()
        y = self._chart.mapToValue(pos).y()
        index = round(x)
        serie = self._chart.series()[0]
        bars = [(bar, bar.at(index)) for bar in serie.barSets() if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y]
        if bars:
            right_top = self._chart.mapToPosition(QPointF(self.max_x, self.max_y))
            step_x = round((right_top.x() - self.point_top.x()) / self.category_len)
            posx = self._chart.mapToPosition(QPointF(x, self.min_y))
            self.lineItem.setLine(posx.x(), self.point_top.y(), posx.x(), posx.y())
            self.lineItem.show()
            try:
                title = self.categories[index]
            except:
                title = ''
            t_width = self.toolTipWidget.width()
            t_height = self.toolTipWidget.height()
            x = pos.x() - t_width if self.width() - pos.x() - 20 < t_width else pos.x()
            y = pos.y() - t_height if self.height() - pos.y() - 20 < t_height else pos.y()
            self.toolTipWidget.show(title, bars, QPoint(x, y))
        else:
            self.toolTipWidget.hide()
            self.lineItem.hide()

    def handleMarkerClicked(self):
        if False:
            for i in range(10):
                print('nop')
        marker = self.sender()
        if not marker:
            return
        bar = marker.barset()
        if not bar:
            return
        brush = bar.brush()
        color = brush.color()
        alpha = 0.0 if color.alphaF() == 1.0 else 1.0
        color.setAlphaF(alpha)
        brush.setColor(color)
        bar.setBrush(brush)
        brush = marker.labelBrush()
        color = brush.color()
        alpha = 0.4 if color.alphaF() == 1.0 else 1.0
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setLabelBrush(brush)
        brush = marker.brush()
        color = brush.color()
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setBrush(brush)

    def handleMarkerHovered(self, status):
        if False:
            print('Hello World!')
        marker = self.sender()
        if not marker:
            return
        bar = marker.barset()
        if not bar:
            return
        pen = bar.pen()
        if not pen:
            return
        pen.setWidth(pen.width() + (1 if status else -1))
        bar.setPen(pen)

    def handleBarHoverd(self, status, index):
        if False:
            return 10
        bar = self.sender()
        pen = bar.pen()
        if not pen:
            return
        pen.setWidth(pen.width() + (1 if status else -1))
        bar.setPen(pen)

    def initChart(self):
        if False:
            while True:
                i = 10
        self._chart = QChart(title='柱状图堆叠')
        self._chart.setAcceptHoverEvents(True)
        self._chart.setAnimationOptions(QChart.SeriesAnimations)
        self.categories = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        names = ['邮件营销', '联盟广告', '视频广告', '直接访问', '搜索引擎']
        series = QBarSeries(self._chart)
        for name in names:
            bar = QBarSet(name)
            for _ in range(7):
                bar.append(randint(0, 10))
            series.append(bar)
            bar.hovered.connect(self.handleBarHoverd)
        self._chart.addSeries(series)
        self._chart.createDefaultAxes()
        axis_x = QBarCategoryAxis(self._chart)
        axis_x.append(self.categories)
        self._chart.setAxisX(axis_x, series)
        legend = self._chart.legend()
        legend.setVisible(True)
        for marker in legend.markers():
            marker.clicked.connect(self.handleMarkerClicked)
            marker.hovered.connect(self.handleMarkerHovered)
        self.setChart(self._chart)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = ChartView()
    view.show()
    sys.exit(app.exec_())