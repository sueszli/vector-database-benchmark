"""
Created on 2017年12月28日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: charts.line.LineStack
@description: like http://echarts.baidu.com/demo.html#line-stack
"""
import sys
try:
    from PyQt5.QtChart import QChartView, QChart, QLineSeries, QLegend, QCategoryAxis
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
    QLineSeries = QtCharts.QLineSeries
    QLegend = QtCharts.QLegend
    QCategoryAxis = QtCharts.QCategoryAxis

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
            while True:
                i = 10
        self.textLabel.setText(text)

class ToolTipWidget(QWidget):
    Cache = {}

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ToolTipWidget, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('ToolTipWidget{background: rgba(50, 50, 50, 100);}')
        layout = QVBoxLayout(self)
        self.titleLabel = QLabel(self, styleSheet='color:white;')
        layout.addWidget(self.titleLabel)

    def updateUi(self, title, points):
        if False:
            i = 10
            return i + 15
        self.titleLabel.setText(title)
        for (serie, point) in points:
            if serie not in self.Cache:
                item = ToolTipItem(serie.color(), (serie.name() or '-') + ':' + str(point.y()), self)
                self.layout().addWidget(item)
                self.Cache[serie] = item
            else:
                self.Cache[serie].setText((serie.name() or '-') + ':' + str(point.y()))
            self.Cache[serie].setVisible(serie.isVisible())
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
            for i in range(10):
                print('nop')
        return self.size().width()

    def height(self):
        if False:
            for i in range(10):
                print('nop')
        return self.size().height()

    def show(self, title, points, pos):
        if False:
            return 10
        self.setGeometry(QRectF(pos, self.size()))
        self.tipWidget.updateUi(title, points)
        super(GraphicsProxyWidget, self).show()

class ChartView(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ChartView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        self.setRenderHint(QPainter.Antialiasing)
        self.category = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        self.initChart()
        self.toolTipWidget = GraphicsProxyWidget(self._chart)
        self.lineItem = QGraphicsLineItem(self._chart)
        pen = QPen(Qt.gray)
        pen.setWidth(1)
        self.lineItem.setPen(pen)
        self.lineItem.setZValue(998)
        self.lineItem.hide()
        (axisX, axisY) = (self._chart.axisX(), self._chart.axisY())
        (self.min_x, self.max_x) = (axisX.min(), axisX.max())
        (self.min_y, self.max_y) = (axisY.min(), axisY.max())

    def resizeEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(ChartView, self).resizeEvent(event)
        self.point_top = self._chart.mapToPosition(QPointF(self.min_x, self.max_y))
        self.point_bottom = self._chart.mapToPosition(QPointF(self.min_x, self.min_y))
        self.step_x = (self.max_x - self.min_x) / (self._chart.axisX().tickCount() - 1)

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(ChartView, self).mouseMoveEvent(event)
        pos = event.pos()
        x = self._chart.mapToValue(pos).x()
        y = self._chart.mapToValue(pos).y()
        index = round((x - self.min_x) / self.step_x)
        points = [(serie, serie.at(index)) for serie in self._chart.series() if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y]
        if points:
            pos_x = self._chart.mapToPosition(QPointF(index * self.step_x + self.min_x, self.min_y))
            self.lineItem.setLine(pos_x.x(), self.point_top.y(), pos_x.x(), self.point_bottom.y())
            self.lineItem.show()
            try:
                title = self.category[index]
            except:
                title = ''
            t_width = self.toolTipWidget.width()
            t_height = self.toolTipWidget.height()
            x = pos.x() - t_width if self.width() - pos.x() - 20 < t_width else pos.x()
            y = pos.y() - t_height if self.height() - pos.y() - 20 < t_height else pos.y()
            self.toolTipWidget.show(title, points, QPoint(x, y))
        else:
            self.toolTipWidget.hide()
            self.lineItem.hide()

    def handleMarkerClicked(self):
        if False:
            while True:
                i = 10
        marker = self.sender()
        if not marker:
            return
        visible = not marker.series().isVisible()
        marker.series().setVisible(visible)
        marker.setVisible(True)
        alpha = 1.0 if visible else 0.4
        brush = marker.labelBrush()
        color = brush.color()
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setLabelBrush(brush)
        brush = marker.brush()
        color = brush.color()
        color.setAlphaF(alpha)
        brush.setColor(color)
        marker.setBrush(brush)
        pen = marker.pen()
        color = pen.color()
        color.setAlphaF(alpha)
        pen.setColor(color)
        marker.setPen(pen)

    def handleMarkerHovered(self, status):
        if False:
            for i in range(10):
                print('nop')
        marker = self.sender()
        if not marker:
            return
        series = marker.series()
        if not series:
            return
        pen = series.pen()
        if not pen:
            return
        pen.setWidth(pen.width() + (1 if status else -1))
        series.setPen(pen)

    def handleSeriesHoverd(self, point, state):
        if False:
            for i in range(10):
                print('nop')
        series = self.sender()
        pen = series.pen()
        if not pen:
            return
        pen.setWidth(pen.width() + (1 if state else -1))
        series.setPen(pen)

    def initChart(self):
        if False:
            for i in range(10):
                print('nop')
        self._chart = QChart(title='折线图堆叠')
        self._chart.setAcceptHoverEvents(True)
        self._chart.setAnimationOptions(QChart.SeriesAnimations)
        dataTable = [['邮件营销', [120, 132, 101, 134, 90, 230, 210]], ['联盟广告', [220, 182, 191, 234, 290, 330, 310]], ['视频广告', [150, 232, 201, 154, 190, 330, 410]], ['直接访问', [320, 332, 301, 334, 390, 330, 320]], ['搜索引擎', [820, 932, 901, 934, 1290, 1330, 1320]]]
        for (series_name, data_list) in dataTable:
            series = QLineSeries(self._chart)
            for (j, v) in enumerate(data_list):
                series.append(j, v)
            series.setName(series_name)
            series.setPointsVisible(True)
            series.hovered.connect(self.handleSeriesHoverd)
            self._chart.addSeries(series)
        self._chart.createDefaultAxes()
        axisX = self._chart.axisX()
        axisX.setTickCount(7)
        axisX.setGridLineVisible(False)
        axisY = self._chart.axisY()
        axisY.setTickCount(7)
        axisY.setRange(0, 1500)
        axis_x = QCategoryAxis(self._chart, labelsPosition=QCategoryAxis.AxisLabelsPositionOnValue)
        axis_x.setTickCount(7)
        axis_x.setGridLineVisible(False)
        min_x = axisX.min()
        max_x = axisX.max()
        step = (max_x - min_x) / (7 - 1)
        for i in range(0, 7):
            axis_x.append(self.category[i], min_x + i * step)
        self._chart.setAxisX(axis_x, self._chart.series()[-1])
        legend = self._chart.legend()
        legend.setMarkerShape(QLegend.MarkerShapeFromSeries)
        for marker in legend.markers():
            marker.clicked.connect(self.handleMarkerClicked)
            marker.hovered.connect(self.handleMarkerHovered)
        self.setChart(self._chart)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = ChartView()
    view.show()
    sys.exit(app.exec_())