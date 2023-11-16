"""
Created on 2017年12月23日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ToolTip
@description: 
"""
import sys
try:
    from PyQt5.QtChart import QChartView, QChart, QLineSeries
    from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF
    from PyQt5.QtGui import QPainter, QCursor
    from PyQt5.QtWidgets import QApplication, QGraphicsProxyWidget, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QToolTip, QGraphicsLineItem
except ImportError:
    from PySide2.QtCore import Qt, QRectF, QPoint, QPointF
    from PySide2.QtGui import QPainter, QCursor
    from PySide2.QtWidgets import QApplication, QGraphicsProxyWidget, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QToolTip, QGraphicsLineItem
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QLineSeries = QtCharts.QLineSeries
'\nclass CircleWidget(QGraphicsProxyWidget):\n\n    def __init__(self, color, *args, **kwargs):\n        super(CircleWidget, self).__init__(*args, **kwargs)\n        label = QLabel()\n        label.setMinimumSize(12, 12)\n        label.setMaximumSize(12, 12)\n        label.setStyleSheet(\n            "border:1px solid green;border-radius:6px;background: %s;" % color)\n        self.setWidget(label)\n\n\nclass TextWidget(QGraphicsProxyWidget):\n\n    def __init__(self, text, *args, **kwargs):\n        super(TextWidget, self).__init__(*args, **kwargs)\n        self.setWidget(QLabel(text, styleSheet="color:white;"))\n\n\nclass GraphicsWidget(QGraphicsWidget):\n\n    def __init__(self, *args, **kwargs):\n        super(GraphicsWidget, self).__init__(*args, **kwargs)\n#         self.setFlags(self.ItemClipsChildrenToShape)\n        self.setZValue(999)\n        layout = QGraphicsGridLayout(self)\n        for row in range(6):\n            layout.addItem(CircleWidget("red"), row, 0)\n            layout.addItem(TextWidget("red"), row, 1)\n        self.hide()\n\n    def show(self, pos):\n        self.setGeometry(pos.x(), pos.y(), self.size().width(),\n                         self.size().height())\n        super(GraphicsWidget, self).show()\n'

class ToolTipItem(QWidget):

    def __init__(self, color, text, parent=None):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        self.textLabel.setText(text)

class ToolTipWidget(QWidget):
    Cache = {}

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ToolTipWidget, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('ToolTipWidget{background: rgba(50,50,50,70);}')
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

class GraphicsProxyWidget(QGraphicsProxyWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(GraphicsProxyWidget, self).__init__(*args, **kwargs)
        self.setZValue(999)
        self.tipWidget = ToolTipWidget()
        self.setWidget(self.tipWidget)
        self.hide()

    def show(self, title, points, pos):
        if False:
            while True:
                i = 10
        self.setGeometry(QRectF(pos, self.size()))
        self.tipWidget.updateUi(title, points)
        super(GraphicsProxyWidget, self).show()

class ChartView(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ChartView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        self.setRenderHint(QPainter.Antialiasing)
        self.initChart()
        self.toolTipWidget = GraphicsProxyWidget(self._chart)
        self.lineItem = QGraphicsLineItem(self._chart)
        self.lineItem.setZValue(998)
        self.lineItem.hide()
        (axisX, axisY) = (self._chart.axisX(), self._chart.axisY())
        (self.min_x, self.max_x) = (axisX.min(), axisX.max())
        (self.min_y, self.max_y) = (axisY.min(), axisY.max())
        self.point_top = self._chart.mapToPosition(QPointF(self.min_x, self.max_y))
        self.point_bottom = self._chart.mapToPosition(QPointF(self.min_x, self.min_y))
        self.step_x = (self.max_x - self.min_x) / (axisX.tickCount() - 1)

    def mouseMoveEvent(self, event):
        if False:
            print('Hello World!')
        super(ChartView, self).mouseMoveEvent(event)
        x = self._chart.mapToValue(event.pos()).x()
        y = self._chart.mapToValue(event.pos()).y()
        index = round((x - self.min_x) / self.step_x)
        pos_x = self._chart.mapToPosition(QPointF(index * self.step_x + self.min_x, self.min_y))
        points = [(serie, serie.at(index)) for serie in self._chart.series() if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y]
        if points:
            self.lineItem.setLine(pos_x.x(), self.point_top.y(), pos_x.x(), self.point_bottom.y())
            self.lineItem.show()
            self.toolTipWidget.show('', points, event.pos() + QPoint(20, 20))
        else:
            self.toolTipWidget.hide()
            self.lineItem.hide()

    def onSeriesHoverd(self, point, state):
        if False:
            i = 10
            return i + 15
        if state:
            try:
                name = self.sender().name()
            except:
                name = ''
            QToolTip.showText(QCursor.pos(), '%s\nx: %s\ny: %s' % (name, point.x(), point.y()))

    def initChart(self):
        if False:
            i = 10
            return i + 15
        self._chart = QChart(title='Line Chart')
        self._chart.setAcceptHoverEvents(True)
        dataTable = [[120, 132, 101, 134, 90, 230, 210], [220, 182, 191, 234, 290, 330, 310], [150, 232, 201, 154, 190, 330, 410], [320, 332, 301, 334, 390, 330, 320], [820, 932, 901, 934, 1290, 1330, 1320]]
        for (i, data_list) in enumerate(dataTable):
            series = QLineSeries(self._chart)
            for (j, v) in enumerate(data_list):
                series.append(j, v)
            series.setName('Series ' + str(i))
            series.setPointsVisible(True)
            series.hovered.connect(self.onSeriesHoverd)
            self._chart.addSeries(series)
        self._chart.createDefaultAxes()
        self._chart.axisX().setTickCount(7)
        self._chart.axisY().setTickCount(7)
        self._chart.axisY().setRange(0, 1500)
        self.setChart(self._chart)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('QToolTip {\n    border: none;\n    padding: 5px;\n    color: white;\n    background: rgb(50,50,50);\n    opacity: 100;\n}')
    view = ChartView()
    view.show()
    sys.exit(app.exec_())