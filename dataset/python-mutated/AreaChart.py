"""
Created on 2019/10/2
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: AreaChart
@description: 区域图表
"""
try:
    from PyQt5.QtChart import QChartView, QChart, QLineSeries, QAreaSeries
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QColor, QGradient, QLinearGradient, QPainter, QPen
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PySide2.QtCore import QPointF
    from PySide2.QtGui import QColor, QGradient, QLinearGradient, QPainter, QPen
    from PySide2.QtWidgets import QApplication
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QLineSeries = QtCharts.QLineSeries
    QAreaSeries = QtCharts.QAreaSeries

class Window(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 300)
        self.setRenderHint(QPainter.Antialiasing)
        chart = QChart()
        self.setChart(chart)
        chart.setTitle('Simple areachart example')
        chart.addSeries(self.getSeries())
        chart.createDefaultAxes()
        chart.axisX().setRange(0, 20)
        chart.axisY().setRange(0, 10)

    def getSeries(self):
        if False:
            return 10
        series0 = QLineSeries(self)
        series1 = QLineSeries(self)
        series0 << QPointF(1, 5) << QPointF(3, 7) << QPointF(7, 6) << QPointF(9, 7) << QPointF(12, 6) << QPointF(16, 7) << QPointF(18, 5)
        series1 << QPointF(1, 3) << QPointF(3, 4) << QPointF(7, 3) << QPointF(8, 2) << QPointF(12, 3) << QPointF(16, 4) << QPointF(18, 3)
        series = QAreaSeries(series0, series1)
        series.setName('Batman')
        pen = QPen(366085)
        pen.setWidth(3)
        series.setPen(pen)
        gradient = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        gradient.setColorAt(0.0, QColor(3982908))
        gradient.setColorAt(1.0, QColor(2553382))
        gradient.setCoordinateMode(QGradient.ObjectBoundingMode)
        series.setBrush(gradient)
        return series
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())