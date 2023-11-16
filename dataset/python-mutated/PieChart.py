"""
Created on 2019/10/2
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: PieChart
@description: 饼状图表
"""
try:
    from PyQt5.QtChart import QChartView, QChart, QPieSeries
    from PyQt5.QtGui import QPainter, QColor
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PySide2.QtGui import QPainter, QColor
    from PySide2.QtWidgets import QApplication
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QPieSeries = QtCharts.QPieSeries

class Window(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 300)
        self.setRenderHint(QPainter.Antialiasing)
        chart = QChart()
        self.setChart(chart)
        chart.setTitle('Simple piechart example')
        chart.addSeries(self.getSeries())

    def getSeries(self):
        if False:
            for i in range(10):
                print('nop')
        series = QPieSeries()
        slice0 = series.append('10%', 1)
        series.append('20%', 2)
        series.append('70%', 7)
        series.setLabelsVisible()
        series.setPieSize(0.5)
        slice0.setLabelVisible()
        slice0.setExploded()
        slice0.setColor(QColor(255, 0, 0, 100))
        return series
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())