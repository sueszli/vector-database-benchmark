"""
Created on 2019/10/2
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: HorizontalBarChart
@description: 横向柱状图表
"""
try:
    from PyQt5.QtChart import QChartView, QChart, QBarSet, QHorizontalBarSeries, QBarCategoryAxis
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPainter
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QPainter
    from PySide2.QtWidgets import QApplication
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QBarSet = QtCharts.QBarSet
    QHorizontalBarSeries = QtCharts.QHorizontalBarSeries
    QBarCategoryAxis = QtCharts.QBarCategoryAxis

class Window(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 300)
        self.setRenderHint(QPainter.Antialiasing)
        chart = QChart()
        self.setChart(chart)
        chart.setTitle('Simple horizontal barchart example')
        chart.setAnimationOptions(QChart.SeriesAnimations)
        series = self.getSeries()
        chart.addSeries(series)
        categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        axis = QBarCategoryAxis()
        axis.append(categories)
        chart.createDefaultAxes()
        chart.setAxisY(axis, series)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

    def getSeries(self):
        if False:
            print('Hello World!')
        set0 = QBarSet('Jane')
        set1 = QBarSet('John')
        set2 = QBarSet('Axel')
        set3 = QBarSet('Mary')
        set4 = QBarSet('Samantha')
        set0 << 1 << 2 << 3 << 4 << 5 << 6
        set1 << 5 << 0 << 0 << 4 << 0 << 7
        set2 << 3 << 5 << 8 << 13 << 8 << 5
        set3 << 5 << 6 << 7 << 3 << 4 << 5
        set4 << 9 << 7 << 5 << 3 << 1 << 2
        series = QHorizontalBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)
        series.append(set3)
        series.append(set4)
        return series
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())