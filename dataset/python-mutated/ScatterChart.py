"""
Created on 2019/10/2
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ScatterChart
@description: 散点图表
"""
import random
try:
    from PyQt5.QtChart import QChartView, QChart, QScatterSeries
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPainter
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PySide2.QtCore import QPointF
    from PySide2.QtGui import QPainter
    from PySide2.QtWidgets import QApplication
    from PySide2.QtCharts import QtCharts
    QChartView = QtCharts.QChartView
    QChart = QtCharts.QChart
    QScatterSeries = QtCharts.QScatterSeries

class Window(QChartView):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 300)
        self.setRenderHint(QPainter.Antialiasing)
        self.m_dataTable = self.generateRandomData(3, 10, 7)
        chart = QChart()
        self.setChart(chart)
        chart.setTitle('Scatter chart')
        self.getSeries(chart)
        chart.createDefaultAxes()
        chart.legend().setVisible(False)

    def getSeries(self, chart):
        if False:
            for i in range(10):
                print('nop')
        for (i, data_list) in enumerate(self.m_dataTable):
            series = QScatterSeries(chart)
            for (value, _) in data_list:
                series.append(value)
            series.setName('Series ' + str(i))
            chart.addSeries(series)

    def generateRandomData(self, listCount, valueMax, valueCount):
        if False:
            return 10
        random.seed()
        dataTable = []
        for i in range(listCount):
            dataList = []
            yValue = 0.0
            f_valueCount = float(valueCount)
            for j in range(valueCount):
                yValue += random.uniform(0, valueMax) / f_valueCount
                value = QPointF(j + random.random() * valueMax / f_valueCount, yValue)
                label = 'Slice ' + str(i) + ':' + str(j)
                dataList.append((value, label))
            dataTable.append(dataList)
        return dataTable
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())