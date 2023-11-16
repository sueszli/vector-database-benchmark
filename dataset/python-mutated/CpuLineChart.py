"""
Created on 2021/5/13
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CpuLineChart
@description: 
"""
import sys
from PyQt5.QtChart import QChartView, QChart, QSplineSeries, QDateTimeAxis, QValueAxis
from PyQt5.QtCore import Qt, QTimer, QDateTime, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication
from psutil import cpu_percent

class CpuLineChart(QChart):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(CpuLineChart, self).__init__(*args, **kwargs)
        self.m_count = 10
        self.legend().hide()
        self.m_series = QSplineSeries(self)
        self.m_series.setPen(QPen(QColor('#3B8CFF'), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.addSeries(self.m_series)
        self.m_axisX = QDateTimeAxis(self)
        self.m_axisX.setTickCount(self.m_count + 1)
        self.m_axisX.setFormat('hh:mm:ss')
        now = QDateTime.currentDateTime()
        self.m_axisX.setRange(now.addSecs(-self.m_count), now)
        self.addAxis(self.m_axisX, Qt.AlignBottom)
        self.m_series.attachAxis(self.m_axisX)
        self.m_axisY = QValueAxis(self)
        self.m_axisY.setLabelFormat('%d')
        self.m_axisY.setMinorTickCount(4)
        self.m_axisY.setTickCount(self.m_count + 1)
        self.m_axisY.setRange(0, 100)
        self.addAxis(self.m_axisY, Qt.AlignLeft)
        self.m_series.attachAxis(self.m_axisY)
        self.m_series.append([QPointF(now.addSecs(-i).toMSecsSinceEpoch(), 0) for i in range(self.m_count, -1, -1)])
        self.m_timer = QTimer()
        self.m_timer.timeout.connect(self.update_data)
        self.m_timer.start(1000)

    def update_data(self):
        if False:
            for i in range(10):
                print('nop')
        value = cpu_percent()
        now = QDateTime.currentDateTime()
        self.m_axisX.setRange(now.addSecs(-self.m_count), now)
        points = self.m_series.pointsVector()
        points.pop(0)
        points.append(QPointF(now.toMSecsSinceEpoch(), value))
        self.m_series.replace(points)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    chart = CpuLineChart()
    chart.setTitle('cpu')
    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing)
    view.resize(800, 600)
    view.show()
    sys.exit(app.exec_())