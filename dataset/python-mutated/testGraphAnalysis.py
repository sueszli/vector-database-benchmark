"""
Created on 2019年8月17日
@author: weike32
@site: https://pyqt.site ,https://github.com/weike32
@email: 394967319@qq.com
@file: CopyContent
@description:
"""
import sys
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QScrollArea, QVBoxLayout
from PyQtGraph.Data.graphAnalysis import graph_Form

class CustomViewBox(pg.ViewBox):

    def __init__(self, *args, **kwds):
        if False:
            return 10
        pg.ViewBox.__init__(self, *args, **kwds)
        self.RectMode = 3
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if ev.button() == pg.QtCore.Qt.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev):
        if False:
            return 10
        pg.ViewBox.mouseDragEvent(self, ev)

    def wheelEvent(self, ev, axis=None):
        if False:
            for i in range(10):
                print('nop')
        ev.ignore()

class graphAnalysis(QDialog, graph_Form):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(graphAnalysis, self).__init__()
        self.setupUi(self)
        self.pushButton_7.clicked.connect(self.test)
        self.tabWidget.clear()

    def handleChanged(self, item, column):
        if False:
            i = 10
            return i + 15
        count = item.childCount()
        if item.checkState(column) == Qt.Checked:
            for index in range(count):
                item.child(index).setCheckState(0, Qt.Checked)
        if item.checkState(column) == Qt.Unchecked:
            for index in range(count):
                item.child(index).setCheckState(0, Qt.Unchecked)

    def test(self):
        if False:
            while True:
                i = 10
        tab1 = QWidget()
        scrollArea = QScrollArea(tab1)
        scrollArea.setMinimumSize(650, 550)
        scrollArea.setWidgetResizable(True)
        labelsContainer = QWidget()
        labelsContainer.setMinimumSize(0, 3000 + 200)
        scrollArea.setWidget(labelsContainer)
        layout = QVBoxLayout(labelsContainer)
        time = ['2019-04-20 08:09:00', '2019-04-20 08:09:00', '2019-04-20 08:09:00', '2019-04-20 08:09:00']
        value = [1.2, 2, 1, 4]
        xdict = dict(enumerate(time))
        ticks = [list(zip(range(4), tuple(time)))]
        for i in range(11):
            vb1 = CustomViewBox()
            plt1 = pg.PlotWidget(title='Basic array plotting%s' % i, viewBox=vb1)
            plt1.resize(500, 500)
            plt1.setBackground(background=None)
            plt1.plot(list(xdict.keys()), value)
            plt1.getPlotItem().getAxis('bottom').setTicks(ticks)
            temp1 = QWidget()
            temp1.setMinimumSize(600, 300)
            temp1.setMaximumSize(600, 300)
            layout2 = QVBoxLayout(temp1)
            layout2.addWidget(plt1)
            layout.addWidget(temp1)
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacerItem)
        self.tabWidget.addTab(tab1, '12')
        for i in range(self.tabWidget.count()):
            self.tabWidget.widget(i)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = graphAnalysis()
    w.show()
    sys.exit(app.exec_())