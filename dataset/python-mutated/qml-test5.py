from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QRectF, Qt, QUrl
from PyQt5.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt5.QtQml import qmlRegisterType
from PyQt5.QtQuick import QQuickPaintedItem, QQuickView

class PieChart(QQuickPaintedItem):
    chartCleared = pyqtSignal()

    @pyqtProperty(str)
    def name(self):
        if False:
            print('Hello World!')
        return self._name

    @name.setter
    def name(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._name = name

    @pyqtProperty(QColor)
    def color(self):
        if False:
            print('Hello World!')
        return self._color

    @color.setter
    def color(self, color):
        if False:
            print('Hello World!')
        self._color = QColor(color)

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super(PieChart, self).__init__(parent)
        self._name = ''
        self._color = QColor()

    def paint(self, painter):
        if False:
            while True:
                i = 10
        painter.setPen(QPen(self._color, 2))
        painter.setRenderHints(QPainter.Antialiasing, True)
        rect = QRectF(0, 0, self.width(), self.height()).adjusted(1, 1, -1, -1)
        painter.drawPie(rect, 90 * 16, 290 * 16)

    @pyqtSlot()
    def clearChart(self):
        if False:
            for i in range(10):
                print('nop')
        self.color = QColor(Qt.transparent)
        self.update()
        self.chartCleared.emit()
if __name__ == '__main__':
    import os
    import sys
    app = QGuiApplication(sys.argv)
    qmlRegisterType(PieChart, 'Charts', 1, 0, 'PieChart')
    view = QQuickView()
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    view.setSource(QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), 'test5.qml')))
    view.show()
    sys.exit(app.exec_())