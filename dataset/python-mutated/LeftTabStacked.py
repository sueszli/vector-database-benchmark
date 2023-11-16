"""
Created on 2018年5月29日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: LeftTabWidget
@description:
"""
from random import randint
try:
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QLabel
except ImportError:
    from PySide2.QtCore import Qt, QSize
    from PySide2.QtGui import QIcon
    from PySide2.QtWidgets import QApplication, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QListWidgetItem, QLabel

class LeftTabWidget(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(LeftTabWidget, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)
        self.stackedWidget = QStackedWidget(self)
        layout.addWidget(self.stackedWidget)
        self.initUi()

    def initUi(self):
        if False:
            i = 10
            return i + 15
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)
        self.listWidget.setFrameShape(QListWidget.NoFrame)
        self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for i in range(20):
            item = QListWidgetItem(QIcon('Data/0%d.ico' % randint(1, 8)), str('选 项 %s' % i), self.listWidget)
            item.setSizeHint(QSize(16777215, 60))
            item.setTextAlignment(Qt.AlignCenter)
        for i in range(20):
            label = QLabel('我是页面 %d' % i, self)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet('background: rgb(%d, %d, %d);margin: 50px;' % (randint(0, 255), randint(0, 255), randint(0, 255)))
            self.stackedWidget.addWidget(label)
Stylesheet = '\n/*去掉item虚线边框*/\nQListWidget, QListView, QTreeWidget, QTreeView {\n    outline: 0px;\n}\n/*设置左侧选项的最小最大宽度,文字颜色和背景颜色*/\nQListWidget {\n    min-width: 120px;\n    max-width: 120px;\n    color: white;\n    background: black;\n}\n/*被选中时的背景颜色和左边框颜色*/\nQListWidget::item:selected {\n    background: rgb(52, 52, 52);\n    border-left: 2px solid rgb(9, 187, 7);\n}\n/*鼠标悬停颜色*/\nHistoryPanel::item:hover {\n    background: rgb(52, 52, 52);\n}\n\n/*右侧的层叠窗口的背景颜色*/\nQStackedWidget {\n    background: rgb(30, 30, 30);\n}\n/*模拟的页面*/\nQLabel {\n    color: white;\n}\n'
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(Stylesheet)
    w = LeftTabWidget()
    w.show()
    sys.exit(app.exec_())