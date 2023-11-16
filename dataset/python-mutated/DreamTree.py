"""
Created on 2017年4月6日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: DreamTree
@description: 
"""
import sys
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPalette
from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        palette = self.palette()
        palette.setBrush(QPalette.Base, Qt.transparent)
        self.setPalette(palette)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.webView = QWebView(self)
        layout.addWidget(self.webView)
        self.webView.setContextMenuPolicy(Qt.NoContextMenu)
        self.mainFrame = self.webView.page().mainFrame()
        self.mainFrame.setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAlwaysOff)
        self.mainFrame.setScrollBarPolicy(Qt.Horizontal, Qt.ScrollBarAlwaysOff)
        rect = app.desktop().availableGeometry()
        self.resize(rect.size())
        self.webView.resize(rect.size())

    def load(self):
        if False:
            i = 10
            return i + 15
        self.webView.load(QUrl('qrc:/tree.html'))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    w.load()
    sys.exit(app.exec_())