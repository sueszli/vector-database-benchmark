"""
Created on 2019年4月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QWebEngineView.JsSignals
@description: 
"""
import os
from time import time
from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal
from PyQt5.QtWebKit import QWebSettings
from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QPushButton

class WebView(QWebView):
    customSignal = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(WebView, self).__init__(*args, **kwargs)
        self.initSettings()
        self.page().mainFrame().javaScriptWindowObjectCleared.connect(self._exposeInterface)

    def _exposeInterface(self):
        if False:
            print('Hello World!')
        '向Js暴露调用本地方法接口\n        '
        self.page().mainFrame().addToJavaScriptWindowObject('Bridge', self)

    @pyqtSlot(str)
    def callFromJs(self, text):
        if False:
            for i in range(10):
                print('nop')
        QMessageBox.information(self, '提示', '来自js调用：{}'.format(text))

    def sendCustomSignal(self):
        if False:
            while True:
                i = 10
        self.customSignal.emit('当前时间: ' + str(time()))

    @pyqtSlot(str)
    @pyqtSlot(QUrl)
    def load(self, url):
        if False:
            for i in range(10):
                print('nop')
        '\n        eg: load("https://pyqt.site")\n        :param url: 网址\n        '
        return super(WebView, self).load(QUrl(url))

    def initSettings(self):
        if False:
            print('Hello World!')
        '\n        eg: 初始化设置\n        '
        settings = self.settings()
        settings.setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        settings.setDefaultTextEncoding('UTF-8')

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.webview = WebView(self)
        layout.addWidget(self.webview)
        layout.addWidget(QPushButton('发送自定义信号', self, clicked=self.webview.sendCustomSignal))
        self.webview.windowTitleChanged.connect(self.setWindowTitle)
        self.webview.load(QUrl.fromLocalFile(os.path.abspath('Data/JsSignals.html')))
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    w.move(100, 100)
    sys.exit(app.exec_())