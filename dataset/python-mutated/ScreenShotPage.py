"""
Created on 2019年7月8日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ScreenShotPage
@description: 网页整体截图
"""
import base64
import cgitb
import sys
from PyQt5.QtCore import QUrl, Qt, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPainter, QIcon, QPixmap
from PyQt5.QtWebKit import QWebSettings
from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QGroupBox, QLineEdit, QHBoxLayout, QListWidget, QListWidgetItem, QProgressDialog
CODE = '\nvar el = $("%s");\nhtml2canvas(el[0], {\n    width: el.outerWidth(true), \n    windowWidth: el.outerWidth(true),\n}).then(function(canvas) {\n    _self.saveImage(canvas.toDataURL());\n});\n'

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        self.resize(600, 400)
        layout = QHBoxLayout(self)
        widgetLeft = QWidget(self)
        layoutLeft = QVBoxLayout(widgetLeft)
        self.widgetRight = QListWidget(self, minimumWidth=200, iconSize=QSize(150, 150))
        self.widgetRight.setViewMode(QListWidget.IconMode)
        layout.addWidget(widgetLeft)
        layout.addWidget(self.widgetRight)
        self.webView = QWebView()
        layoutLeft.addWidget(self.webView)
        groupBox1 = QGroupBox('截图方式一', self)
        layout1 = QVBoxLayout(groupBox1)
        layout1.addWidget(QPushButton('截图1', self, clicked=self.onScreenShot1))
        layoutLeft.addWidget(groupBox1)
        groupBox2 = QGroupBox('截图方式二', self)
        layout2 = QVBoxLayout(groupBox2)
        self.codeEdit = QLineEdit('body', groupBox2, placeholderText='请输入需要截图的元素、ID或者class：如body、#id .class')
        layout2.addWidget(self.codeEdit)
        self.btnMethod2 = QPushButton('', self, clicked=self.onScreenShot2, enabled=False)
        layout2.addWidget(self.btnMethod2)
        layoutLeft.addWidget(groupBox2)
        QWebSettings.globalSettings().setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        self.webView.loadStarted.connect(self.onLoadStarted)
        self.webView.loadFinished.connect(self.onLoadFinished)
        self.webView.load(QUrl('https://pyqt.site'))
        self.webView.page().mainFrame().javaScriptWindowObjectCleared.connect(self.populateJavaScriptWindowObject)

    def populateJavaScriptWindowObject(self):
        if False:
            return 10
        self.webView.page().mainFrame().addToJavaScriptWindowObject('_self', self)

    def onLoadStarted(self):
        if False:
            while True:
                i = 10
        print('load started')
        self.btnMethod2.setEnabled(False)
        self.btnMethod2.setText('暂时无法使用（等待页面加载完成）')

    def onLoadFinished(self):
        if False:
            while True:
                i = 10
        mainFrame = self.webView.page().mainFrame()
        mainFrame.evaluateJavaScript(open('Data/jquery.js', 'rb').read().decode())
        mainFrame.evaluateJavaScript(open('Data/promise-7.0.4.min.js', 'rb').read().decode())
        mainFrame.evaluateJavaScript(open('Data/html2canvas.min.js', 'rb').read().decode())
        print('inject js ok')
        self.btnMethod2.setText('截图2')
        self.btnMethod2.setEnabled(True)

    def onScreenShot1(self):
        if False:
            i = 10
            return i + 15
        page = self.webView.page()
        frame = page.mainFrame()
        size = frame.contentsSize()
        image = QImage(size, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        oldSize = page.viewportSize()
        page.setViewportSize(size)
        frame.render(painter)
        painter.end()
        page.setViewportSize(oldSize)
        item = QListWidgetItem(self.widgetRight)
        image = QPixmap.fromImage(image)
        item.setIcon(QIcon(image))
        item.setData(Qt.UserRole + 1, image)

    def onScreenShot2(self):
        if False:
            print('Hello World!')
        code = self.codeEdit.text().strip()
        if not code:
            return
        self.progressdialog = QProgressDialog(self, windowTitle='正在截图中')
        self.progressdialog.setRange(0, 0)
        self.webView.page().mainFrame().evaluateJavaScript(CODE % code)
        self.progressdialog.exec_()

    @pyqtSlot(str)
    def saveImage(self, image):
        if False:
            print('Hello World!')
        self.progressdialog.close()
        if not image.startswith('data:image'):
            return
        data = base64.b64decode(image.split(';base64,')[1])
        image = QPixmap()
        image.loadFromData(data)
        item = QListWidgetItem(self.widgetRight)
        item.setIcon(QIcon(image))
        item.setData(Qt.UserRole + 1, image)
if __name__ == '__main__':
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())