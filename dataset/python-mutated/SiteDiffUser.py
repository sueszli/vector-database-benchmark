"""
Created on 2019年8月23日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QWebEngineView.SiteDiffUser
@description: 同个网站不同用户
"""
import os
try:
    from PyQt5.QtCore import QUrl
    from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView
    from PyQt5.QtWidgets import QApplication, QTabWidget
except ImportError:
    from PySide2.QtCore import QUrl
    from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView
    from PySide2.QtWidgets import QApplication, QTabWidget

class Window(QTabWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        self.webView1 = QWebEngineView(self)
        profile1 = QWebEngineProfile('storage1', self.webView1)
        profile1.setPersistentStoragePath(os.path.abspath('Tmp/Storage1'))
        print(profile1.cookieStore())
        page1 = QWebEnginePage(profile1, self.webView1)
        self.webView1.setPage(page1)
        self.addTab(self.webView1, '用户1')
        self.webView2 = QWebEngineView(self)
        profile2 = QWebEngineProfile('storage2', self.webView2)
        profile2.setPersistentStoragePath(os.path.abspath('Tmp/Storage2'))
        print(profile2.cookieStore())
        page2 = QWebEnginePage(profile2, self.webView2)
        self.webView2.setPage(page2)
        self.addTab(self.webView2, '用户2')
        self.webView1.load(QUrl('https://v.qq.com'))
        self.webView2.load(QUrl('https://v.qq.com'))
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())