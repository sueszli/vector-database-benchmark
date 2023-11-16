import sys
import cgitb
sys.excepthook = cgitb.Hook(1, None, 5, sys.stderr, 'text')
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from multiprocessing import Process, Pool

def runPool(i):
    if False:
        return 10
    print(i)
    t = py_process()
    t.run()

class SWebEngineView(QWebEngineView):
    """
    浏览器类。
    """

    def __init__(self, parent=None, url=''):
        if False:
            print('Hello World!')
        super(SWebEngineView, self).__init__()
        self.parent = parent
        self.url = url
        self.tempurl = ''
        self.loadFinished.connect(self.gethtml)
        self.show()
        self.a = 0

    def gethtml(self, *a, **b):
        if False:
            for i in range(10):
                print('nop')
        self.a += 1
        print('times:', self.a, '--', self.page().url())

    def closeEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.deleteLater()

    def clickLieBiao(self):
        if False:
            i = 10
            return i + 15
        print('end')
        self.page().runJavaScript('$("#alarmtitle").text()', self.get_title)
        self.page().runJavaScript('$("#alarmcontent").text()', self.get_content)
        self.page().runJavaScript('$("div.RecoveryDirectoryNav").text()', self.get_datetime)
        self.page().runJavaScript('$("#alarmimg").attr("src")', self.get_img)

    def get_title(self, balance):
        if False:
            for i in range(10):
                print('nop')
        self.appInList(balance)

    def get_content(self, balance):
        if False:
            i = 10
            return i + 15
        self.appInList(balance)

    def get_datetime(self, balance):
        if False:
            i = 10
            return i + 15
        self.appInList(balance)

    def get_img(self, balance):
        if False:
            while True:
                i = 10
        self.appInList(balance)

    def appInList(self, blance):
        if False:
            while True:
                i = 10
        print(blance)

class py_process(Process):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(py_process, self).__init__()
        print('1')
        self.url = 'https://siteserver.progressivedirect.com/session/setidredirect/?&product=AU&statecode=DC&type=New&refer=PGRX&URL=https://qad.progressivedirect.com/ApplicationStart.aspx?Page=Create&OfferingID=DC&state=DC&zip=20007&SessionStart=True'
        self.app = QApplication(sys.argv)
        self.browser = SWebEngineView(self, self.url)

    def run(self):
        if False:
            return 10
        print('run1')
        self.browser.setUrl(QUrl(self.url))
        print('run2')
        self.app.exec_()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    import os
    os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '9000'
    url = 'http://www.progressive.com'
    browser = SWebEngineView(url)
    print('run1')
    browser.setUrl(QUrl(url))
    print('run2')
    sys.exit(app.exec_())