"""
Created on 2020年2月18日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: BlockRequestData
@description: 拦截请求内容
"""
try:
    from PyQt5.QtCore import QUrl, QFile, QIODevice, QByteArray
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWebEngineCore import QWebEngineUrlSchemeHandler, QWebEngineUrlRequestInterceptor, QWebEngineUrlScheme
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
except ImportError:
    from PySide2.QtCore import QUrl, QFile, QIODevice, QByteArray
    from PySide2.QtWidgets import QApplication
    from PySide2.QtWebEngineCore import QWebEngineUrlSchemeHandler, QWebEngineUrlRequestInterceptor, QWebEngineUrlScheme
    from PySide2.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile

class UrlSchemeHandler(QWebEngineUrlSchemeHandler):

    def requestStarted(self, job):
        if False:
            return 10
        url = job.requestUrl().toString()
        if url == 'myurl://png':
            file = QFile('Data/app.png', job)
            file.open(QIODevice.ReadOnly)
            job.reply(b'image/png', file)

class RequestInterceptor(QWebEngineUrlRequestInterceptor):

    def interceptRequest(self, info):
        if False:
            print('Hello World!')
        url = info.requestUrl().toString()
        if url.endswith('.png'):
            info.redirect(QUrl('myurl://png'))

class Window(QWebEngineView):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        h1 = QWebEngineUrlScheme.schemeByName(QByteArray(b'http'))
        h2 = QWebEngineUrlScheme.schemeByName(QByteArray(b'https'))
        CorsEnabled = 128
        h1.setFlags(h1.flags() | QWebEngineUrlScheme.SecureScheme | QWebEngineUrlScheme.LocalScheme | QWebEngineUrlScheme.LocalAccessAllowed | CorsEnabled)
        h2.setFlags(h2.flags() | QWebEngineUrlScheme.SecureScheme | QWebEngineUrlScheme.LocalScheme | QWebEngineUrlScheme.LocalAccessAllowed | CorsEnabled)
        de = QWebEngineProfile.defaultProfile()
        de.setRequestInterceptor(RequestInterceptor(self))
        de.installUrlSchemeHandler(QByteArray(b'myurl'), UrlSchemeHandler(self))
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    w.load(QUrl('https://www.baidu.com/'))
    sys.exit(app.exec_())