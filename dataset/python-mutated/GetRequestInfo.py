"""
Created on 2019年9月24日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QWebEngineView.BlockAds
@description: 拦截请求
"""
try:
    from PyQt5.QtCore import QUrl, QByteArray
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
    from PyQt5.QtWebEngineCore import QWebEngineUrlSchemeHandler, QWebEngineUrlScheme, QWebEngineUrlRequestInterceptor
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
except ImportError:
    from PySide2.QtCore import QUrl, QByteArray
    from PySide2.QtWidgets import QApplication
    from PySide2.QtNetwork import QNetworkAccessManager, QNetworkRequest
    from PySide2.QtWebEngineCore import QWebEngineUrlSchemeHandler, QWebEngineUrlScheme, QWebEngineUrlRequestInterceptor
    from PySide2.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile

class UrlSchemeHandler(QWebEngineUrlSchemeHandler):
    AttrType = QNetworkRequest.User + 1

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(UrlSchemeHandler, self).__init__(*args, **kwargs)
        self._manager = QNetworkAccessManager(self)
        self._manager.finished.connect(self.onFinished)

    def requestStarted(self, request):
        if False:
            while True:
                i = 10
        print('requestMethod:', request.requestMethod())
        print('requestHeaders:', request.requestHeaders())
        url = request.requestUrl()
        if url.scheme().startswith('myurl'):
            url.setScheme(url.scheme().replace('myurl', 'http'))
        print('requestUrl:', url)
        req = QNetworkRequest(url)
        req.setAttribute(self.AttrType, request)
        for (headerName, headerValue) in request.requestHeaders().items():
            req.setRawHeader(headerName, headerValue)
        method = request.requestMethod()
        if method == b'GET':
            self._manager.get(req)
        elif method == b'POST':
            self._manager.post(req)

    def onFinished(self, reply):
        if False:
            return 10
        req = reply.request()
        o_req = req.attribute(self.AttrType, None)
        if o_req:
            o_req.reply(req.header(QNetworkRequest.ContentTypeHeader) or b'text/html', reply)
            o_req.destroyed.connect(reply.deleteLater)

class RequestInterceptor(QWebEngineUrlRequestInterceptor):

    def interceptRequest(self, info):
        if False:
            return 10
        url = info.requestUrl()
        if url.scheme() == 'http':
            url.setScheme('myurl')
            info.redirect(url)
        elif url.scheme() == 'https':
            url.setScheme('myurls')
            info.redirect(url)

class Window(QWebEngineView):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        profile = QWebEngineProfile.defaultProfile()
        o_http = QWebEngineUrlScheme.schemeByName(QByteArray(b'http'))
        o_https = QWebEngineUrlScheme.schemeByName(QByteArray(b'https'))
        print('scheme:', o_http, o_https)
        CorsEnabled = 128
        o_http.setFlags(o_http.flags() | QWebEngineUrlScheme.SecureScheme | QWebEngineUrlScheme.LocalScheme | QWebEngineUrlScheme.LocalAccessAllowed | CorsEnabled)
        o_https.setFlags(o_https.flags() | QWebEngineUrlScheme.SecureScheme | QWebEngineUrlScheme.LocalScheme | QWebEngineUrlScheme.LocalAccessAllowed | CorsEnabled)
        de = QWebEngineProfile.defaultProfile()
        de.setRequestInterceptor(RequestInterceptor(self))
        self.urlSchemeHandler = UrlSchemeHandler(self)
        de.installUrlSchemeHandler(QByteArray(b'myurl'), self.urlSchemeHandler)
        de.installUrlSchemeHandler(QByteArray(b'myurls'), self.urlSchemeHandler)
if __name__ == '__main__':
    import sys
    import os
    import webbrowser
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '9966'
    webbrowser.open_new_tab('http://127.0.0.1:9966')
    w = Window()
    w.show()
    w.load(QUrl('https://pyqt.site'))
    sys.exit(app.exec_())