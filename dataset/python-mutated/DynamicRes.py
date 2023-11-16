"""
Created on 2020/6/3
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: DynamicRes
@description: 
"""
from threading import Thread
import requests
try:
    from PyQt5.QtCore import QUrl, QByteArray
    from PyQt5.QtGui import QImage, QTextDocument
    from PyQt5.QtWidgets import QApplication, QTextBrowser, QWidget, QVBoxLayout, QPushButton
except ImportError:
    from PySide2.QtCore import QUrl, QByteArray
    from PySide2.QtGui import QImage, QTextDocument
    from PySide2.QtWidgets import QApplication, QTextBrowser, QWidget, QVBoxLayout, QPushButton

class TextBrowser(QTextBrowser):
    NetImages = {}

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(TextBrowser, self).__init__(*args, **kwargs)
        self.setOpenLinks(False)

    def downloadImage(self, url):
        if False:
            i = 10
            return i + 15
        try:
            self.NetImages[url] = [QByteArray(requests.get(url.toString()).content), 1]
            print('下载完成', url)
        except Exception as e:
            print('下载失败', url, e)
            self.NetImages[url] = [QByteArray(), 1]

    def loadResource(self, rtype, url):
        if False:
            print('Hello World!')
        ret = super(TextBrowser, self).loadResource(rtype, url)
        if rtype == QTextDocument.ImageResource:
            if ret:
                return ret
            if url.toString().startswith('irony'):
                print('加载本地', '../Donate/zhifubao.png', url)
                return QImage('../Donate/zhifubao.png')
            elif url.toString().startswith('http'):
                (img, status) = self.NetImages.get(url, [None, None])
                if url not in self.NetImages or status is None:
                    self.NetImages[url] = [None, 1]
                    print('download ', url)
                    Thread(target=self.downloadImage, args=(url,), daemon=True).start()
                elif img:
                    return img
        return ret

    def mouseDoubleClickEvent(self, event):
        if False:
            print('Hello World!')
        super(TextBrowser, self).mouseDoubleClickEvent(event)
        url = self.anchorAt(event.pos())
        if url:
            print('url:', url, self.document().resource(QTextDocument.ImageResource, QUrl(url)))

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textBrowser = TextBrowser(self)
        self.downButton = QPushButton('加载网络图片', self)
        layout.addWidget(self.textBrowser)
        layout.addWidget(self.downButton)
        img = QImage('../Donate/weixin.png')
        self.textBrowser.document().addResource(QTextDocument.ImageResource, QUrl('dynamic:/images/weixin.png'), img)
        self.textBrowser.setHtml('<p><a href="../Donate/weixin.png"><img src="../Donate/weixin.png"></a></p><p><a href="dynamic:/images/weixin.png"><img src="dynamic:/images/weixin.png"></a></p><p><a href="irony://zhifubao.png"><img src="irony://zhifubao.png"></a></p><p><a href="https://blog.pyqt.site/img/avatar.png"><img src="https://blog.pyqt.site/img/avatar.png"></a></p>')
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())