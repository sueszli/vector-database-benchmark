"""
Created on 2023/02/22
@author: Irony
@site: https://pyqt.site https://github.com/PyQt5
@email: 892768447@qq.com
@file: CoverItemWidget.py
@description:
"""
try:
    from PyQt5.QtCore import QSize, QUrl
    from PyQt5.QtGui import QPaintEvent, QPixmap
    from PyQt5.QtNetwork import QNetworkRequest
    from PyQt5.QtWidgets import QWidget
except ImportError:
    from PySide2.QtCore import QSize, QUrl
    from PySide2.QtGui import QPaintEvent, QPixmap
    from PySide2.QtNetwork import QNetworkRequest
    from PySide2.QtWidgets import QWidget
from .Ui_CoverItemWidget import Ui_CoverItemWidget

class CoverItemWidget(QWidget, Ui_CoverItemWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self._manager = kwargs.pop('manager', None)
        super(CoverItemWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)

    def init(self, cover_path, playlist_title, playlist_author, play_count, play_url, cover_url, img_path):
        if False:
            while True:
                i = 10
        self.img_path = img_path
        self.cover_url = cover_url
        self.labelCover.init(cover_path, play_url, play_count)
        self.labelTitle.setText(playlist_title)
        self.labelAuthor.setText(playlist_author)

    def setCover(self, path):
        if False:
            return 10
        self.labelCover.setCoverPath(path)
        self.labelCover.setPixmap(QPixmap(path))

    def sizeHint(self):
        if False:
            print('Hello World!')
        return QSize(200, 256)

    def event(self, event):
        if False:
            return 10
        if isinstance(event, QPaintEvent):
            if event.rect().height() > 20 and hasattr(self, 'labelCover'):
                if self.labelCover.cover_path.find('pic_v.png') > -1:
                    req = QNetworkRequest(QUrl(self.cover_url))
                    req.setAttribute(QNetworkRequest.User + 1, self)
                    req.setAttribute(QNetworkRequest.User + 2, self.img_path)
                    if self._manager:
                        self._manager.get(req)
        return super(CoverItemWidget, self).event(event)