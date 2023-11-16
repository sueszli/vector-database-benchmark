import warnings
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtWidgets import QLabel, QSizePolicy, QMenu
from feeluown.gui.drawers import PixmapDrawer
from feeluown.gui.image import open_image

class CoverLabel(QLabel):

    def __init__(self, parent=None, pixmap=None, radius=3):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self._radius = radius
        self.drawer = None
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

    def show_pixmap(self, pixmap):
        if False:
            print('Hello World!')
        '\n        .. versiondeprecated:: 3.8.11\n        '
        warnings.warn('You should use show_img', DeprecationWarning)
        self.updateGeometry()
        self.update()

    def show_img(self, img: QImage):
        if False:
            while True:
                i = 10
        if not img or img.isNull():
            self.drawer = None
            return
        self.drawer = PixmapDrawer(img, self, self._radius)
        self.updateGeometry()
        self.update()

    def paintEvent(self, e):
        if False:
            return 10
        '\n        draw pixmap with border radius\n\n        We found two way to draw pixmap with border radius,\n        one is as follow, the other way is using bitmap mask,\n        but in our practice, the mask way has poor render effects\n        '
        if self.drawer:
            painter = QPainter(self)
            self.drawer.draw(painter)

    def contextMenuEvent(self, e):
        if False:
            print('Hello World!')
        if self.drawer is None:
            return
        menu = QMenu()
        action = menu.addAction('查看原图')
        action.triggered.connect(lambda : open_image(self.drawer.get_img()))
        menu.exec(e.globalPos())

    def resizeEvent(self, e):
        if False:
            print('Hello World!')
        super().resizeEvent(e)
        self.updateGeometry()

    def sizeHint(self):
        if False:
            print('Hello World!')
        super_size = super().sizeHint()
        if self.drawer is None:
            return super_size
        h = self.width() * self.drawer.get_pixmap().height() // self.drawer.get_pixmap().width()
        w = self.width()
        return QSize(w, min(w, h))

class CoverLabelV2(CoverLabel):
    """

    .. versionadded:: 3.7.8
    """

    def __init__(self, app, parent=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent, **kwargs)
        self._app = app

    async def show_cover(self, url, cover_uid):
        content = await self._app.img_mgr.get(url, cover_uid)
        img = QImage()
        img.loadFromData(content)
        self.show_img(img)