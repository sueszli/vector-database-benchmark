from typing import Optional
from AnyQt.QtCore import Qt, QSizeF, QRectF, QPointF
from AnyQt.QtGui import QPixmap, QTransform, QPainter
from AnyQt.QtWidgets import QGraphicsWidget, QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from Orange.widgets.utils.graphicslayoutitem import scaled

class GraphicsPixmapWidget(QGraphicsWidget):

    def __init__(self, parent: Optional[QGraphicsItem]=None, pixmap: Optional[QPixmap]=None, scaleContents=False, aspectMode=Qt.KeepAspectRatio, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__scaleContents = scaleContents
        self.__aspectMode = aspectMode
        self.__pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemUsesExtendedStyleOption, True)
        self.setContentsMargins(0, 0, 0, 0)
        if parent is not None:
            self.setParentItem(parent)

    def setPixmap(self, pixmap: QPixmap) -> None:
        if False:
            while True:
                i = 10
        self.prepareGeometryChange()
        self.__pixmap = QPixmap(pixmap)
        self.updateGeometry()

    def pixmap(self) -> QPixmap:
        if False:
            print('Hello World!')
        return QPixmap(self.__pixmap)

    def setAspectRatioMode(self, mode: Qt.AspectRatioMode) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.__aspectMode != mode:
            self.__aspectMode = mode
            sp = self.sizePolicy()
            sp.setHeightForWidth(self.__aspectMode != Qt.IgnoreAspectRatio and self.__scaleContents)
            self.setSizePolicy(sp)
            self.updateGeometry()

    def aspectRatioMode(self) -> Qt.AspectRatioMode:
        if False:
            return 10
        return self.__aspectMode

    def setScaleContents(self, scale: bool) -> None:
        if False:
            return 10
        if self.__scaleContents != scale:
            self.__scaleContents = bool(scale)
            sp = self.sizePolicy()
            sp.setHeightForWidth(self.__aspectMode != Qt.IgnoreAspectRatio and self.__scaleContents)
            self.setSizePolicy(sp)
            self.updateGeometry()

    def scaleContents(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.__scaleContents

    def sizeHint(self, which, constraint=QSizeF(-1, -1)) -> QSizeF:
        if False:
            for i in range(10):
                print('nop')
        if which == Qt.PreferredSize:
            sh = QSizeF(self.__pixmap.size())
            if self.__scaleContents:
                sh = scaled(sh, constraint, self.__aspectMode)
            return sh
        elif which == Qt.MinimumSize:
            if self.__scaleContents:
                return QSizeF(0, 0)
            else:
                return QSizeF(self.__pixmap.size())
        elif which == Qt.MaximumSize:
            if self.__scaleContents:
                return QSizeF()
            else:
                return QSizeF(self.__pixmap.size())
        else:
            return QSizeF()

    def pixmapTransform(self) -> QTransform:
        if False:
            while True:
                i = 10
        if self.__pixmap.isNull():
            return QTransform()
        pxsize = QSizeF(self.__pixmap.size())
        crect = self.contentsRect()
        transform = QTransform()
        transform = transform.translate(crect.left(), crect.top())
        if self.__scaleContents:
            csize = scaled(pxsize, crect.size(), self.__aspectMode)
        else:
            csize = pxsize
        xscale = csize.width() / pxsize.width()
        yscale = csize.height() / pxsize.height()
        return transform.scale(xscale, yscale)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget]=None) -> None:
        if False:
            return 10
        if self.__pixmap.isNull():
            return
        pixmap = self.__pixmap
        crect = self.contentsRect()
        exposed = option.exposedRect
        exposedcrect = crect.intersected(exposed)
        pixmaptransform = self.pixmapTransform()
        assert pixmaptransform.type() in (QTransform.TxNone, QTransform.TxTranslate, QTransform.TxScale)
        (pixmaptransform, ok) = pixmaptransform.inverted()
        if not ok:
            painter.drawPixmap(crect, pixmap, QRectF(QPointF(0, 0), QSizeF(pixmap.size())))
        else:
            exposedpixmap = pixmaptransform.mapRect(exposed)
            painter.drawPixmap(exposedcrect, pixmap, exposedpixmap)