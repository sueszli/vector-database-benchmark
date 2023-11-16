import warnings
from typing import Union
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPixmap, QPainterPath
from PyQt5.QtWidgets import QLabel, QApplication, QWidget
try:
    from ...common.image_utils import gaussianBlur
    isAcrylicAvailable = True
except ImportError as e:
    isAcrylicAvailable = False

    def gaussianBlur(imagePath, blurRadius=18, brightFactor=1, blurPicSize=None):
        if False:
            print('Hello World!')
        return QPixmap(imagePath)

def checkAcrylicAvailability():
    if False:
        for i in range(10):
            print('nop')
    if not isAcrylicAvailable:
        warnings.warn('`AcrylicLabel` is not supported in current qfluentwidgets, use `pip install PyQt-Fluent-Widgets[full]` to enable it.')
    return isAcrylicAvailable

class BlurCoverThread(QThread):
    """ Blur album cover thread """
    blurFinished = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.imagePath = ''
        self.blurRadius = 7
        self.maxSize = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.imagePath:
            return
        pixmap = gaussianBlur(self.imagePath, self.blurRadius, 0.85, self.maxSize)
        self.blurFinished.emit(pixmap)

    def blur(self, imagePath: str, blurRadius=6, maxSize: tuple=(450, 450)):
        if False:
            i = 10
            return i + 15
        self.imagePath = imagePath
        self.blurRadius = blurRadius
        self.maxSize = maxSize or self.maxSize
        self.start()

class AcrylicTextureLabel(QLabel):
    """ Acrylic texture label """

    def __init__(self, tintColor: QColor, luminosityColor: QColor, noiseOpacity=0.03, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        tintColor: QColor\n            RGB tint color\n\n        luminosityColor: QColor\n            luminosity layer color\n\n        noiseOpacity: float\n            noise layer opacity\n\n        parent:\n            parent window\n        '
        super().__init__(parent=parent)
        self.tintColor = QColor(tintColor)
        self.luminosityColor = QColor(luminosityColor)
        self.noiseOpacity = noiseOpacity
        self.noiseImage = QImage(':/qfluentwidgets/images/acrylic/noise.png')
        self.setAttribute(Qt.WA_TranslucentBackground)

    def setTintColor(self, color: QColor):
        if False:
            for i in range(10):
                print('nop')
        self.tintColor = color
        self.update()

    def paintEvent(self, e):
        if False:
            return 10
        acrylicTexture = QImage(64, 64, QImage.Format_ARGB32_Premultiplied)
        acrylicTexture.fill(self.luminosityColor)
        painter = QPainter(acrylicTexture)
        painter.fillRect(acrylicTexture.rect(), self.tintColor)
        painter.setOpacity(self.noiseOpacity)
        painter.drawImage(acrylicTexture.rect(), self.noiseImage)
        acrylicBrush = QBrush(acrylicTexture)
        painter = QPainter(self)
        painter.fillRect(self.rect(), acrylicBrush)

class AcrylicLabel(QLabel):
    """ Acrylic label """

    def __init__(self, blurRadius: int, tintColor: QColor, luminosityColor=QColor(255, 255, 255, 0), maxBlurSize: tuple=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        blurRadius: int\n            blur radius\n\n        tintColor: QColor\n            tint color\n\n        luminosityColor: QColor\n            luminosity layer color\n\n        maxBlurSize: tuple\n            maximum image size\n\n        parent:\n            parent window\n        '
        super().__init__(parent=parent)
        checkAcrylicAvailability()
        self.imagePath = ''
        self.blurPixmap = QPixmap()
        self.blurRadius = blurRadius
        self.maxBlurSize = maxBlurSize
        self.acrylicTextureLabel = AcrylicTextureLabel(tintColor, luminosityColor, parent=self)
        self.blurThread = BlurCoverThread(self)
        self.blurThread.blurFinished.connect(self.__onBlurFinished)

    def __onBlurFinished(self, blurPixmap: QPixmap):
        if False:
            while True:
                i = 10
        ' blur finished slot '
        self.blurPixmap = blurPixmap
        self.setPixmap(self.blurPixmap)
        self.adjustSize()

    def setImage(self, imagePath: str):
        if False:
            i = 10
            return i + 15
        ' set the image to be blurred '
        self.imagePath = imagePath
        self.blurThread.blur(imagePath, self.blurRadius, self.maxBlurSize)

    def setTintColor(self, color: QColor):
        if False:
            for i in range(10):
                print('nop')
        self.acrylicTextureLabel.setTintColor(color)

    def resizeEvent(self, e):
        if False:
            i = 10
            return i + 15
        super().resizeEvent(e)
        self.acrylicTextureLabel.resize(self.size())
        if not self.blurPixmap.isNull() and self.blurPixmap.size() != self.size():
            self.setPixmap(self.blurPixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

class AcrylicBrush:
    """ Acrylic brush """

    def __init__(self, device: QWidget, blurRadius: int, tintColor=QColor(242, 242, 242, 150), luminosityColor=QColor(255, 255, 255, 10), noiseOpacity=0.03):
        if False:
            while True:
                i = 10
        self.device = device
        self.blurRadius = blurRadius
        self.tintColor = QColor(tintColor)
        self.luminosityColor = QColor(luminosityColor)
        self.noiseOpacity = noiseOpacity
        self.noiseImage = QImage(':/qfluentwidgets/images/acrylic/noise.png')
        self.originalImage = QPixmap()
        self.image = QPixmap()
        self.clipPath = QPainterPath()

    def setBlurRadius(self, radius: int):
        if False:
            for i in range(10):
                print('nop')
        if radius == self.blurRadius:
            return
        self.blurRadius = radius
        self.setImage(self.originalImage)

    def setTintColor(self, color: QColor):
        if False:
            return 10
        self.tintColor = QColor(color)
        self.device.update()

    def setLuminosityColor(self, color: QColor):
        if False:
            return 10
        self.luminosityColor = QColor(color)
        self.device.update()

    def isAvailable(self):
        if False:
            i = 10
            return i + 15
        return isAcrylicAvailable

    def grabImage(self, rect: QRect):
        if False:
            while True:
                i = 10
        ' grab image from screen\n\n        Parameters\n        ----------\n        rect: QRect\n            grabbed region\n        '
        screen = QApplication.screenAt(self.device.window().pos())
        if not screen:
            screen = QApplication.screens()[0]
        (x, y, w, h) = (rect.x(), rect.y(), rect.width(), rect.height())
        self.setImage(screen.grabWindow(0, x, y, w, h))

    def setImage(self, image: Union[str, QImage, QPixmap]):
        if False:
            for i in range(10):
                print('nop')
        ' set blurred image '
        if isinstance(image, str):
            image = QPixmap(image)
        elif isinstance(image, QImage):
            image = QPixmap.fromImage(image)
        self.originalImage = image
        if not image.isNull():
            checkAcrylicAvailability()
            self.image = gaussianBlur(image, self.blurRadius)
        self.device.update()

    def setClipPath(self, path: QPainterPath):
        if False:
            while True:
                i = 10
        self.clipPath = path
        self.device.update()

    def textureImage(self):
        if False:
            i = 10
            return i + 15
        texture = QImage(64, 64, QImage.Format_ARGB32_Premultiplied)
        texture.fill(self.luminosityColor)
        painter = QPainter(texture)
        painter.fillRect(texture.rect(), self.tintColor)
        painter.setOpacity(self.noiseOpacity)
        painter.drawImage(texture.rect(), self.noiseImage)
        return texture

    def paint(self):
        if False:
            i = 10
            return i + 15
        device = self.device
        painter = QPainter(device)
        painter.setRenderHints(QPainter.Antialiasing)
        if not self.clipPath.isEmpty():
            painter.setClipPath(self.clipPath)
        image = self.image.scaled(device.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, image)
        painter.fillRect(device.rect(), QBrush(self.textureImage()))