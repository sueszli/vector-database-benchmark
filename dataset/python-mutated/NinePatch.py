"""
Created on 2018年10月25日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: NinePatch
@description: 
"""
from math import fabs
try:
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QImage, QColor, QPainter, qRed, qGreen, qBlue, qAlpha
except ImportError:
    from PySide2.QtCore import QRect
    from PySide2.QtGui import QImage, QColor, QPainter, qRed, qGreen, qBlue, qAlpha

class _Exception(Exception):

    def __init__(self, imgW, imgH):
        if False:
            for i in range(10):
                print('nop')
        self.imgW = imgW
        self.imgH = imgH

class NinePatchException(Exception):

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Nine patch error'

class ExceptionIncorrectWidth(_Exception):

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Input incorrect width. Mimimum width = :{imgW}'.format(imgW=self.imgW)

class ExceptionIncorrectWidthAndHeight(_Exception):

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Input incorrect width width and height. Minimum width = :{imgW} . Minimum height = :{imgH}'.format(imgW=self.imgW, imgH=self.imgH)

class ExceptionIncorrectHeight(_Exception):

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Input incorrect height. Minimum height = :{imgW}'.format(imgW=self.imgW)

class ExceptionNot9Patch(Exception):

    def __str__(self):
        if False:
            return 10
        return 'It is not nine patch image'

class NinePatch:

    def __init__(self, fileName):
        if False:
            for i in range(10):
                print('nop')
        self.CachedImage = None
        self.OldWidth = -1
        self.OldHeight = -1
        self.ResizeDistancesX = []
        self.ResizeDistancesY = []
        self.setImage(fileName)

    def width(self):
        if False:
            i = 10
            return i + 15
        return self.Image.width()

    def height(self):
        if False:
            return 10
        return self.Image.height()

    def setImage(self, fileName):
        if False:
            print('Hello World!')
        self.Image = QImage(fileName)
        if self.Image.isNull():
            return
        self.ContentArea = self.GetContentArea()
        self.GetResizeArea()
        if not self.ResizeDistancesX or not self.ResizeDistancesY:
            raise ExceptionNot9Patch

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'CachedImage'):
            del self.CachedImage
        if hasattr(self, 'Image'):
            del self.Image

    def Draw(self, painter, x, y):
        if False:
            print('Hello World!')
        painter.drawImage(x, y, self.CachedImage)

    def SetImageSize(self, width, height):
        if False:
            while True:
                i = 10
        resizeWidth = 0
        resizeHeight = 0
        for i in range(len(self.ResizeDistancesX)):
            resizeWidth += self.ResizeDistancesX[i][1]
        for i in range(len(self.ResizeDistancesY)):
            resizeHeight += self.ResizeDistancesY[i][1]
        if width < self.Image.width() - 2 - resizeWidth and height < self.Image.height() - 2 - resizeHeight:
            raise ExceptionIncorrectWidthAndHeight(self.Image.width() - 2, self.Image.height() - 2)
        if width < self.Image.width() - 2 - resizeWidth:
            raise ExceptionIncorrectWidth(self.Image.width() - 2, self.Image.height() - 2)
        if height < self.Image.height() - 2 - resizeHeight:
            raise ExceptionIncorrectHeight(self.Image.width() - 2, self.Image.height() - 2)
        if width != self.OldWidth or height != self.OldHeight:
            self.OldWidth = width
            self.OldHeight = height
            self.UpdateCachedImage(width, height)

    @classmethod
    def GetContentAreaRect(self, width, height):
        if False:
            while True:
                i = 10
        return QRect(self.ContentArea.x(), self.ContentArea.y(), width - (self.Image.width() - 2 - self.ContentArea.width()), height - (self.Image.height() - 2 - self.ContentArea.height()))

    def DrawScaledPart(self, oldRect, newRect, painter):
        if False:
            while True:
                i = 10
        if newRect.width() and newRect.height():
            img = self.Image.copy(oldRect)
            img = img.scaled(newRect.width(), newRect.height())
            painter.drawImage(newRect.x(), newRect.y(), img, 0, 0, newRect.width(), newRect.height())

    def DrawConstPart(self, oldRect, newRect, painter):
        if False:
            i = 10
            return i + 15
        img = self.Image.copy(oldRect)
        painter.drawImage(newRect.x(), newRect.y(), img, 0, 0, newRect.width(), newRect.height())

    def IsColorBlack(self, color):
        if False:
            while True:
                i = 10
        r = qRed(color)
        g = qGreen(color)
        b = qBlue(color)
        a = qAlpha(color)
        if a < 128:
            return False
        return r < 128 and g < 128 and (b < 128)

    def GetContentArea(self):
        if False:
            for i in range(10):
                print('nop')
        j = self.Image.height() - 1
        left = 0
        right = 0
        for i in range(self.Image.width()):
            if self.IsColorBlack(self.Image.pixel(i, j)) and left == 0:
                left = i
            elif left != 0 and self.IsColorBlack(self.Image.pixel(i, j)):
                right = i
        if left and (not right):
            right = left
        left -= 1
        i = self.Image.width() - 1
        top = 0
        bot = 0
        for j in range(self.Image.height()):
            if self.IsColorBlack(self.Image.pixel(i, j)) and top == 0:
                top = j
            elif top and self.IsColorBlack(self.Image.pixel(i, j)):
                bot = j
        if top and (not bot):
            bot = top
        top -= 1
        return QRect(left, top, right - left, bot - top)

    def GetResizeArea(self):
        if False:
            while True:
                i = 10
        j = 0
        left = 0
        right = 0
        for i in range(self.Image.width()):
            if self.IsColorBlack(self.Image.pixel(i, j)) and left == 0:
                left = i
            if left and self.IsColorBlack(self.Image.pixel(i, j)) and (not self.IsColorBlack(self.Image.pixel(i + 1, j))):
                right = i
                left -= 1
                self.ResizeDistancesX.append((left, right - left))
                right = 0
                left = 0
        i = 0
        top = 0
        bot = 0
        for j in range(self.Image.height()):
            if self.IsColorBlack(self.Image.pixel(i, j)) and top == 0:
                top = j
            if top and self.IsColorBlack(self.Image.pixel(i, j)) and (not self.IsColorBlack(self.Image.pixel(i, j + 1))):
                bot = j
                top -= 1
                self.ResizeDistancesY.append((top, bot - top))
                top = 0
                bot = 0

    def GetFactor(self, width, height, factorX, factorY):
        if False:
            print('Hello World!')
        topResize = width - (self.Image.width() - 2)
        leftResize = height - (self.Image.height() - 2)
        for i in range(len(self.ResizeDistancesX)):
            topResize += self.ResizeDistancesX[i][1]
            factorX += self.ResizeDistancesX[i][1]
        for i in range(len(self.ResizeDistancesY)):
            leftResize += self.ResizeDistancesY[i][1]
            factorY += self.ResizeDistancesY[i][1]
        factorX = float(topResize) / factorX
        factorY = float(leftResize) / factorY
        return (factorX, factorY)

    def UpdateCachedImage(self, width, height):
        if False:
            print('Hello World!')
        self.CachedImage = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        self.CachedImage.fill(QColor(0, 0, 0, 0))
        painter = QPainter(self.CachedImage)
        factorX = 0.0
        factorY = 0.0
        (factorX, factorY) = self.GetFactor(width, height, factorX, factorY)
        lostX = 0.0
        lostY = 0.0
        x1 = 0
        y1 = 0
        resizeX = 0
        resizeY = 0
        offsetX = 0
        offsetY = 0
        for i in range(len(self.ResizeDistancesX)):
            y1 = 0
            offsetY = 0
            lostY = 0.0
            for j in range(len(self.ResizeDistancesY)):
                widthResize = self.ResizeDistancesX[i][0] - x1
                heightResize = self.ResizeDistancesY[j][0] - y1
                self.DrawConstPart(QRect(x1 + 1, y1 + 1, widthResize, heightResize), QRect(x1 + offsetX, y1 + offsetY, widthResize, heightResize), painter)
                y2 = self.ResizeDistancesY[j][0]
                heightResize = self.ResizeDistancesY[j][1]
                resizeY = round(float(heightResize) * factorY)
                lostY += resizeY - float(heightResize) * factorY
                if fabs(lostY) >= 1.0:
                    if lostY < 0:
                        resizeY += 1
                        lostY += 1.0
                    else:
                        resizeY -= 1
                        lostY -= 1.0
                self.DrawScaledPart(QRect(x1 + 1, y2 + 1, widthResize, heightResize), QRect(x1 + offsetX, y2 + offsetY, widthResize, resizeY), painter)
                x2 = self.ResizeDistancesX[i][0]
                widthResize = self.ResizeDistancesX[i][1]
                heightResize = self.ResizeDistancesY[j][0] - y1
                resizeX = round(float(widthResize) * factorX)
                lostX += resizeX - float(widthResize) * factorX
                if fabs(lostX) >= 1.0:
                    if lostX < 0:
                        resizeX += 1
                        lostX += 1.0
                    else:
                        resizeX -= 1
                        lostX -= 1.0
                self.DrawScaledPart(QRect(x2 + 1, y1 + 1, widthResize, heightResize), QRect(x2 + offsetX, y1 + offsetY, resizeX, heightResize), painter)
                heightResize = self.ResizeDistancesY[j][1]
                self.DrawScaledPart(QRect(x2 + 1, y2 + 1, widthResize, heightResize), QRect(x2 + offsetX, y2 + offsetY, resizeX, resizeY), painter)
                y1 = self.ResizeDistancesY[j][0] + self.ResizeDistancesY[j][1]
                offsetY += resizeY - self.ResizeDistancesY[j][1]
            x1 = self.ResizeDistancesX[i][0] + self.ResizeDistancesX[i][1]
            offsetX += resizeX - self.ResizeDistancesX[i][1]
        x1 = self.ResizeDistancesX[len(self.ResizeDistancesX) - 1][0] + self.ResizeDistancesX[len(self.ResizeDistancesX) - 1][1]
        widthResize = self.Image.width() - x1 - 2
        y1 = 0
        lostX = 0.0
        lostY = 0.0
        offsetY = 0
        for i in range(len(self.ResizeDistancesY)):
            self.DrawConstPart(QRect(x1 + 1, y1 + 1, widthResize, self.ResizeDistancesY[i][0] - y1), QRect(x1 + offsetX, y1 + offsetY, widthResize, self.ResizeDistancesY[i][0] - y1), painter)
            y1 = self.ResizeDistancesY[i][0]
            resizeY = round(float(self.ResizeDistancesY[i][1]) * factorY)
            lostY += resizeY - float(self.ResizeDistancesY[i][1]) * factorY
            if fabs(lostY) >= 1.0:
                if lostY < 0:
                    resizeY += 1
                    lostY += 1.0
                else:
                    resizeY -= 1
                    lostY -= 1.0
            self.DrawScaledPart(QRect(x1 + 1, y1 + 1, widthResize, self.ResizeDistancesY[i][1]), QRect(x1 + offsetX, y1 + offsetY, widthResize, resizeY), painter)
            y1 = self.ResizeDistancesY[i][0] + self.ResizeDistancesY[i][1]
            offsetY += resizeY - self.ResizeDistancesY[i][1]
        y1 = self.ResizeDistancesY[len(self.ResizeDistancesY) - 1][0] + self.ResizeDistancesY[len(self.ResizeDistancesY) - 1][1]
        heightResize = self.Image.height() - y1 - 2
        x1 = 0
        offsetX = 0
        for i in range(len(self.ResizeDistancesX)):
            self.DrawConstPart(QRect(x1 + 1, y1 + 1, self.ResizeDistancesX[i][0] - x1, heightResize), QRect(x1 + offsetX, y1 + offsetY, self.ResizeDistancesX[i][0] - x1, heightResize), painter)
            x1 = self.ResizeDistancesX[i][0]
            resizeX = round(float(self.ResizeDistancesX[i][1]) * factorX)
            lostX += resizeX - float(self.ResizeDistancesX[i][1]) * factorX
            if fabs(lostX) >= 1.0:
                if lostX < 0:
                    resizeX += 1
                    lostX += 1.0
                else:
                    resizeX -= 1
                    lostX += 1.0
            self.DrawScaledPart(QRect(x1 + 1, y1 + 1, self.ResizeDistancesX[i][1], heightResize), QRect(x1 + offsetX, y1 + offsetY, resizeX, heightResize), painter)
            x1 = self.ResizeDistancesX[i][0] + self.ResizeDistancesX[i][1]
            offsetX += resizeX - self.ResizeDistancesX[i][1]
        x1 = self.ResizeDistancesX[len(self.ResizeDistancesX) - 1][0] + self.ResizeDistancesX[len(self.ResizeDistancesX) - 1][1]
        widthResize = self.Image.width() - x1 - 2
        y1 = self.ResizeDistancesY[len(self.ResizeDistancesY) - 1][0] + self.ResizeDistancesY[len(self.ResizeDistancesY) - 1][1]
        heightResize = self.Image.height() - y1 - 2
        self.DrawConstPart(QRect(x1 + 1, y1 + 1, widthResize, heightResize), QRect(x1 + offsetX, y1 + offsetY, widthResize, heightResize), painter)