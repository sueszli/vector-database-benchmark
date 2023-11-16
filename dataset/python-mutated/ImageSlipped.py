"""
Created on 2018年10月18日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ImageSlipped
@description: 
"""
try:
    from PyQt5.QtGui import QPixmap, QPainter
    from PyQt5.QtWidgets import QWidget, QApplication
except ImportError:
    from PySide2.QtGui import QPixmap, QPainter
    from PySide2.QtWidgets import QWidget, QApplication

class SlippedImgWidget(QWidget):

    def __init__(self, bg, fg, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SlippedImgWidget, self).__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.bgPixmap = QPixmap(bg)
        self.pePixmap = QPixmap(fg)
        size = self.bgPixmap.size()
        self.setMinimumSize(size.width() - 10, size.height() - 10)
        self.setMaximumSize(size.width() - 10, size.height() - 10)
        self.stepX = size.width() / 10
        self.stepY = size.height() / 10
        self._offsets = [-4, -4, -4, -4]

    def mouseMoveEvent(self, event):
        if False:
            return 10
        super(SlippedImgWidget, self).mouseMoveEvent(event)
        pos = event.pos()
        offsetX = 5 - int(pos.x() / self.stepX)
        offsetY = 5 - int(pos.y() / self.stepY)
        self._offsets[0] = offsetX
        self._offsets[1] = offsetY
        self._offsets[2] = offsetX
        self._offsets[3] = offsetY
        self.update()

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(SlippedImgWidget, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(-5 + self._offsets[0], -5 + self._offsets[1], self.bgPixmap)
        painter.drawPixmap(self.width() - self.pePixmap.width() + 5 - self._offsets[2], self.height() - self.pePixmap.height() + 5 - self._offsets[3], self.pePixmap)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = SlippedImgWidget('Data/bg1.jpg', 'Data/fg1.png')
    w.show()
    sys.exit(app.exec_())