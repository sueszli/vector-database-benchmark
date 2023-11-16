"""
Created on 2018年1月20日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CircleImage
@description: 圆形图片
"""
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QPainter, QPainterPath
    from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QApplication
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QPixmap, QPainter, QPainterPath
    from PySide2.QtWidgets import QLabel, QWidget, QHBoxLayout, QApplication

class Label(QLabel):

    def __init__(self, *args, antialiasing=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Label, self).__init__(*args, **kwargs)
        self.Antialiasing = antialiasing
        self.setMaximumSize(200, 200)
        self.setMinimumSize(200, 200)
        self.radius = 100
        self.target = QPixmap(self.size())
        self.target.fill(Qt.transparent)
        p = QPixmap('Data/Images/head.jpg').scaled(200, 200, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        painter = QPainter(self.target)
        if self.Antialiasing:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self.radius, self.radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, p)
        self.setPixmap(self.target)

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.addWidget(Label(self))
        layout.addWidget(Label(self, antialiasing=False))
        self.setStyleSheet('background: black;')
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())