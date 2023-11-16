"""
Created on 2020年3月2日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file:
@description: 冷色调
"""
import sys
from ctypes import CDLL
from time import time
try:
    try:
        from PyQt5 import sip
    except ImportError:
        import sip
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSlider, QVBoxLayout
except ImportError:
    import shiboken2
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QImage, QPixmap
    from PySide2.QtWidgets import QApplication, QWidget, QLabel, QSlider, QVBoxLayout

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.imgLabel = QLabel(self)
        self.coldSlider = QSlider(Qt.Horizontal, self)
        self.coldSlider.valueChanged.connect(self.doChange)
        self.coldSlider.setRange(0, 255)
        layout.addWidget(self.imgLabel)
        layout.addWidget(self.coldSlider)
        self.srcImg = QImage('src.jpg')
        self.imgLabel.setPixmap(QPixmap.fromImage(self.srcImg).scaledToWidth(800, Qt.SmoothTransformation))
        self.dll = CDLL('Cold.dll')
        print(self.dll)

    def doChange(self, value):
        if False:
            for i in range(10):
                print('nop')
        t = time()
        img = self.srcImg.copy()
        if 'sip' in sys.modules:
            self.dll.cold(sip.unwrapinstance(img), value)
        elif 'shiboken2' in sys.modules:
            self.dll.cold(shiboken2.getCppPointer(img)[0], value)
        self.imgLabel.setPixmap(QPixmap.fromImage(img).scaledToWidth(800, Qt.SmoothTransformation))
        print('use time:', time() - t)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = Window()
    view.show()
    sys.exit(app.exec_())