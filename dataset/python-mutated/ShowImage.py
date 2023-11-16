"""
Created on 2017年12月23日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ShowImage
@description: 
"""
import sys
try:
    from PyQt5.QtCore import QResource
    from PyQt5.QtGui import QPixmap, QMovie
    from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QLabel
except ImportError:
    from PySide2.QtCore import QResource
    from PySide2.QtGui import QPixmap, QMovie
    from PySide2.QtWidgets import QWidget, QApplication, QHBoxLayout, QLabel
from Lib.xpmres import image_head

class ImageView(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ImageView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QHBoxLayout(self)
        layout.addWidget(QLabel(self, pixmap=QPixmap('Data/head.jpg')))
        layout.addWidget(QLabel(self, pixmap=QPixmap(':/images/head.jpg')))
        QResource.registerResource('Data/res.rcc')
        layout.addWidget(QLabel(self, pixmap=QPixmap(':/myfile/images/head.jpg')))
        layout.addWidget(QLabel(self, pixmap=QPixmap(image_head)))
        movie = QMovie('Data/loading.gif')
        label = QLabel(self)
        label.setMovie(movie)
        layout.addWidget(label)
        movie.start()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ImageView()
    w.show()
    sys.exit(app.exec_())