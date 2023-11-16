"""
Created on 2020/6/11
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file:
@description: 
"""
from time import sleep
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMovie
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QWidget
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QMovie
    from PySide2.QtWidgets import QApplication, QSplashScreen, QWidget

class GifSplashScreen(QSplashScreen):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(GifSplashScreen, self).__init__(*args, **kwargs)
        self.movie = QMovie('Data/splash.gif')
        self.movie.frameChanged.connect(self.onFrameChanged)
        self.movie.start()

    def onFrameChanged(self, _):
        if False:
            return 10
        self.setPixmap(self.movie.currentPixmap())

    def finish(self, widget):
        if False:
            while True:
                i = 10
        self.movie.stop()
        super(GifSplashScreen, self).finish(widget)

class BusyWindow(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(BusyWindow, self).__init__(*args, **kwargs)
        for i in range(5):
            sleep(1)
            splash.showMessage('加载进度: %d' % i, Qt.AlignHCenter | Qt.AlignBottom, Qt.white)
            QApplication.instance().processEvents()
        splash.showMessage('初始化完成', Qt.AlignHCenter | Qt.AlignBottom, Qt.white)
        splash.finish(self)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    global splash
    splash = GifSplashScreen()
    splash.show()
    w = BusyWindow()
    w.show()
    splash.showMessage('等待创建界面', Qt.AlignHCenter | Qt.AlignBottom, Qt.white)
    sys.exit(app.exec_())