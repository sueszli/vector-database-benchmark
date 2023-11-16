"""
Created on 2018年9月25日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ShadowEffect
@description: 
"""
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit, QApplication
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QPixmap
    from PySide2.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit, QApplication
from Lib.AnimationShadowEffect import AnimationShadowEffect

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        labelGreen = QLabel(self, pixmap=QPixmap('Data/1.jpg').scaled(100, 100))
        layout.addWidget(labelGreen)
        aniGreen = AnimationShadowEffect(Qt.darkGreen, labelGreen)
        labelGreen.setGraphicsEffect(aniGreen)
        aniGreen.start()
        labelRed = QLabel(self)
        labelRed.setMinimumSize(100, 100)
        labelRed.setMaximumSize(100, 100)
        labelRed.setStyleSheet('border-image: url(Data/1.jpg);border-radius: 50px;')
        layout.addWidget(labelRed)
        aniRed = AnimationShadowEffect(Qt.red, labelGreen)
        labelRed.setGraphicsEffect(aniRed)
        aniRed.start()
        button = QPushButton('按钮', self)
        aniButton = AnimationShadowEffect(Qt.blue, button)
        layout.addWidget(button)
        button.setGraphicsEffect(aniButton)
        button.clicked.connect(aniButton.stop)
        aniButton.start()
        lineedit = QLineEdit(self)
        aniEdit = AnimationShadowEffect(Qt.cyan, lineedit)
        layout.addWidget(lineedit)
        lineedit.setGraphicsEffect(aniEdit)
        aniEdit.start()
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())