"""
Created on 2019年4月19日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FramelessDialog
@description: 无边框圆角对话框 
"""
try:
    from PyQt5.QtCore import Qt, QSize, QTimer
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QWidget, QGraphicsDropShadowEffect, QPushButton, QGridLayout, QSpacerItem, QSizePolicy, QApplication
except ImportError:
    from PySide2.QtCore import Qt, QSize, QTimer
    from PySide2.QtWidgets import QDialog, QVBoxLayout, QWidget, QGraphicsDropShadowEffect, QPushButton, QGridLayout, QSpacerItem, QSizePolicy, QApplication
Stylesheet = '\n#Custom_Widget {\n    background: white;\n    border-radius: 10px;\n}\n\n#closeButton {\n    min-width: 36px;\n    min-height: 36px;\n    font-family: "Webdings";\n    qproperty-text: "r";\n    border-radius: 10px;\n}\n#closeButton:hover {\n    color: white;\n    background: red;\n}\n'

class Dialog(QDialog):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Dialog, self).__init__(*args, **kwargs)
        self.setObjectName('Custom_Dialog')
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet(Stylesheet)
        self.initUi()
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(12)
        effect.setOffset(0, 0)
        effect.setColor(Qt.gray)
        self.setGraphicsEffect(effect)

    def initUi(self):
        if False:
            while True:
                i = 10
        layout = QVBoxLayout(self)
        self.widget = QWidget(self)
        self.widget.setObjectName('Custom_Widget')
        layout.addWidget(self.widget)
        layout = QGridLayout(self.widget)
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum), 0, 0)
        layout.addWidget(QPushButton('r', self, clicked=self.accept, objectName='closeButton'), 0, 1)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 1, 0)

    def sizeHint(self):
        if False:
            while True:
                i = 10
        return QSize(600, 400)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Dialog()
    w.exec_()
    QTimer.singleShot(200, app.quit)
    sys.exit(app.exec_())