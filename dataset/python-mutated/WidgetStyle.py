"""
Created on 2017年12月10日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: test
@description: 
"""
import sys
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import QWidget, QApplication, QHBoxLayout
from Lib.CustomPaintWidget import CustomPaintWidget
from Lib.CustomWidget import CustomWidget

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.addWidget(CustomPaintWidget(self))
        layout.addWidget(CustomWidget(self))
        wc = CustomWidget(self)
        wc.setAttribute(Qt.WA_StyledBackground)
        layout.addWidget(wc)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('\nCustomPaintWidget {\n    min-width: 100px;\n    min-height: 100px;\n    border: 1px solid green;\n    border-radius: 20px;\n    background: green;\n}\nCustomWidget {\n    min-width: 200px;\n    min-height: 200px;\n    max-width: 200px;\n    max-height: 200px;\n    border: 1px solid orange;\n    border-radius: 100px;\n    background: orange;\n}\n    ')
    w = Window()
    w.show()
    sys.exit(app.exec_())