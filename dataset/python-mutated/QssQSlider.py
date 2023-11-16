"""
Created on 2018年5月15日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QssQSlider
@description: 通过QSS美化QSlider
"""
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
StyleSheet = '\nQWidget {\n    background: gray;\n}\n\n/*横向*/\nQSlider:horizontal {\n    min-height: 60px;\n}\nQSlider::groove:horizontal {\n    height: 1px;\n    background: white; \n}\nQSlider::handle:horizontal {\n    width: 30px;\n    margin-top: -15px;\n    margin-bottom: -15px;\n    border-radius: 15px;\n    background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.6 rgba(210, 210, 210, 255), stop:0.7 rgba(210, 210, 210, 100));\n}\nQSlider::handle:horizontal:hover {\n    background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.6 rgba(255, 255, 255, 255), stop:0.7 rgba(255, 255, 255, 100));\n}\n\n/*竖向*/\nQSlider:vertical {\n    min-width: 60px;\n}\nQSlider::groove:vertical {\n    width: 1px;\n    background: white; \n}\nQSlider::handle:vertical {\n    height: 30px;\n    margin-left: -15px;\n    margin-right: -15px;\n    border-radius: 15px;\n    background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.6 rgba(210, 210, 210, 255), stop:0.7 rgba(210, 210, 210, 100));\n}\nQSlider::handle:vertical:hover {\n    background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.6 rgba(255, 255, 255, 255), stop:0.7 rgba(255, 255, 255, 100));\n}\n'

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.addWidget(QSlider(Qt.Vertical, self))
        layout.addWidget(QSlider(Qt.Horizontal, self))
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())