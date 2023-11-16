"""
Created on 2018年1月29日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: NormalStyle
@description: 
"""
import sys
try:
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QApplication
except ImportError:
    from PySide2.QtWidgets import QWidget, QHBoxLayout, QPushButton, QApplication
StyleSheet = '\n/*这里是通用设置，所有按钮都有效，不过后面的可以覆盖这个*/\nQPushButton {\n    border: none; /*去掉边框*/\n}\n\n/*\nQPushButton#xxx\n或者\n#xx\n都表示通过设置的objectName来指定\n*/\nQPushButton#RedButton {\n    background-color: #f44336; /*背景颜色*/\n}\n#RedButton:hover {\n    background-color: #e57373; /*鼠标悬停时背景颜色*/\n}\n/*注意pressed一定要放在hover的后面，否则没有效果*/\n#RedButton:pressed {\n    background-color: #ffcdd2; /*鼠标按下不放时背景颜色*/\n}\n\n#GreenButton {\n    background-color: #4caf50;\n    border-radius: 5px; /*圆角*/\n}\n#GreenButton:hover {\n    background-color: #81c784;\n}\n#GreenButton:pressed {\n    background-color: #c8e6c9;\n}\n\n#BlueButton {\n    background-color: #2196f3;\n    /*限制最小最大尺寸*/\n    min-width: 96px;\n    max-width: 96px;\n    min-height: 96px;\n    max-height: 96px;\n    border-radius: 48px; /*圆形*/\n}\n#BlueButton:hover {\n    background-color: #64b5f6;\n}\n#BlueButton:pressed {\n    background-color: #bbdefb;\n}\n\n#OrangeButton {\n    max-height: 48px;\n    border-top-right-radius: 20px; /*右上角圆角*/\n    border-bottom-left-radius: 20px; /*左下角圆角*/\n    background-color: #ff9800;\n}\n#OrangeButton:hover {\n    background-color: #ffb74d;\n}\n#OrangeButton:pressed {\n    background-color: #ffe0b2;\n}\n\n/*根据文字内容来区分按钮,同理还可以根据其它属性来区分*/\nQPushButton[text="purple button"] {\n    color: white; /*文字颜色*/\n    background-color: #9c27b0;\n}\n'

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.addWidget(QPushButton('red button', self, objectName='RedButton', minimumHeight=48))
        layout.addWidget(QPushButton('green button', self, objectName='GreenButton', minimumHeight=48))
        layout.addWidget(QPushButton('blue button', self, objectName='BlueButton', minimumHeight=48))
        layout.addWidget(QPushButton('orange button', self, objectName='OrangeButton', minimumHeight=48))
        layout.addWidget(QPushButton('purple button', self, objectName='PurpleButton', minimumHeight=48))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())