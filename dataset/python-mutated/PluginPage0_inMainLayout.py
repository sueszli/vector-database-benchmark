"""
插件例子1.
"""
'\nCreated on 2018-09-18 <br>\ndescription: $description$ <br>\nauthor: 625781186@qq.com <br>\nsite: https://github.com/625781186 <br>\n更多经典例子:https://github.com/892768447/PyQt <br>\n课件: https://github.com/625781186/WoHowLearn_PyQt5 <br>\n视频教程: https://space.bilibili.com/1863103/#/ <br>\n'
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
try:
    from Ui_PluginPage1 import Ui_Form
except:
    from page1.Ui_PluginPage1 import Ui_Form
className = 'Form'

class Form(QWidget, Ui_Form):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        '\n        \n        '
        super(Form, self).__init__(parent)
        self.setupUi(self)
        self.__mw = parent

    def getParentLayout(self):
        if False:
            while True:
                i = 10
        '\n        布局函数,必须.\n        '
        return self.__mw.verticalLayout

    def toInterface(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        插入到界面,必须\n        '
        layout = self.getParentLayout()
        layout.addWidget(self)

    def __del__(self):
        if False:
            i = 10
            return i + 15
        print('die')

    @pyqtSlot()
    def on_pushButton_clicked(self):
        if False:
            i = 10
            return i + 15
        print(2)
        pass

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        if False:
            return 10
        print(2)
        pass

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        if False:
            i = 10
            return i + 15
        print(3)
        pass
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    ui = Form()
    ui.show()
    sys.exit(app.exec_())