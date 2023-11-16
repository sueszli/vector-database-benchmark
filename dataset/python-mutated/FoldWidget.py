"""
Created on 2019年5月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: FoldWidget
@description: 自定义item折叠控件仿QTreeWidget
"""
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QPushButton, QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QCheckBox

class CustomWidget(QWidget):

    def __init__(self, item, *args, **kwargs):
        if False:
            return 10
        super(CustomWidget, self).__init__(*args, **kwargs)
        self.oldSize = None
        self.item = item
        layout = QFormLayout(self)
        layout.addRow('我是label', QLineEdit(self))
        layout.addRow('点击', QCheckBox('隐藏下面的按钮', self, toggled=self.hideChild))
        self.button = QPushButton('我是被隐藏的', self)
        layout.addRow(self.button)

    def hideChild(self, v):
        if False:
            print('Hello World!')
        self.button.setVisible(not v)
        self.adjustSize()

    def resizeEvent(self, event):
        if False:
            print('Hello World!')
        super(CustomWidget, self).resizeEvent(event)
        self.item.setSizeHint(QSize(self.minimumWidth(), self.height()))

class CustomButton(QPushButton):

    def __init__(self, item, *args, **kwargs):
        if False:
            print('Hello World!')
        super(CustomButton, self).__init__(*args, **kwargs)
        self.item = item
        self.setCheckable(True)

    def resizeEvent(self, event):
        if False:
            while True:
                i = 10
        super(CustomButton, self).resizeEvent(event)
        self.item.setSizeHint(QSize(self.minimumWidth(), self.height()))

class Window(QListWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        for _ in range(3):
            item = QListWidgetItem(self)
            btn = CustomButton(item, '折叠', self, objectName='testBtn')
            self.setItemWidget(item, btn)
            item = QListWidgetItem(self)
            btn.toggled.connect(item.setHidden)
            self.setItemWidget(item, CustomWidget(item, self))
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyleSheet('#testBtn{min-height:40px;}')
    w = Window()
    w.show()
    sys.exit(app.exec_())