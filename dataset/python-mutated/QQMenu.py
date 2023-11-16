"""
Created on 2021/4/7
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QQMenu
@description: 
"""
import string
from random import choice, randint
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QPainter, QFont, QIcon
    from PyQt5.QtWidgets import QLabel, QMenu, QApplication
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QPixmap, QPainter, QFont, QIcon
    from PySide2.QtWidgets import QLabel, QMenu, QApplication
Style = '\nQMenu {\n    /* 半透明效果 */\n    background-color: rgba(255, 255, 255, 230);\n    border: none;\n    border-radius: 4px;\n}\n\nQMenu::item {\n    border-radius: 4px;\n    /* 这个距离很麻烦需要根据菜单的长度和图标等因素微调 */\n    padding: 8px 48px 8px 36px; /* 36px是文字距离左侧距离*/\n    background-color: transparent;\n}\n\n/* 鼠标悬停和按下效果 */\nQMenu::item:selected {\n    border-radius: 0px;\n    /* 半透明效果 */\n    background-color: rgba(232, 232, 232, 232);\n}\n\n/* 禁用效果 */\nQMenu::item:disabled {\n    background-color: transparent;\n}\n\n/* 图标距离左侧距离 */\nQMenu::icon {\n    left: 15px;\n}\n\n/* 分割线效果 */\nQMenu::separator {\n    height: 1px;\n    background-color: rgb(232, 236, 243);\n}\n'

def get_icon():
    if False:
        for i in range(10):
            print('nop')
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setFont(QFont('Webdings', 11))
    painter.setPen(Qt.GlobalColor(randint(4, 18)))
    painter.drawText(0, 0, 16, 16, Qt.AlignCenter, choice(string.ascii_letters))
    painter.end()
    return QIcon(pixmap)

def about_qt():
    if False:
        i = 10
        return i + 15
    QApplication.instance().aboutQt()

class Window(QLabel):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Window, self).__init__(*args, **kwargs)
        self.resize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setText('右键弹出菜单')
        self.context_menu = QMenu(self)
        self.init_menu()

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        self.context_menu.exec_(event.globalPos())

    def init_menu(self):
        if False:
            i = 10
            return i + 15
        self.context_menu.setAttribute(Qt.WA_TranslucentBackground)
        self.context_menu.setWindowFlags(self.context_menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        for i in range(10):
            if i % 2 == 0:
                action = self.context_menu.addAction('菜单 %d' % i, about_qt)
                action.setEnabled(i % 4)
            elif i % 3 == 0:
                self.context_menu.addAction(get_icon(), '菜单 %d' % i, about_qt)
            if i % 4 == 0:
                self.context_menu.addSeparator()
            if i % 5 == 0:
                menu = QMenu('二级菜单 %d' % i, self.context_menu)
                menu.setAttribute(Qt.WA_TranslucentBackground)
                menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
                for j in range(3):
                    menu.addAction(get_icon(), '子菜单 %d' % j)
                self.context_menu.addMenu(menu)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    app.setStyleSheet(Style)
    w = Window()
    w.show()
    sys.exit(app.exec_())