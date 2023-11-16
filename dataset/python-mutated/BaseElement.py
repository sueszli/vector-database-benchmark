"""
description: 抽象类模块

Created on 2018年7月7日

Author: 人间白头

email: 625781186@qq.com

"""
import sip, functools
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Tools.qmf_showError import f_showERROR, w_showERROR
from Tools.CommonHelper import CommonHelper
SHOWMENU = {'yes': True, 'no': False, 'setShow': True, 'setHide': False}
ENTERMENU = {'yes': True, 'no': False}

class SingeleWidget(QWidget):
    """
    菜单栏的每个框。
    """
    Button_hideFlag = SHOWMENU['setHide']

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        '\n        Button_hideFlag：  0 表明没有显示弹窗；1表示显示了弹窗。\n        '
        super(SingeleWidget, self).__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.m_menu = QWidget()
        self.setProperty('WID', 'isTrue')

    def enterEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.m_menu.setMinimumWidth(self.width())
        self.m_menu.setMaximumWidth(self.width())
        menu_Pos = self.mapToGlobal(QPoint(self.parent().x(), self.parent().height()))
        self.m_menu.move(menu_Pos)
        self.m_menu.show()
        self.Button_hideFlag = SHOWMENU['setShow']

    def leaveEvent(self, e):
        if False:
            while True:
                i = 10
        '\n        离开时判断是否显示了窗体，80ms后发射到_jugement去检测。\n        '
        if self.Button_hideFlag is SHOWMENU['yes']:
            QTimer.singleShot(80, self._jugement)

    def _jugement(self):
        if False:
            i = 10
            return i + 15
        '\n        离开上面窗体之后80ms, 1：进入旁边的菜单框；2：进入弹出的菜单。\n        '
        if self.m_menu.Menu_hideFlag is ENTERMENU['no']:
            self.m_menu.hide()
            self.m_menu.close()
            self.Button_hideFlag = SHOWMENU['setHide']

class BaseMenuWidget(QTableWidget):
    """
    下拉菜单。
    """
    m_currentRow = None
    m_currentCol = None
    Menu_hideFlag = ENTERMENU['no']

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        '\n        Menu_hideFlag: 0时隐藏，1时显示；\n        '
        super(BaseMenuWidget, self).__init__(parent)
        self.__initUI()

    def __initUI(self):
        if False:
            for i in range(10):
                print('nop')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.Widget)
        self.horizontalHeader().setSectionResizeMode(3)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setSectionResizeMode(1)
        self.verticalHeader().setStretchLastSection(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setColumnCount(1)
        self._setHeight()
        self.parent().readCSS(self)

    def _setHeight(self):
        if False:
            print('Hello World!')
        height = self.rowCount() * 40
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)

    def enterEvent(self, e):
        if False:
            while True:
                i = 10
        self.Menu_hideFlag = ENTERMENU['yes']

    def leaveEvent(self, e):
        if False:
            return 10
        self.Menu_hideFlag = ENTERMENU['no']
        self.hide()
        if self.m_currentRow is not None:
            self.clearSelection()
            self.cellWidget(self.m_currentRow, self.m_currentCol).setCheckable(False)

    def _addAction(self, text, MyWidget=None, func=None, *args, **kwags):
        if False:
            for i in range(10):
                print('nop')
        '\n        obj : QPushButton对象；\n        text：obj的字；\n        func：obj点击链接的信号；\n        MyWidget:想要显示的窗体对象；\n        '
        self.insertRow(self.rowCount())
        self._setHeight()
        row = self.rowCount() - 1
        col = self.columnCount() - 1
        obj = QPushButton(text)
        obj.setProperty('M_Action', 'isTrue')
        obj.setFlat(True)
        obj.setCheckable(True)
        obj.setAutoExclusive(True)
        if func == None:
            func = self.changeTab
        obj.clicked.connect(lambda : setattr(self, 'm_currentRow', row))
        obj.clicked.connect(lambda : setattr(self, 'm_currentCol', col))
        obj.clicked.connect(functools.partial(func, text, MyWidget, *args, **kwags))
        self.setCellWidget(row, col, obj)

    def _findParent(self, currentObj):
        if False:
            print('Hello World!')
        '\n        递归找父窗口。\n        '
        if currentObj.parent().objectName() == 'MainWindow':
            return currentObj.parent()
        return self._findParent(currentObj.parent())

    def changeTab(self, text, MyWidget, *args, **kwags):
        if False:
            print('Hello World!')
        mw = self._findParent(self)
        if 'save' in kwags and 'id' in kwags:
            (save, id) = (kwags['save'], kwags['id'])
            _key = 'b' + self.__class__.__name__[-1]
            childrens = mw.bottomWidget.children()[1:]
            if childrens != []:
                for obj in childrens:
                    obj.setVisible(False)
                    if not hasattr(obj, 'SAVE'):
                        sip.delete(obj)
                        del obj
            if save == 's':
                if MyWidget is not None:
                    if id in mw.Wid_Obj[_key].keys():
                        print('存在wid_obj:', mw.Wid_Obj[_key][id])
                        print('EXIT?:', mw.Wid_Obj[_key][id].SAVE)
                        mw.Wid_Obj[_key][id].setVisible(True)
                    else:
                        print('saving..')
                        obj_Widget = MyWidget()
                        obj_Widget.SAVE = True
                        mw.Wid_Obj[_key][id] = obj_Widget
                        mw.Bottom_Vbox.addWidget(obj_Widget)
            elif save == 'd':
                obj_Widget = MyWidget(mw)
                mw.Bottom_Vbox.addWidget(obj_Widget)
        print(mw.Wid_Obj)

class BaseButton(QPushButton):
    """
    菜单栏的按钮的样式。
    """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super(BaseButton, self).__init__(parent)
        self.setMinimumWidth(70)
        self.setMaximumWidth(88)
        self.setMinimumHeight(self.width())
        self.setFocusPolicy(Qt.NoFocus)
        self.setFlat(True)
        self.clicked.connect(self._todo)
        self.png = QLabel(self)

    def _createLabel(self, path):
        if False:
            i = 10
            return i + 15
        '\n        path：主菜单图标的路径。\n        '
        self.png.resize(self.size())
        self.png_pixmap = QPixmap(path)
        self.png.setPixmap(self.png_pixmap)
        self.png.setScaledContents(True)
        pass

    def _todo(self, *args, **kwgs):
        if False:
            while True:
                i = 10
        '\n        每个按钮要重新实现的功能函数。\n        '
        pass

    def resizeEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.setMinimumHeight(self.width())
        self.png.resize(self.size())