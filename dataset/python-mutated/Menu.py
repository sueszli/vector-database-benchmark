"""
description: 要增加菜单栏时，在这里添加。<br>

Created on 2018年7月7日

Author: 人间白头

email: 625781186@qq.com

BX: 主菜单按钮；<br>
LX：子菜单按钮 BaseMenuWidget - 是个QTableWidget；<br>
"""
from U_FuncWidget.BaseElement import *
from U_FuncWidget.UThroughTrain4 import GeographicAnalysis_Widget
from U_FuncWidget.UCompetitiveProduct2 import SKU_Widget

class B1(BaseButton):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super(B1, self).__init__(parent)
        self._createLabel(':/static/store_data.png')

    def _todo(self, *args, **kwgs):
        if False:
            return 10
        self.msg = QErrorMessage()
        self.msg.showMessage('你可以在此添加额外功能。')

class L1(BaseMenuWidget):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super(L1, self).__init__(parent)
        self._addAction('病人信息', GeographicAnalysis_Widget.Form, id='1', save='s')
        self._addAction('病人')

class B2(BaseButton):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super(B2, self).__init__(parent)
        self._createLabel(':/static/competitiveProductAnalysis.png')

    def _todo(self, *args, **kwgs):
        if False:
            return 10
        super(B2, self)._todo()

class L2(BaseMenuWidget):

    def __init__(self, parent=None):
        if False:
            return 10
        super(L2, self).__init__(parent)
        self._addAction('病人信息', SKU_Widget.Form, id='2', save='s')
        self._addAction('检测1')
        self._addAction('检测2', 'hello')
        self._addAction('标定3', 'word', self.cusTomerFunc)
        self._addAction('标定4', 'word', self.cusTomerFunc, id='4', save='d')

    def cusTomerFunc(self, *a, **kw):
        if False:
            return 10
        self.changeTab('', GeographicAnalysis_Widget.Form, *a, **kw)
        print(a, kw)

class B3(BaseButton):

    def __init__(self, parent=None):
        if False:
            return 10
        super(B3, self).__init__(parent)
        self._createLabel(':/static/search.png')

    def _todo(self, *args, **kwgs):
        if False:
            i = 10
            return i + 15
        super(B3, self)._todo()

class L3(BaseMenuWidget):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super(L3, self).__init__(parent)

class B4(BaseButton):

    def __init__(self, parent=None):
        if False:
            return 10
        super(B4, self).__init__(parent)
        self._createLabel(':/static/throughTrain.png')

class L4(BaseMenuWidget):

    def __init__(self, parent=None):
        if False:
            return 10
        super(L4, self).__init__(parent)