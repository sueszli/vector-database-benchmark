"""
Created on 2018年12月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: TabBarStyle
@description: 
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProxyStyle

class TabBarStyle(QProxyStyle):

    def sizeFromContents(self, types, option, size, widget):
        if False:
            while True:
                i = 10
        size = super(TabBarStyle, self).sizeFromContents(types, option, size, widget)
        if types == self.CT_TabBarTab:
            size.transpose()
        return size

    def drawControl(self, element, option, painter, widget):
        if False:
            for i in range(10):
                print('nop')
        if element == self.CE_TabBarTabLabel:
            painter.drawText(option.rect, Qt.AlignCenter, option.text)
            return
        super(TabBarStyle, self).drawControl(element, option, painter, widget)