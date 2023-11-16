"""
Created on 2018年1月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CityLinkage
@description: 下拉联动
"""
import json
import sys
import chardet
try:
    from PyQt5.QtCore import Qt, QSortFilterProxyModel, QRegExp
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QSizePolicy
except ImportError:
    from PySide2.QtCore import Qt, QSortFilterProxyModel, QRegExp
    from PySide2.QtGui import QStandardItemModel, QStandardItem
    from PySide2.QtWidgets import QWidget, QApplication, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QSizePolicy

class SortFilterProxyModel(QSortFilterProxyModel):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(SortFilterProxyModel, self).__init__(*args, **kwargs)
        self.setFilterRole(Qt.ToolTipRole)
        self._model = QStandardItemModel(self)
        self.setSourceModel(self._model)

    def appendRow(self, item):
        if False:
            return 10
        self._model.appendRow(item)

    def setFilter(self, _):
        if False:
            print('Hello World!')
        item_code = self.sender().currentData(Qt.ToolTipRole)
        if not item_code:
            return
        if item_code.endswith('0000'):
            self.setFilterRegExp(QRegExp(item_code[:-4] + '\\d\\d00'))
        elif item_code.endswith('00'):
            self.setFilterRegExp(QRegExp(item_code[:-2] + '\\d\\d'))

class CityLinkageWindow(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(CityLinkageWindow, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.province_box = QComboBox(self, minimumWidth=200)
        self.city_box = QComboBox(self, minimumWidth=200)
        self.county_box = QComboBox(self, minimumWidth=200)
        layout.addWidget(QLabel('省/直辖市/特别行政区', self))
        layout.addWidget(self.province_box)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(QLabel('市', self))
        layout.addWidget(self.city_box)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(QLabel('区/县', self))
        layout.addWidget(self.county_box)
        self.initModel()
        self.initSignal()
        self.initData()

    def initSignal(self):
        if False:
            print('Hello World!')
        self.province_box.currentIndexChanged.connect(self.city_model.setFilter)
        self.city_box.currentIndexChanged.connect(self.county_model.setFilter)

    def initModel(self):
        if False:
            while True:
                i = 10
        self.province_model = SortFilterProxyModel(self)
        self.city_model = SortFilterProxyModel(self)
        self.county_model = SortFilterProxyModel(self)
        self.province_box.setModel(self.province_model)
        self.city_box.setModel(self.city_model)
        self.county_box.setModel(self.county_model)

    def initData(self):
        if False:
            while True:
                i = 10
        datas = open('Data/data.json', 'rb').read()
        encoding = chardet.detect(datas) or {}
        datas = datas.decode(encoding.get('encoding', 'utf-8'))
        datas = json.loads(datas)
        for data in datas:
            item_code = data.get('item_code')
            item_name = data.get('item_name')
            item = QStandardItem(item_name)
            item.setData(item_code, Qt.ToolTipRole)
            if item_code.endswith('0000'):
                self.province_model.appendRow(item)
            elif item_code.endswith('00'):
                self.city_model.appendRow(item)
            else:
                self.county_model.appendRow(item)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CityLinkageWindow()
    w.show()
    sys.exit(app.exec_())