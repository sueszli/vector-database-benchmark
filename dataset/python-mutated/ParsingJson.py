"""
Created on 2018年4月8日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: ParsingJson
@description: 
"""
import json
import webbrowser
import chardet
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QWidget, QLabel, QSpacerItem, QSizePolicy, QHBoxLayout
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QIcon
    from PySide2.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QWidget, PySide2, QSpacerItem, QSizePolicy, QHBoxLayout

class ItemWidget(QWidget):
    """自定义的item"""

    def __init__(self, text, badge, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ItemWidget, self).__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(text, self, styleSheet='color: white;'))
        layout.addSpacerItem(QSpacerItem(60, 1, QSizePolicy.Maximum, QSizePolicy.Minimum))
        if badge and len(badge) == 2:
            layout.addWidget(QLabel(badge[0], self, alignment=Qt.AlignCenter, styleSheet='min-width: 80px; \n                    max-width: 80px; \n                    min-height: 38px; \n                    max-height: 38px;\n                    color: white; \n                    border:none; \n                    border-radius: 4px; \n                    background: %s' % badge[1]))

class JsonTreeWidget(QTreeWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(JsonTreeWidget, self).__init__(*args, **kwargs)
        self.setEditTriggers(self.NoEditTriggers)
        self.header().setVisible(False)
        self.itemClicked.connect(self.onItemClicked)

    def onItemClicked(self, item):
        if False:
            print('Hello World!')
        'item单击事件'
        if item.url:
            webbrowser.open_new_tab(item.url)

    def parseData(self, datas, parent=None):
        if False:
            i = 10
            return i + 15
        '解析json数据'
        for data in datas:
            url = data.get('url', '')
            items = data.get('items', [])
            _item = QTreeWidgetItem(parent)
            _item.setIcon(0, QIcon(data.get('icon', '')))
            _widget = ItemWidget(data.get('name', ''), data.get('badge', []), self)
            _item.url = url
            self.setItemWidget(_item, 0, _widget)
            if url:
                continue
            if items:
                self.parseData(items, _item)

    def loadData(self, path):
        if False:
            i = 10
            return i + 15
        '加载json数据'
        datas = open(path, 'rb').read()
        datas = datas.decode(chardet.detect(datas).get('encoding', 'utf-8'))
        self.parseData(json.loads(datas), self)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet('QTreeView {\n    outline: 0px;\n    background: rgb(47, 64, 78);\n}\nQTreeView::item {\n    min-height: 92px;\n}\nQTreeView::item:hover {\n    background: rgb(41, 56, 71);\n}\nQTreeView::item:selected {\n    background: rgb(41, 56, 71);\n}\n\nQTreeView::item:selected:active{\n    background: rgb(41, 56, 71);\n}\nQTreeView::item:selected:!active{\n    background: rgb(41, 56, 71);\n}\n\nQTreeView::branch:open:has-children {\n    background: rgb(41, 56, 71);\n}\n\nQTreeView::branch:has-siblings:!adjoins-item {\n    background: green;\n}\nQTreeView::branch:closed:has-children:has-siblings {\n    background: rgb(47, 64, 78);\n}\n\nQTreeView::branch:has-children:!has-siblings:closed {\n    background: rgb(47, 64, 78);\n}\n\nQTreeView::branch:open:has-children:has-siblings {\n    background: rgb(41, 56, 71);\n}\n\nQTreeView::branch:open:has-children:!has-siblings {\n    background: rgb(41, 56, 71);\n}\nQTreeView:branch:hover {\n    background: rgb(41, 56, 71);\n}\nQTreeView:branch:selected {\n    background: rgb(41, 56, 71);\n}\n    ')
    w = JsonTreeWidget()
    w.show()
    w.loadData('Data/data.json')
    sys.exit(app.exec_())