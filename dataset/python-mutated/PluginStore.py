"""
插件仓库管理界面.
"""
'\nCreated on 2018-09-18 <br>\ndescription: $description$ <br>\nauthor: 625781186@qq.com <br>\nsite: https://github.com/625781186 <br>\n更多经典例子:https://github.com/892768447/PyQt <br>\n课件: https://github.com/625781186/WoHowLearn_PyQt5 <br>\n视频教程: https://space.bilibili.com/1863103/#/ <br>\n'
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QDialog
try:
    from Ui_PluginStore import Ui_Dialog
except:
    from .Ui_PluginStore import Ui_Dialog
from Tools.pmf_myjson import *

class PluginStore(QDialog, Ui_Dialog):
    """
    Class documentation goes here.
    """

    def __init__(self, manager, parent=None):
        if False:
            i = 10
            return i + 15
        '\n        Constructor\n        \n        @param parent reference to the parent widget\n        @type QWidget\n        '
        super(PluginStore, self).__init__(parent)
        self.setupUi(self)
        self.manager = manager
        self.__mw = parent
        self.model = manager.model
        self.index = manager.index
        header = manager.header
        jsonPlugin = manager.jsonPlugin
        activeInfo = manager.pluginsInfo
        self.__initUI(header, jsonPlugin, activeInfo)

    def __initUI(self, header, jsonPlugin, activeInfo):
        if False:
            i = 10
            return i + 15
        self.model.RE_UN_LoadSignal.connect(self.re_un_load)
        self.model.AutoStartSignal.connect(self.allow_un_start)
        self.tableView.setModel(self.model)
        self.tableView.setRootIndex(self.index)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        for col in [0, 3, 4, 5]:
            self.tableView.resizeColumnToContents(col)
        self.tableView.horizontalHeader().resizeSection(1, 0)
        self.tableView.horizontalHeader().resizeSection(2, 0)
        self.tableView.verticalHeader().setSectionResizeMode(2)
        self.tableView.setMouseTracking(True)
        self.tableView.entered.connect(lambda index: self.setToolTip(index.data()) if index.column() != 1 else 0)
        self.tableView.customContextMenuRequested.connect(self.myListWidgetContext)
        self.tableView.setContextMenuPolicy(Qt.CustomContextMenu)

    def myListWidgetContext(self):
        if False:
            i = 10
            return i + 15
        popMenu = QMenu()
        popMenu.addAction(u'重载模块', lambda : self.re_un_load(1))
        popMenu.addAction(u'卸载模块', lambda : self.re_un_load(2))
        popMenu.exec_(QCursor.pos())

    def re_un_load(self, type=1):
        if False:
            return 10
        '\n        加载/重载和卸载插件.\n        '
        if isinstance(type, tuple):
            mod = type[0]
            index = type[1]
            self.__mw.activateWindow()
            self.manager.dia.tableView.activateWindow()
            self.manager.dia.tableView.setCurrentIndex(index)
            if self.manager.pluginsInfo['StartModule'][mod]['active']:
                msg = QMessageBox.information(self, '确认', '即将卸载插件.', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if msg == QMessageBox.Yes:
                    self.manager.unload(mod)
            else:
                self.manager.reload(mod)
        else:
            for index in self.tableView.selectionModel().selectedRows():
                mod = index.data()[:-3]
                if type == 1:
                    self.manager.reload(mod)
                elif type == 2:
                    self.manager.unload(mod)

    def del_Item(self):
        if False:
            i = 10
            return i + 15
        pass

    def allow_un_start(self, mod):
        if False:
            for i in range(10):
                print('nop')
        '\n        允许/禁止 插件的自启动.\n        '
        if self.manager.jsonPlugin[mod]['Allow']:
            msg = QMessageBox.information(self, '确认', '即将禁止插件自启.', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if msg == QMessageBox.Yes:
                mfunc_AKrCVJson([mod, 'Allow'], False)
                self.manager.jsonPlugin[mod]['Allow'] = False
        else:
            QMessageBox.information(self, '允许', '已允许插件自启.')
            mfunc_AKrCVJson([mod, 'Allow'], True)
            self.manager.jsonPlugin[mod]['Allow'] = True