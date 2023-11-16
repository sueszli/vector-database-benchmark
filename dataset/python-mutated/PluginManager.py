"""
管理插件的加载 , 卸载 , 监控文件的添加/删除.
"""
import os, time, sys, importlib, sip, traceback
pluginsManagerPath = os.path.dirname(os.path.abspath(__file__))
mainPath = os.path.dirname(pluginsManagerPath)
pluginsPath = os.path.join(mainPath, 'Plugins')
pluginsPath2 = os.path.join(os.path.dirname(sys.argv[0]), 'Plugins')
AllPluginsPath = {'customer': pluginsPath, 'afterPacket': pluginsPath2}
for key in AllPluginsPath:
    if AllPluginsPath[key] not in sys.path:
        sys.path.insert(0, AllPluginsPath[key])
from copy import deepcopy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PluginManager.PluginStore import PluginStore
from PluginManager.PluginStore.StoreModel import FileModel
from Tools.pmf_myjson import *
'setting_flie -> From Tools.pmf_myjson , json写入的位置'

class PluginManager(QObject):
    """
    管理插件的加载 , 卸载 , 监控文件的添加/删除.
    """

    def __init__(self, parent=None, *args, **kwargs):
        if False:
            return 10
        super(PluginManager, self).__init__(parent, *args, **kwargs)
        self.__mw = parent
        self.__initUI()
        self.pluginDirs = {'pluginFolder': os.path.join(os.path.abspath('./'), 'Plugins')}
        self.header = ['PlugName', 'Allow', 'CreateTime', 'ModifyTime']
        self.pluginsInfo = {'StartModule': {}}
        self.jsonPlugin = None

    def __initUI(self):
        if False:
            for i in range(10):
                print('nop')
        mw = self.__mw
        if mw.findChild(QMenuBar, 'menuBar'):
            mw.menuPlugin = QAction('Plugin', mw.menuBar, triggered=self.__createPluginStoreDialog)
            mw.menuBar.addAction(mw.menuPlugin)
        else:
            QMessageBox.information(mw, '', '主窗体没有菜单栏, 请先创建.')
        self.model = FileModel(self)
        self.model.setRootPath('./Plugins')
        self.model.setFilter(QDir.Files)
        self.model.setNameFilters(['Plugin*.py'])
        self.model.setNameFilterDisables(False)
        self.index = self.model.index('./Plugins')
        self.model.directoryLoaded.connect(self.start)

    def __createPluginStoreDialog(self):
        if False:
            return 10
        '\n        显示插件加载情况的 窗体.\n        '
        if not hasattr(self, 'dia'):
            self.dia = PluginStore.PluginStore(self, self.__mw)
        self.dia.show()

    def __m_rowsRemoved(self, index, first, last):
        if False:
            while True:
                i = 10
        '\n        文件被删除或重命名时候被调用.\n        '
        print('removeName:', self.model.index(first, 0, index).data(), first)
        mod = self.model.index(first, 0, index).data()[:-3]
        self.unload(mod)
        self.pluginsInfo['StartModule'].pop(mod)
        self.delJson(self.jsonPlugin, self.pluginsInfo['StartModule'])

    def __m_rowsInserted(self, index, first, last):
        if False:
            while True:
                i = 10
        '\n        文件增加或重命名时候被调用.\n        '
        print('insertName:', self.model.index(first, 0, index).data(), first)
        f = self.model.index(first, 0, index).data()
        mod = f[:-3]
        fullPath = os.path.join(self.pluginDirs['pluginFolder'], f)
        self.pluginsInfo['StartModule'][mod] = {'path': fullPath}
        (mod, data) = self.addJson(fullPath, mod)
        self.jsonPlugin[mod] = data
        self.load(mod)

    def start(self):
        if False:
            while True:
                i = 10
        '\n        self.model 异步加载完成之后开始调用 self.startGetPlugin.\n        '
        self.jsonPlugin = self.startGetPlugin(self.pluginDirs['pluginFolder'])
        self.loadAll()
        self.model.rowsAboutToBeRemoved.connect(self.__m_rowsRemoved)
        self.model.rowsInserted.connect(self.__m_rowsInserted)
        self.model.directoryLoaded.disconnect(self.start)
        self.__createPluginStoreDialog()

    def startGetPlugin(self, pluginFolder: './Plugins', CHANGE=False) -> 'FoJson':
        if False:
            for i in range(10):
                print('nop')
        '\n        1 . 程序启动加载插件.\n        '
        try:
            jsonPlugin = mfunc_readJson(setting_flie)
        except:
            jsonPlugin = {}
        pluginInfo = {}
        rowCount = self.model.rowCount(self.index)
        for row in range(rowCount):
            index = self.model.index(row, 0, self.index)
            f = index.data()
            module = f[:-3]
            fullPath = os.path.join(pluginFolder, f)
            pluginInfo[module] = {'path': fullPath}
            if module not in jsonPlugin:
                (module, data) = self.addJson(fullPath, module)
                jsonPlugin[module] = data
        if CHANGE is False:
            self.pluginsInfo['StartModule'] = deepcopy(pluginInfo)
        jsonPlugin = self.delJson({}, pluginInfo)
        return jsonPlugin

    def addJson(self, fullPath, module) -> 'ToJson':
        if False:
            return 10
        '\n        1.1写入插件 的json配置.\n        '
        _ctime = time.localtime(os.stat(fullPath).st_ctime)
        ctime = time.strftime('%Y-%m-%d-%H:%M:%S', _ctime)
        _mtime = time.localtime(os.stat(fullPath).st_mtime)
        mtime = time.strftime('%Y-%m-%d-%H:%M:%S', _mtime)
        data = {self.header[1]: True, self.header[2]: ctime, self.header[3]: mtime}
        mfunc_AKrCVJson(module, data, self=self)
        return (module, data)

    def delJson(self, jsonPlugin, pluginInfo) -> 'ToJson':
        if False:
            print('Hello World!')
        '\n        1.2删除插件 的json配置.\n        '
        if jsonPlugin == {}:
            jsonPlugin = mfunc_readJson(setting_flie)
        if len(jsonPlugin) - len(pluginInfo):
            (long, short) = (jsonPlugin, pluginInfo)
        else:
            (long, short) = (pluginInfo, jsonPlugin)
        with open(setting_flie, 'a+', encoding='utf-8') as f:
            delPlugin = set(long) - set(short)
            for item in delPlugin:
                jsonPlugin.pop(item)
            mfunc_reDumpJson(f, jsonPlugin)
        return jsonPlugin

    def loadAll(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        2.加载所有模块.\n        '
        for mod in self.jsonPlugin:
            if self.jsonPlugin[mod]['Allow']:
                try:
                    self.load(mod)
                except:
                    continue
            else:
                self.pluginsInfo['StartModule'][mod]['active'] = False

    def load(self, mod: 'str'):
        if False:
            print('Hello World!')
        '\n        2.1 载入模块.\n        '
        try:
            _pluginModule = importlib.import_module(mod)
        except:
            errmsg = traceback.format_exc()
            QMessageBox.information(self.__mw, '模块导入异常', '%s,请在%s.py检查模块.' % (errmsg, mod))
            self.pluginsInfo['StartModule'][mod]['active'] = False
            return False
        self.instantiation(mod, _pluginModule)
        return True

    def instantiation(self, mod, moduleObj, NeedRplace=False):
        if False:
            return 10
        '\n        2.1.1    实例化类.\n        2.2.1.1  \n        3.1.1\n        实例化新对象来替换旧对象.\n        '
        try:
            className = getattr(moduleObj, 'className')
            pluginClass = getattr(moduleObj, className)
        except:
            self.pluginsInfo['StartModule'][mod]['active'] = False
            errmsg = traceback.format_exc()
            QMessageBox.information(self.__mw, '插件加载错误', '%s ,请在%s.py全局指定className值.' % (errmsg, mod))
            return False
        try:
            pluginObject = pluginClass(self.__mw)
            pluginObject.setObjectName(mod)
            self.pluginsInfo['StartModule'][mod]['active'] = True
            self.pluginsInfo['StartModule'][mod]['pluginClass'] = pluginClass
            self.pluginsInfo['StartModule'][mod]['parent'] = pluginObject.parent()
        except:
            self.pluginsInfo['StartModule'][mod]['active'] = False
            errmsg = traceback.format_exc()
            QMessageBox.information(self.__mw, '插件加载错误', '%s ,请在%s.py全局指定className值.' % (errmsg, mod))
        if not NeedRplace:
            layout = pluginObject.getParentLayout()
            pluginObject.toInterface()
            self.pluginsInfo['StartModule'][mod]['layout'] = layout
            self.pluginsInfo['StartModule'][mod]['old'] = pluginObject
        else:
            self.pluginsInfo['StartModule'][mod]['new'] = pluginObject
            return pluginObject

    def reload(self, mod):
        if False:
            print('Hello World!')
        '\n        2.2 重载插件.\n        '
        if mod in sys.modules:
            print('reload')
            importlib.reload(sys.modules[mod])
            moduleObj = sys.modules[mod]
            try:
                objInfo = self.findOldObj(mod, moduleObj, True)
            except:
                errmsg = traceback.format_exc()
                QMessageBox.information(self.__mw, '模块导入异常', '%s,请在%s.py检查模块.' % (errmsg, mod))
            (oldObj, newObj, layout) = (objInfo['oldObj'], objInfo['newObj'], objInfo['layout'])
            layout.replaceWidget(oldObj, newObj)
            self.pluginsInfo['StartModule'][mod]['old'] = newObj
            oldObj.flag = 'reload'
            sip.delete(oldObj)
        else:
            self.load(mod)

    def findOldObj(self, mod, moduleObj=None, needRplace=False):
        if False:
            return 10
        '\n        3.1\n        2.2.1\n        找到需要删除或替换的对象.\n        '
        oldObj = self.pluginsInfo['StartModule'][mod]['old']
        parentWidget = self.pluginsInfo['StartModule'][mod]['parent']
        layout = self.pluginsInfo['StartModule'][mod]['layout']
        pluginClass = self.pluginsInfo['StartModule'][mod]['pluginClass']
        if needRplace:
            if moduleObj == None:
                QMessageBox.information(self.__mw, '错误', '请传入moduleObj值.')
            else:
                newObj = self.instantiation(mod, moduleObj, needRplace)
        else:
            newObj = None
        return {'oldObj': oldObj, 'newObj': newObj, 'parentWidget': parentWidget, 'layout': layout, 'pluginClass': pluginClass}

    def unload(self, mod: 'str'):
        if False:
            return 10
        '\n        3. 卸载插件 , 移除模块.\n        '
        if mod in sys.modules:
            self.pluginsInfo['StartModule'][mod]['active'] = False
            objInfo = self.findOldObj(mod)
            oldObj = objInfo['oldObj']
            oldObj.flag = 'unload'
            sip.delete(oldObj)
            self.pluginsInfo['StartModule'][mod]['old'] = None
            sys.modules.pop(mod)
        return True

    def unloadAll(self):
        if False:
            return 10
        pass

    def PluginToInterFace(self):
        if False:
            while True:
                i = 10
        pass