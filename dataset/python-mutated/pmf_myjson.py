"""
增删改查建json.
"""
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication
from PyQt5.QtCore import *
from subprocess import Popen
import json, os
setting_path = './PluginManager'
setting_Name = 'plugin.json'
setting_flie = os.path.join(setting_path, setting_Name)
if not os.path.exists(setting_path):
    os.makedirs(setting_path)
    if not os.path.exists(setting_path):
        f1 = open(setting_flie, 'a+', encoding='utf-8')
        f1.close()
name = {}

def mfunc_readJson(f) -> 'datas':
    if False:
        print('Hello World!')
    '\n    从头读取文件内容。\n    '
    if isinstance(f, str):
        f = open(setting_flie, 'a+', encoding='utf-8')
    f.seek(0)
    text = f.read()
    datas = json.loads(text)
    f.close()
    return datas

def mfunc_initJson(setting_flie, self=None) -> 'datas':
    if False:
        i = 10
        return i + 15
    '\n    初始化创建json。\n    \n    @param setting_flie json路径\n    @type str\n    '
    with open(setting_flie, 'a+', encoding='utf-8') as f:
        if self == None:
            self = QWidget()
        try:
            datas = mfunc_readJson(f)
        except:
            json.dump(name, f, ensure_ascii=False, indent=1)
            try:
                datas = mfunc_readJson(f)
            except:
                QMessageBox.warning(self, '配置文件格式错误', '请严格按照JSON格式，                        \n解决不了请联系程序员：QQ62578186', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                Popen(['write ', setting_flie])
        finally:
            return datas

def mfunc_AKrCVJson(key_name, data, self=None):
    if False:
        while True:
            i = 10
    '\n    createKey or changeValue。\n    \n    @param key_name 要修改的json节点的键名\n    @type str or list-str\n\n    @param data 新数据\n    @type all\n\n    '
    if self == None:
        self = QWidget()
    with open(setting_flie, 'a+', encoding='utf-8') as f:
        datas = mfunc_initJson(setting_flie)
        if isinstance(key_name, str):
            datas[key_name] = data
        elif isinstance(key_name, list):
            if key_name[0] not in datas:
                msg = QMessageBox.warning(self, '没有找到这个配置节点', '请检查参数格式。                \n如果配置不来请清空配置文件。\n是否打开配置文件？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if msg == QMessageBox.Yes:
                    Popen(['write ', setting_flie])
                elif msg == QMessageBox.No:
                    pass
                return
            else:
                cmd_txt = 'datas'
                for key in key_name:
                    cmd_txt += '["%s"]' % key
                cmd_txt += '=data'
                exec(cmd_txt)
        mfunc_reDumpJson(f, datas)

def mfunc_reDumpJson(f, datas):
    if False:
        print('Hello World!')
    f.seek(0)
    f.truncate()
    json.dump(datas, f, ensure_ascii=False, indent=1)