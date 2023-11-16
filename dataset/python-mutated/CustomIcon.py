"""
Created on 2018年1月26日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CustomIcon
@description: 
"""
import sys
try:
    from PyQt5.QtCore import QFileInfo
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QFileSystemModel, QFileIconProvider, QApplication, QTreeView
except ImportError:
    from PySide2.QtCore import QFileInfo
    from PySide2.QtGui import QIcon
    from PySide2.QtWidgets import QFileSystemModel, QFileIconProvider, QApplication, QTreeView

class FileIconProvider(QFileIconProvider):
    """图标提供类"""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(FileIconProvider, self).__init__(*args, **kwargs)
        self.DirIcon = QIcon('Data/icons/folder.png')
        self.TxtIcon = QIcon('Data/icons/file.png')

    def icon(self, type_info):
        if False:
            i = 10
            return i + 15
        '\n        :param fileInfo: 参考http://doc.qt.io/qt-5/qfileinfo.html\n        '
        if isinstance(type_info, QFileInfo):
            return self.getInfoIcon(type_info)
        '\n        QFileIconProvider::Computer     0\n        QFileIconProvider::Desktop      1\n        QFileIconProvider::Trashcan     2\n        QFileIconProvider::Network      3\n        QFileIconProvider::Drive        4\n        QFileIconProvider::Folder       5\n        QFileIconProvider::File         6\n        '
        if type_info == QFileIconProvider.Folder:
            return self.DirIcon
        return super(FileIconProvider, self).icon(type_info)

    def getInfoIcon(self, type_info):
        if False:
            while True:
                i = 10
        if type_info.isDir():
            return self.DirIcon
        if type_info.isFile() and type_info.suffix() == 'txt':
            return self.TxtIcon
        return super(FileIconProvider, self).icon(type_info)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = QFileSystemModel()
    model.setIconProvider(FileIconProvider())
    model.setRootPath('')
    tree = QTreeView()
    tree.setModel(model)
    tree.setAnimated(False)
    tree.setIndentation(20)
    tree.setSortingEnabled(True)
    tree.setWindowTitle('Dir View')
    tree.resize(640, 480)
    tree.show()
    sys.exit(app.exec_())