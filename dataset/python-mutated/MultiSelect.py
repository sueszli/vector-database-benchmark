"""
Created on 2018年10月24日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: MultiSelect
@description: 
"""
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMenu, QAction
except ImportError:
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMenu, QAction

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.labelInfo = QLabel(self)
        self.button = QPushButton('带按钮的菜单', self)
        layout.addWidget(self.labelInfo)
        layout.addWidget(self.button)
        self._initMenu()

    def _initMenu(self):
        if False:
            return 10
        self._menu = QMenu(self.button)
        self._menu.mouseReleaseEvent = self._menu_mouseReleaseEvent
        self._menu.addAction('菜单1', self._checkAction)
        self._menu.addAction('菜单2', self._checkAction)
        self._menu.addAction(QAction('菜单3', self._menu, triggered=self._checkAction))
        action = QAction('菜单4', self._menu, triggered=self._checkAction)
        action.setProperty('canHide', True)
        self._menu.addAction(action)
        for action in self._menu.actions():
            action.setCheckable(True)
        self.button.setMenu(self._menu)

    def _menu_mouseReleaseEvent(self, event):
        if False:
            return 10
        action = self._menu.actionAt(event.pos())
        if not action:
            return QMenu.mouseReleaseEvent(self._menu, event)
        if action.property('canHide'):
            return QMenu.mouseReleaseEvent(self._menu, event)
        action.activate(action.Trigger)

    def _checkAction(self):
        if False:
            print('Hello World!')
        self.labelInfo.setText('\n'.join(['{}\t选中：{}'.format(action.text(), action.isChecked()) for action in self._menu.actions()]))
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.resize(400, 400)
    w.show()
    sys.exit(app.exec_())