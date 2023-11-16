from qt.core import QAbstractItemView, QAction, QComboBox, QCoreApplication, QDialog, QDialogButtonBox, QDrag, QDropEvent, QEvent, QEventLoop, QFontMetrics, QFormLayout, QFrame, QHoverEvent, QImage, QIODevice, QLayout, QLineEdit, QMenu, QMessageBox, QModelIndex, QPalette, QSinglePointEvent, QSizePolicy, Qt, QThread, QToolButton
from calibre_extensions import progress_indicator
QSinglePointEvent.x = lambda self: int(self.position().x())
QSinglePointEvent.y = lambda self: int(self.position().y())
QSinglePointEvent.globalPos = lambda self: self.globalPosition().toPoint()
QSinglePointEvent.globalX = lambda self: self.globalPosition().toPoint().x()
QSinglePointEvent.globalY = lambda self: self.globalPosition().toPoint().y()
QSinglePointEvent.localPos = lambda self: self.position()
QSinglePointEvent.screenPos = lambda self: self.globalPosition()
QSinglePointEvent.windowPos = lambda self: self.scenePosition()
QDropEvent.pos = lambda self: self.position().toPoint()
QDropEvent.posF = lambda self: self.position()
QHoverEvent.pos = lambda self: self.position().toPoint()
QHoverEvent.posF = lambda self: self.position()

def exec_(self, *args, **kwargs):
    if False:
        print('Hello World!')
    return self.exec(*args, **kwargs)
QDialog.exec_ = exec_
QMenu.exec_ = exec_
QDrag.exec_ = exec_
QEventLoop.exec_ = exec_
QThread.exec_ = exec_
QMessageBox.exec_ = exec_
QCoreApplication.exec_ = exec_

def set_menu(self, menu):
    if False:
        print('Hello World!')
    self.keep_menu_ref = menu
    progress_indicator.set_menu_on_action(self, menu)
QAction.setMenu = set_menu
QAction.menu = lambda self: progress_indicator.menu_for_action(self)
QModelIndex.child = lambda self, row, column: self.model().index(row, column, self)
QFontMetrics.width = lambda self, text: self.horizontalAdvance(text)
for cls in (Qt, QDialog, QToolButton, QAbstractItemView, QDialogButtonBox, QFrame, QComboBox, QLineEdit, QAction, QImage, QIODevice, QPalette, QFormLayout, QEvent, QMessageBox, QSizePolicy, QLayout):
    for var in tuple(vars(cls).values()):
        m = getattr(var, '__members__', {})
        for (k, v) in m.items():
            if not hasattr(cls, k):
                setattr(cls, k, v)