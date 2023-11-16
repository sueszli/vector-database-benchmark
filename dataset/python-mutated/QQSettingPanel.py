"""
Created on 2018年3月28日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QQSettingPanel
@description:
"""
try:
    from PyQt5.QtWidgets import QApplication, QWidget
except ImportError:
    from PySide2.QtWidgets import QApplication, QWidget
from Lib.SettingUi import Ui_Setting

class Window(QWidget, Ui_Setting):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.resize(700, 435)
        self._blockSignals = False
        self.scrollArea.verticalScrollBar().valueChanged.connect(self.onValueChanged)
        self.listWidget.itemClicked.connect(self.onItemClicked)

    def onValueChanged(self, value):
        if False:
            i = 10
            return i + 15
        '滚动条'
        if self._blockSignals:
            return
        for i in range(8):
            widget = getattr(self, 'widget_%d' % i, None)
            if widget and (not widget.visibleRegion().isEmpty()):
                self.listWidget.setCurrentRow(i)
                return

    def onItemClicked(self, item):
        if False:
            return 10
        '左侧item'
        row = self.listWidget.row(item)
        widget = getattr(self, 'widget_%d' % row, None)
        if not widget:
            return
        self._blockSignals = True
        self.scrollArea.verticalScrollBar().setSliderPosition(widget.pos().y())
        self._blockSignals = False
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(open('Data/style.qss', 'rb').read().decode('utf-8'))
    w = Window()
    w.show()
    sys.exit(app.exec_())