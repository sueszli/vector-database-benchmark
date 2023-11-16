from enum import Enum
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QCheckBox, QStyle, QStyleOptionButton, QWidget
from ...common.icon import FluentIconBase, Theme, getIconColor
from ...common.style_sheet import FluentStyleSheet
from ...common.overload import singledispatchmethod

class CheckBoxIcon(FluentIconBase, Enum):
    """ CheckBoxIcon """
    ACCEPT = 'Accept'
    PARTIAL_ACCEPT = 'PartialAccept'

    def path(self, theme=Theme.AUTO):
        if False:
            while True:
                i = 10
        c = getIconColor(theme, reverse=True)
        return f':/qfluentwidgets/images/check_box/{self.value}_{c}.svg'

class CheckBox(QCheckBox):
    """ Check box

    Constructors
    ------------
    * CheckBox(`parent`: QWidget = None)
    * CheckBox(`text`: str, `parent`: QWidget = None)
    """

    @singledispatchmethod
    def __init__(self, parent: QWidget=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        FluentStyleSheet.CHECK_BOX.apply(self)

    @__init__.register
    def _(self, text: str, parent: QWidget=None):
        if False:
            while True:
                i = 10
        self.__init__(parent)
        self.setText(text)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().paintEvent(e)
        painter = QPainter(self)
        if not self.isEnabled():
            painter.setOpacity(0.8)
        opt = QStyleOptionButton()
        opt.initFrom(self)
        rect = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, opt, self)
        if self.checkState() == Qt.Checked:
            CheckBoxIcon.ACCEPT.render(painter, rect)
        elif self.checkState() == Qt.PartiallyChecked:
            CheckBoxIcon.PARTIAL_ACCEPT.render(painter, rect)