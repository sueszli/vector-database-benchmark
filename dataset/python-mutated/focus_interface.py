from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QGraphicsDropShadowEffect
from qfluentwidgets import FluentIcon, setFont, InfoBarIcon
from view.Ui_FocusInterface import Ui_FocusInterface

class FocusInterface(Ui_FocusInterface, QWidget):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.setupUi(self)
        self.pinButton.setIcon(FluentIcon.PIN)
        self.moreButton.setIcon(FluentIcon.MORE)
        self.startFocusButton.setIcon(FluentIcon.POWER_BUTTON)
        self.editButton.setIcon(FluentIcon.EDIT)
        self.addTaskButton.setIcon(FluentIcon.ADD)
        self.moreTaskButton.setIcon(FluentIcon.MORE)
        self.taskIcon1.setIcon(InfoBarIcon.SUCCESS)
        self.taskIcon2.setIcon(InfoBarIcon.WARNING)
        self.taskIcon3.setIcon(InfoBarIcon.WARNING)
        setFont(self.progressRing, 16)
        self.setShadowEffect(self.focusCard)
        self.setShadowEffect(self.progressCard)
        self.setShadowEffect(self.taskCard)

    def setShadowEffect(self, card: QWidget):
        if False:
            while True:
                i = 10
        shadowEffect = QGraphicsDropShadowEffect(self)
        shadowEffect.setColor(QColor(0, 0, 0, 15))
        shadowEffect.setBlurRadius(10)
        shadowEffect.setOffset(0, 0)
        card.setGraphicsEffect(shadowEffect)