try:
    from PySide6 import QtSvg
    from PySide6.QtWidgets import QDesktopWidget, QHBoxLayout, QVBoxLayout, QLabel, QWidget
    from PySide6.QtCore import Qt, QSize, QRect, QPoint
except:
    from PyQt5 import QtSvg
    from PyQt5.QtWidgets import QDesktopWidget, QHBoxLayout, QVBoxLayout, QLabel, QWidget
    from PyQt5.QtCore import Qt, QSize, QRect, QPoint

class Windows_Notification_UI(QWidget):

    def __init__(self, parent, persepolis_setting):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.persepolis_setting = persepolis_setting
        ui_direction = self.persepolis_setting.value('ui_direction')
        if ui_direction == 'rtl':
            self.setLayoutDirection(Qt.RightToLeft)
        elif ui_direction in 'ltr':
            self.setLayoutDirection(Qt.LeftToRight)
        self.resize(QSize(400, 80))
        self.setFixedWidth(400)
        self.setWindowFlags(Qt.ToolTip)
        bottom_right_screen = QDesktopWidget().availableGeometry().bottomRight()
        bottom_right_notification = QRect(QPoint(0, 0), QSize(410, 120))
        bottom_right_notification.moveBottomRight(bottom_right_screen)
        self.move(bottom_right_notification.topLeft())
        icons = ':/' + str(self.persepolis_setting.value('settings/icons')) + '/'
        notification_horizontalLayout = QHBoxLayout(self)
        svgWidget = QtSvg.QSvgWidget(':/persepolis.svg')
        svgWidget.setFixedSize(QSize(64, 64))
        notification_horizontalLayout.addWidget(svgWidget)
        notification_verticalLayout = QVBoxLayout()
        self.label1 = QLabel(self)
        self.label1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label1.setStyleSheet('font-weight: bold')
        self.label1.setWordWrap(True)
        self.label2 = QLabel(self)
        self.label2.setWordWrap(True)
        self.label2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        notification_verticalLayout.addWidget(self.label1)
        notification_verticalLayout.addWidget(self.label2)
        notification_horizontalLayout.addLayout(notification_verticalLayout)