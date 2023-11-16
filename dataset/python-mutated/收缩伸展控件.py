"""
@resource:http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
@description: 摘录自上方
@Created on none
@email: none
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Spoiler(QWidget):

    def __init__(self, parent=None, title='', animationDuration=300):
        if False:
            return 10
        '\n        References:\n            # Adapted from c++ version\n            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt\n        '
        super(Spoiler, self).__init__(parent=parent)
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(210, 30, 95, 134))
        self.groupBox.setObjectName('groupBox')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName('verticalLayout')
        self.pushButton3 = QtWidgets.QPushButton(self.groupBox)
        icon = QtGui.QIcon()
        self.pushButton3.setIcon(icon)
        self.pushButton3.setObjectName('pushButton3')
        self.verticalLayout.addWidget(self.pushButton3)
        self.pushButton2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton2.setObjectName('pushButton2')
        self.verticalLayout.addWidget(self.pushButton2)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName('label_2')
        self.verticalLayout.addWidget(self.label_2)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setIcon(icon)
        self.pushButton.setObjectName('pushButton')
        self.verticalLayout.addWidget(self.pushButton)
        self.animationDuration = 300
        self.toggleAnimation = QParallelAnimationGroup()
        self.contentArea = QScrollArea()
        self.headerLine = QFrame()
        self.toggleButton = QToolButton()
        self.mainLayout = QGridLayout()
        toggleButton = self.toggleButton
        toggleButton.setStyleSheet('QToolButton { border: none; }')
        toggleButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)
        headerLine = self.headerLine
        headerLine.setFrameShape(QFrame.HLine)
        headerLine.setFrameShadow(QFrame.Sunken)
        headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.contentArea.setStyleSheet('QScrollArea { background-color: white; border: none; }')
        self.contentArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QPropertyAnimation(self, b'minimumHeight'))
        toggleAnimation.addAnimation(QPropertyAnimation(self, b'maximumHeight'))
        toggleAnimation.addAnimation(QPropertyAnimation(self.contentArea, b'maximumHeight'))
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            if False:
                while True:
                    i = 10
            arrow_type = Qt.DownArrow if checked else Qt.RightArrow
            direction = QAbstractAnimation.Forward if checked else QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()
        self.toggleButton.clicked.connect(start_animation)

    def setContentLayout(self, contentLayout):
        if False:
            print('Hello World!')
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount() - 1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Spoiler()
    ui.setContentLayout(ui.verticalLayout)
    ui.show()
    sys.exit(app.exec_())