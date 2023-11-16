try:
    from PySide6.QtWidgets import QWidget, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
    from PySide6.QtGui import QIcon
    from PySide6.QtCore import Qt, QSize, QSettings
except:
    from PyQt5.QtWidgets import QWidget, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
    from PyQt5.QtCore import Qt, QSize, QSettings
    from PyQt5.QtGui import QIcon
from persepolis.scripts.data_base import PersepolisDB
from persepolis.scripts import osCommands
from persepolis.gui import resources

class ErrorWindow(QWidget):

    def __init__(self, text):
        if False:
            while True:
                i = 10
        super().__init__()
        self.setMinimumSize(QSize(363, 300))
        self.setWindowIcon(QIcon.fromTheme('persepolis', QIcon(':/persepolis.svg')))
        self.setWindowTitle('Persepolis Download Manager')
        verticalLayout = QVBoxLayout(self)
        horizontalLayout = QHBoxLayout()
        horizontalLayout.addStretch(1)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.insertPlainText(text)
        verticalLayout.addWidget(self.text_edit)
        self.label2 = QLabel(self)
        self.label2.setText('Reseting persepolis may solving problem.\nDo not panic!If you add your download links again,\npersepolis will resume your downloads\nPlease copy this error message and press "Report Issue" button\nand open a new issue in Github for it.\nWe answer you as soon as possible. \nreporting this issue help us to improve persepolis.\nThank you!')
        verticalLayout.addWidget(self.label2)
        self.report_pushButton = QPushButton(self)
        self.report_pushButton.setText('Report Issue')
        horizontalLayout.addWidget(self.report_pushButton)
        self.reset_persepolis_pushButton = QPushButton(self)
        self.reset_persepolis_pushButton.clicked.connect(self.resetPushButtonPressed)
        self.reset_persepolis_pushButton.setText('Reset Persepolis')
        horizontalLayout.addWidget(self.reset_persepolis_pushButton)
        self.close_pushButton = QPushButton(self)
        self.close_pushButton.setText('close')
        horizontalLayout.addWidget(self.close_pushButton)
        verticalLayout.addLayout(horizontalLayout)
        self.report_pushButton.clicked.connect(self.reportPushButtonPressed)
        self.close_pushButton.clicked.connect(self.closePushButtonPressed)

    def reportPushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        osCommands.xdgOpen('https://github.com/persepolisdm/persepolis/issues')

    def keyPressEvent(self, event):
        if False:
            return 10
        if event.key() == Qt.Key_Escape:
            self.close()

    def closePushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        self.close()

    def resetPushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        persepolis_db = PersepolisDB()
        persepolis_db.resetDataBase()
        persepolis_db.closeConnections()
        persepolis_setting = QSettings('persepolis_download_manager', 'persepolis')
        persepolis_setting.clear()
        persepolis_setting.sync()