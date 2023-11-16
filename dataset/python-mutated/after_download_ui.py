try:
    from PySide6.QtWidgets import QCheckBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
    from PySide6.QtCore import Qt, QTranslator, QCoreApplication, QLocale
    from PySide6.QtGui import QIcon
except:
    from PyQt5.QtWidgets import QCheckBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
    from PyQt5.QtCore import Qt, QTranslator, QCoreApplication, QLocale
    from PyQt5.QtGui import QIcon
from persepolis.gui import resources

class AfterDownloadWindow_Ui(QWidget):

    def __init__(self, persepolis_setting):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.persepolis_setting = persepolis_setting
        locale = str(self.persepolis_setting.value('settings/locale'))
        QLocale.setDefault(QLocale(locale))
        self.translator = QTranslator()
        if self.translator.load(':/translations/locales/ui_' + locale, 'ts'):
            QCoreApplication.installTranslator(self.translator)
        ui_direction = self.persepolis_setting.value('ui_direction')
        if ui_direction == 'rtl':
            self.setLayoutDirection(Qt.RightToLeft)
        elif ui_direction in 'ltr':
            self.setLayoutDirection(Qt.LeftToRight)
        icons = ':/' + str(self.persepolis_setting.value('settings/icons')) + '/'
        self.setWindowIcon(QIcon.fromTheme('persepolis', QIcon(':/persepolis.svg')))
        self.setWindowTitle(QCoreApplication.translate('after_download_ui_tr', 'Persepolis Download Manager'))
        window_verticalLayout = QVBoxLayout()
        window_verticalLayout.setContentsMargins(21, 21, 21, 21)
        self.complete_label = QLabel()
        window_verticalLayout.addWidget(self.complete_label)
        self.file_name_label = QLabel()
        window_verticalLayout.addWidget(self.file_name_label)
        self.size_label = QLabel()
        window_verticalLayout.addWidget(self.size_label)
        self.link_label = QLabel()
        window_verticalLayout.addWidget(self.link_label)
        self.link_lineEdit = QLineEdit()
        window_verticalLayout.addWidget(self.link_lineEdit)
        self.save_as_label = QLabel()
        window_verticalLayout.addWidget(self.save_as_label)
        self.save_as_lineEdit = QLineEdit()
        window_verticalLayout.addWidget(self.save_as_lineEdit)
        button_horizontalLayout = QHBoxLayout()
        button_horizontalLayout.setContentsMargins(10, 10, 10, 10)
        button_horizontalLayout.addStretch(1)
        self.open_pushButtun = QPushButton()
        self.open_pushButtun.setIcon(QIcon(icons + 'file'))
        button_horizontalLayout.addWidget(self.open_pushButtun)
        self.open_folder_pushButtun = QPushButton()
        self.open_folder_pushButtun.setIcon(QIcon(icons + 'folder'))
        button_horizontalLayout.addWidget(self.open_folder_pushButtun)
        self.ok_pushButton = QPushButton()
        self.ok_pushButton.setIcon(QIcon(icons + 'ok'))
        button_horizontalLayout.addWidget(self.ok_pushButton)
        window_verticalLayout.addLayout(button_horizontalLayout)
        self.dont_show_checkBox = QCheckBox()
        window_verticalLayout.addWidget(self.dont_show_checkBox)
        window_verticalLayout.addStretch(1)
        self.setLayout(window_verticalLayout)
        self.open_pushButtun.setText(QCoreApplication.translate('after_download_ui_tr', '  Open File  '))
        self.open_folder_pushButtun.setText(QCoreApplication.translate('after_download_ui_tr', 'Open Download Folder'))
        self.ok_pushButton.setText(QCoreApplication.translate('after_download_ui_tr', '   OK   '))
        self.dont_show_checkBox.setText(QCoreApplication.translate('after_download_ui_tr', "Don't show this message again."))
        self.complete_label.setText(QCoreApplication.translate('after_download_ui_tr', '<b>Download Completed!</b>'))
        self.save_as_label.setText(QCoreApplication.translate('after_download_ui_tr', '<b>Save as</b>: '))
        self.link_label.setText(QCoreApplication.translate('after_download_ui_tr', '<b>Link</b>: '))