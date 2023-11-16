import sys
try:
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QCheckBox, QSystemTrayIcon, QSpacerItem, QSizePolicy, QMenu, QAction, QStyle
except ImportError:
    from PySide2.QtCore import QSize
    from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QCheckBox, QSystemTrayIcon, QSpacerItem, QSizePolicy, QMenu, QAction, QStyle

class MainWindow(QMainWindow):
    """
         Сheckbox and system tray icons.
         Will initialize in the constructor.
    """
    check_box = None
    tray_icon = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(480, 80))
        self.setWindowTitle('System Tray Application')
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        grid_layout = QGridLayout(self)
        central_widget.setLayout(grid_layout)
        grid_layout.addWidget(QLabel('Application, which can minimize to Tray', self), 0, 0)
        self.check_box = QCheckBox('Minimize to Tray')
        grid_layout.addWidget(self.check_box, 1, 0)
        grid_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding), 2, 0)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        '\n            Define and add steps to work with the system tray icon\n            show - show window\n            hide - hide window\n            exit - exit from application\n        '
        show_action = QAction('Show', self)
        quit_action = QAction('Exit', self)
        hide_action = QAction('Hide', self)
        show_action.triggered.connect(self.show)
        hide_action.triggered.connect(self.hide)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu = QMenu()
        tray_menu.addAction(show_action)
        tray_menu.addAction(hide_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def closeEvent(self, event):
        if False:
            return 10
        if self.check_box.isChecked():
            event.ignore()
            self.hide()
            self.tray_icon.showMessage('Tray Program', 'Application was minimized to Tray', QSystemTrayIcon.Information, 2000)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())