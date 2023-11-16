from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QWidget
from .qrcodewidget import QRCodeWidget
from electrum.i18n import _

class QR_Window(QWidget):

    def __init__(self, win):
        if False:
            print('Hello World!')
        QWidget.__init__(self)
        self.main_window = win
        self.setWindowTitle('Electrum - ' + _('Payment Request'))
        self.setMinimumSize(800, 800)
        self.setFocusPolicy(Qt.NoFocus)
        main_box = QHBoxLayout()
        self.qrw = QRCodeWidget()
        main_box.addWidget(self.qrw, 1)
        self.setLayout(main_box)

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        self.main_window.receive_tab.qr_menu_action.setChecked(False)