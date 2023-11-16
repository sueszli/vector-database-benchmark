from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from tribler.gui.utilities import get_ui_file_path

class QtBug(QWidget):

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        uic.loadUi(get_ui_file_path('qtbug.ui'), self)