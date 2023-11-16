from typing import Tuple
from PyQt5.QtWidgets import QWidget
from autokey.qtui import common as ui_common
logger = ui_common.logger.getChild('DetectDialog')

class DetectDialog(*ui_common.inherits_from_ui_file_with_name('detectdialog')):
    """
    The DetectDialog lets the user select window properties of a chosen window.
    The dialog shows the window title and window class of the chosen window
    and lets the user select one of those two options.
    """

    def __init__(self, parent: QWidget):
        if False:
            print('Hello World!')
        super(DetectDialog, self).__init__(parent)
        self.setupUi(self)
        self.window_title = ''
        self.window_class = ''

    def populate(self, window_info: Tuple[str, str]):
        if False:
            print('Hello World!')
        (self.window_title, self.window_class) = window_info
        self.detected_title.setText(self.window_title)
        self.detected_class.setText(self.window_class)
        logger.info('Detected window with properties title: {}, window class: {}'.format(self.window_title, self.window_class))

    def get_choice(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.classButton.isChecked():
            logger.debug('User has chosen the window class: {}'.format(self.window_class))
            return self.window_class
        else:
            logger.debug('User has chosen the window title: {}'.format(self.window_title))
            return self.window_title