import re
from PyQt5.QtWidgets import QDialog
import autokey.iomediator.windowgrabber
import autokey.model.folder
import autokey.model.modelTypes
from autokey.qtui import common as qtui_common
from autokey import UI_common_functions as UI_common
from .detectdialog import DetectDialog
from autokey import iomediator
logger = __import__('autokey.logger').logger.get_logger(__name__)

class WindowFilterSettingsDialog(*qtui_common.inherits_from_ui_file_with_name('window_filter_settings_dialog')):

    def __init__(self, parent):
        if False:
            return 10
        super(WindowFilterSettingsDialog, self).__init__(parent)
        self.setupUi(self)
        self.target_item = None
        self.grabber = None

    def load(self, item: autokey.model.modelTypes.Item):
        if False:
            while True:
                i = 10
        self.target_item = item
        if not isinstance(item, autokey.model.folder.Folder):
            self.apply_recursive_check_box.hide()
        else:
            self.apply_recursive_check_box.show()
        if not item.has_filter():
            self.reset()
        else:
            self.trigger_regex_line_edit.setText(item.get_filter_regex())
            self.apply_recursive_check_box.setChecked(item.isRecursive)

    def save(self, item):
        if False:
            for i in range(10):
                print('nop')
        UI_common.save_item_filter(self, item)

    def get_is_recursive(self):
        if False:
            for i in range(10):
                print('nop')
        return self.apply_recursive_check_box.isChecked()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.trigger_regex_line_edit.clear()
        self.apply_recursive_check_box.setChecked(False)

    def reset_focus(self):
        if False:
            print('Hello World!')
        self.trigger_regex_line_edit.setFocus()

    def get_filter_text(self):
        if False:
            i = 10
            return i + 15
        return str(self.trigger_regex_line_edit.text())

    def receive_window_info(self, info):
        if False:
            return 10
        self.parentWidget().window().app.exec_in_main(self._receiveWindowInfo, info)

    def _receiveWindowInfo(self, info):
        if False:
            return 10
        dlg = DetectDialog(self)
        dlg.populate(info)
        dlg.exec_()
        if dlg.result() == QDialog.Accepted:
            self.trigger_regex_line_edit.setText(dlg.get_choice())
        self.detect_window_properties_button.setEnabled(True)

    def on_detect_window_properties_button_pressed(self):
        if False:
            print('Hello World!')
        self.detect_window_properties_button.setEnabled(False)
        self.grabber = iomediator.windowgrabber.WindowGrabber(self)
        self.grabber.start()

    def slotButtonClicked(self, button):
        if False:
            while True:
                i = 10
        if button == QDialog.Cancel:
            self.load(self.targetItem)
        QDialog.slotButtonClicked(self, button)