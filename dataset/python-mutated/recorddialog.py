from autokey.qtui import common as ui_common
logger = __import__('autokey.logger').logger.get_logger(__name__)

class RecordDialog(*ui_common.inherits_from_ui_file_with_name('record_dialog')):

    def __init__(self, parent, closure):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setupUi(self)
        self.closure = closure

    def get_record_keyboard(self):
        if False:
            while True:
                i = 10
        return self.record_keyboard_button.isChecked()

    def get_record_mouse(self):
        if False:
            print('Hello World!')
        return self.record_mouse_button.isChecked()

    def get_delay(self):
        if False:
            i = 10
            return i + 15
        return self.delay_recording_start_seconds_spin_box.value()

    def accept(self):
        if False:
            print('Hello World!')
        super().accept()
        logger.info('Dialog accepted: Record keyboard: {}, record mouse: {}, delay: {} s'.format(self.get_record_keyboard(), self.get_record_mouse(), self.get_delay()))
        self.closure(True, self.get_record_keyboard(), self.get_record_mouse(), self.get_delay())

    def reject(self):
        if False:
            while True:
                i = 10
        super().reject()
        logger.info('Dialog closed (rejected/aborted): Record keyboard: {}, record mouse: {}, delay: {} s'.format(self.get_record_keyboard(), self.get_record_mouse(), self.get_delay()))
        self.closure(False, self.get_record_keyboard(), self.get_record_mouse(), self.get_delay())