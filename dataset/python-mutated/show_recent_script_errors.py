import typing
from PyQt5.QtWidgets import QApplication, QAbstractButton, QDialogButtonBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from autokey.model.script import ScriptErrorRecord
from autokey.qtui import common as ui_common
logger = __import__('autokey.logger').logger.get_logger(__name__)

class ShowRecentScriptErrorsDialog(*ui_common.inherits_from_ui_file_with_name('show_recent_script_errors_dialog')):
    """
    This dialogue is used to show errors that caused user Scripts to abort. The ScriptRunner class holds a list
    with errors, which can be viewed and cleared using this dialogue window.

    """
    has_previous_error = pyqtSignal(bool, name='has_previous_error')
    has_next_error = pyqtSignal(bool, name='has_next_error')
    script_errors_available = pyqtSignal(bool, name='script_errors_available')

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super(ShowRecentScriptErrorsDialog, self).__init__(parent)
        self.setupUi(self)
        self.stack_trace_text_browser.setFontFamily('monospace')
        self.currently_shown_error_number_spin_box_suffix = self.currently_shown_error_number_spin_box.suffix()
        self.recent_script_errors = QApplication.instance().service.scriptRunner.error_records
        self.currently_viewed_error_index = 0

    @property
    def total_error_count(self):
        if False:
            print('Hello World!')
        return len(self.recent_script_errors)

    def _emit_has_next_error(self):
        if False:
            while True:
                i = 10
        has_next_error = self.currently_viewed_error_index < self.total_error_count - 1
        self.has_next_error.emit(has_next_error)

    def _emit_has_previous_error(self):
        if False:
            while True:
                i = 10
        has_previous_error = self.currently_viewed_error_index > 0
        self.has_previous_error.emit(has_previous_error)

    def _emit_script_errors_available(self):
        if False:
            return 10
        script_errors_available = bool(self.recent_script_errors)
        self.script_errors_available.emit(script_errors_available)

    def hide(self):
        if False:
            return 10
        self._emit_script_errors_available()
        super(ShowRecentScriptErrorsDialog, self).hide()

    @pyqtSlot(QAbstractButton)
    def handle_button_box_button_clicks(self, clicked_button: QAbstractButton):
        if False:
            i = 10
            return i + 15
        "\n        Used to connect the button presses with logic for 'Reset error list' and 'Discard current error' buttons in\n        the button box at the bottom of the dialogue window.\n        "
        button_role = self.buttonBox.buttonRole(clicked_button)
        if button_role == QDialogButtonBox.DestructiveRole:
            self.remove_currently_shown_error_from_error_list()
        elif button_role == QDialogButtonBox.ResetRole:
            self.clear_error_list_and_hide()

    @pyqtSlot()
    def update_and_show(self):
        if False:
            while True:
                i = 10
        error_count = self.total_error_count
        if error_count:
            if self.currently_viewed_error_index >= error_count:
                self.currently_viewed_error_index = error_count - 1
            self._emit_has_next_error()
            self._emit_has_previous_error()
            logger.info('User views the last script errors. There are {} errors to review.'.format(error_count))
            self._show_currently_viewed_error()
            self.show()
        else:
            logger.error('User is able to view the script error dialogue, even if no errors are available. This should be impossible. Do not show the dialogue window.')

    @pyqtSlot()
    def show_next_error(self):
        if False:
            return 10
        '\n        Switch to the next error in the error list.\n        The connection from the Next button to this slot function is defined in the .ui file.\n        '
        self.currently_viewed_error_index += 1
        self._emit_has_next_error()
        self._show_currently_viewed_error()

    @pyqtSlot()
    def show_previous_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Switch to the previous error in the error list.\n        The connection from the Previous button to this slot function is defined in the .ui file.\n        '
        self.currently_viewed_error_index -= 1
        self._emit_has_previous_error()
        self._show_currently_viewed_error()

    @pyqtSlot()
    def show_last_error(self):
        if False:
            print('Hello World!')
        '\n        Switch to the last error in the error list.\n        The connection from the Last button to this slot function is defined in the .ui file.\n        '
        self.currently_viewed_error_index = self.total_error_count - 1
        self._emit_has_next_error()
        self._show_currently_viewed_error()

    @pyqtSlot()
    def show_first_error(self):
        if False:
            return 10
        '\n        Switch to the first error in the error list.\n        The connection from the First button to this slot function is defined in the .ui file.\n        '
        self.currently_viewed_error_index = 0
        self._emit_has_previous_error()
        self._show_currently_viewed_error()

    @pyqtSlot(int)
    def show_error_at_index(self, error_index: int):
        if False:
            while True:
                i = 10
        '\n        Switch to a specific error in the error list. Subtract one, because the GUI uses a 1-based index.\n        The connection from the error number spin box to this slot function is defined in the .ui file.\n        '
        self.currently_viewed_error_index = error_index - 1
        self._emit_has_next_error()
        self._emit_has_previous_error()
        self._show_currently_viewed_error()

    def remove_currently_shown_error_from_error_list(self):
        if False:
            print('Hello World!')
        '\n        Delete the currently shown error from the error list.\n        Shows the next error, if available. Otherwise, show the previous error, if available.\n        Or clear the list and hide the window, if the deleted error was the only one in the list.\n        '
        if self.total_error_count == 1:
            self.clear_error_list_and_hide()
        else:
            del self.recent_script_errors[self.currently_viewed_error_index]
            if self.currently_viewed_error_index == self.total_error_count - 1:
                self.show_previous_error()
            else:
                self._emit_has_next_error()
                self._show_currently_viewed_error()

    def clear_error_list_and_hide(self):
        if False:
            return 10
        'Clears all errors in the error list and hides the dialogue window.'
        self.currently_viewed_error_index = 0
        QApplication.instance().service.scriptRunner.clear_error_records()
        self.hide()

    def _show_currently_viewed_error(self):
        if False:
            print('Hello World!')
        'Update the GUI to show the error at the current list index.'
        script_error = self.recent_script_errors[self.currently_viewed_error_index]
        error_count = self.total_error_count
        logger.debug('User views error {} / {}.'.format(self.currently_viewed_error_index + 1, error_count))
        self.currently_shown_error_number_spin_box.setMaximum(error_count)
        self.currently_shown_error_number_spin_box.setValue(self.currently_viewed_error_index + 1)
        self.currently_shown_error_number_spin_box.setSuffix(self.currently_shown_error_number_spin_box_suffix.format(error_count))
        self.script_start_time_edit.setTime(script_error.start_time)
        self.script_error_time_edit.setTime(script_error.error_time)
        self.script_name_view.setText(script_error.script_name)
        self.stack_trace_text_browser.setText(script_error.error_traceback)