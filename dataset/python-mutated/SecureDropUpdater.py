import os
import re
import socket
import subprocess
import sys
import syslog as log
import pexpect
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from journalist_gui import resources_rc, strings, updaterUI
FLAG_LOCATION = '/home/amnesia/Persistent/.securedrop/securedrop_update.flag'
ESCAPE_POD = re.compile('\\x1B\\[[0-?]*[ -/]*[@-~]')

def password_is_set():
    if False:
        return 10
    pwd_flag = subprocess.check_output(['passwd', '--status']).decode('utf-8').split()[1]
    if pwd_flag == 'NP':
        return False
    return True

def prevent_second_instance(app: QtWidgets.QApplication, name: str) -> None:
    if False:
        print('Hello World!')
    IDENTIFIER = '\x00' + name
    ALREADY_BOUND_ERRNO = 98
    app.instance_binding = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        app.instance_binding.bind(IDENTIFIER)
    except OSError as e:
        if e.errno == ALREADY_BOUND_ERRNO:
            log.syslog(log.LOG_NOTICE, name + strings.app_is_already_running)
            sys.exit()
        else:
            raise

class SetupThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        if False:
            return 10
        QThread.__init__(self)
        self.output = ''
        self.update_success = False
        self.failure_reason = ''

    def run(self):
        if False:
            while True:
                i = 10
        sdadmin_path = '/home/amnesia/Persistent/securedrop/securedrop-admin'
        update_command = [sdadmin_path, 'setup']
        if not os.path.exists(FLAG_LOCATION):
            open(FLAG_LOCATION, 'a').close()
        try:
            self.output = subprocess.check_output(update_command, stderr=subprocess.STDOUT).decode('utf-8')
            if 'Failed to install' in self.output:
                self.update_success = False
                self.failure_reason = strings.update_failed_generic_reason
            else:
                self.update_success = True
        except subprocess.CalledProcessError as e:
            self.output += e.output.decode('utf-8')
            self.update_success = False
            self.failure_reason = strings.update_failed_generic_reason
        result = {'status': self.update_success, 'output': self.output, 'failure_reason': self.failure_reason}
        self.signal.emit(result)

class UpdateThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        QThread.__init__(self)
        self.output = ''
        self.update_success = False
        self.failure_reason = ''

    def run(self):
        if False:
            i = 10
            return i + 15
        sdadmin_path = '/home/amnesia/Persistent/securedrop/securedrop-admin'
        update_command = [sdadmin_path, 'update']
        try:
            self.output = subprocess.check_output(update_command, stderr=subprocess.STDOUT).decode('utf-8')
            if 'Signature verification successful' in self.output:
                self.update_success = True
            else:
                self.failure_reason = strings.update_failed_generic_reason
        except subprocess.CalledProcessError as e:
            self.update_success = False
            self.output += e.output.decode('utf-8')
            if 'Signature verification failed' in self.output:
                self.failure_reason = strings.update_failed_sig_failure
            else:
                self.failure_reason = strings.update_failed_generic_reason
        result = {'status': self.update_success, 'output': self.output, 'failure_reason': self.failure_reason}
        self.signal.emit(result)

class TailsconfigThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        QThread.__init__(self)
        self.output = ''
        self.update_success = False
        self.failure_reason = ''
        self.sudo_password = ''

    def run(self):
        if False:
            return 10
        tailsconfig_command = '/home/amnesia/Persistent/securedrop/securedrop-admin tailsconfig'
        self.failure_reason = ''
        try:
            child = pexpect.spawn(tailsconfig_command)
            child.expect('SUDO password:')
            self.output += child.before.decode('utf-8')
            child.sendline(self.sudo_password)
            child.expect(pexpect.EOF, timeout=120)
            self.output += child.before.decode('utf-8')
            child.close()
            if child.exitstatus:
                self.update_success = False
                if '[sudo via ansible' in self.output:
                    self.failure_reason = strings.tailsconfig_failed_sudo_password
                else:
                    self.failure_reason = strings.tailsconfig_failed_generic_reason
            else:
                self.update_success = True
        except pexpect.exceptions.TIMEOUT:
            self.update_success = False
            self.failure_reason = strings.tailsconfig_failed_timeout
        except subprocess.CalledProcessError:
            self.update_success = False
            self.failure_reason = strings.tailsconfig_failed_generic_reason
        result = {'status': self.update_success, 'output': ESCAPE_POD.sub('', self.output), 'failure_reason': self.failure_reason}
        self.signal.emit(result)

class UpdaterApp(QtWidgets.QMainWindow, updaterUI.Ui_MainWindow):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setupUi(self)
        self.statusbar.setSizeGripEnabled(False)
        self.output = strings.initial_text_box
        self.plainTextEdit.setPlainText(self.output)
        self.update_success = False
        pixmap = QtGui.QPixmap(':/images/static/banner.png')
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)
        self.progressBar.setProperty('value', 0)
        self.setWindowTitle(strings.window_title)
        self.setWindowIcon(QtGui.QIcon(':/images/static/securedrop_icon.png'))
        self.label.setText(strings.update_in_progress)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), strings.main_tab)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), strings.output_tab)
        self.pushButton.setText(strings.install_later_button)
        self.pushButton.setStyleSheet('background-color: lightgrey;\n                                      min-height: 2em;\n                                      border-radius: 10px')
        self.pushButton.clicked.connect(self.close)
        self.pushButton_2.setText(strings.install_update_button)
        self.pushButton_2.setStyleSheet('background-color: #E6FFEB;\n                                        min-height: 2em;\n                                        border-radius: 10px;')
        self.pushButton_2.clicked.connect(self.update_securedrop)
        self.update_thread = UpdateThread()
        self.update_thread.signal.connect(self.update_status)
        self.tails_thread = TailsconfigThread()
        self.tails_thread.signal.connect(self.tails_status)
        self.setup_thread = SetupThread()
        self.setup_thread.signal.connect(self.setup_status)

    def setup_status(self, result):
        if False:
            return 10
        'This is the slot for setup thread'
        self.output += result['output']
        self.update_success = result['status']
        self.failure_reason = result['failure_reason']
        self.progressBar.setProperty('value', 60)
        self.plainTextEdit.setPlainText(self.output)
        self.plainTextEdit.setReadOnly = True
        if not self.update_success:
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)
            self.update_status_bar_and_output(self.failure_reason)
            self.progressBar.setProperty('value', 0)
            self.alert_failure(self.failure_reason)
            return
        self.progressBar.setProperty('value', 70)
        self.call_tailsconfig()

    def update_status(self, result):
        if False:
            print('Hello World!')
        'This is the slot for update thread'
        self.output += result['output']
        self.update_success = result['status']
        self.failure_reason = result['failure_reason']
        self.progressBar.setProperty('value', 40)
        self.plainTextEdit.setPlainText(self.output)
        self.plainTextEdit.setReadOnly = True
        if not self.update_success:
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)
            self.update_status_bar_and_output(self.failure_reason)
            self.progressBar.setProperty('value', 0)
            self.alert_failure(self.failure_reason)
            return
        self.progressBar.setProperty('value', 50)
        self.update_status_bar_and_output(strings.doing_setup)
        self.setup_thread.start()

    def update_status_bar_and_output(self, status_message):
        if False:
            print('Hello World!')
        'This method updates the status bar and the output window with the\n        status_message.'
        self.statusbar.showMessage(status_message)
        self.output += status_message + '\n'
        self.plainTextEdit.setPlainText(self.output)

    def call_tailsconfig(self):
        if False:
            for i in range(10):
                print('nop')
        if self.update_success:
            sudo_password = self.get_sudo_password()
            if not sudo_password:
                self.update_success = False
                self.failure_reason = strings.missing_sudo_password
                self.on_failure()
                return
            self.tails_thread.sudo_password = sudo_password + '\n'
            self.update_status_bar_and_output(strings.updating_tails_env)
            self.tails_thread.start()
        else:
            self.on_failure()

    def tails_status(self, result):
        if False:
            return 10
        'This is the slot for Tailsconfig thread'
        self.output += result['output']
        self.update_success = result['status']
        self.failure_reason = result['failure_reason']
        self.plainTextEdit.setPlainText(self.output)
        self.progressBar.setProperty('value', 80)
        if self.update_success:
            os.remove(FLAG_LOCATION)
            self.update_status_bar_and_output(strings.finished)
            self.progressBar.setProperty('value', 100)
            self.alert_success()
        else:
            self.on_failure()

    def on_failure(self):
        if False:
            return 10
        self.update_status_bar_and_output(self.failure_reason)
        self.alert_failure(self.failure_reason)
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.progressBar.setProperty('value', 0)

    def update_securedrop(self):
        if False:
            for i in range(10):
                print('nop')
        if password_is_set():
            self.pushButton_2.setEnabled(False)
            self.pushButton.setEnabled(False)
            self.progressBar.setProperty('value', 10)
            self.update_status_bar_and_output(strings.fetching_update)
            self.update_thread.start()
        else:
            self.pushButton_2.setEnabled(False)
            pwd_err_dialog = QtWidgets.QMessageBox()
            pwd_err_dialog.setText(strings.no_password_set_message)
            pwd_err_dialog.exec()

    def alert_success(self):
        if False:
            for i in range(10):
                print('nop')
        self.success_dialog = QtWidgets.QMessageBox()
        self.success_dialog.setIcon(QtWidgets.QMessageBox.Information)
        self.success_dialog.setText(strings.finished_dialog_message)
        self.success_dialog.setWindowTitle(strings.finished_dialog_title)
        self.success_dialog.show()

    def alert_failure(self, failure_reason):
        if False:
            for i in range(10):
                print('nop')
        self.error_dialog = QtWidgets.QMessageBox()
        self.error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        self.error_dialog.setText(self.failure_reason)
        self.error_dialog.setWindowTitle(strings.update_failed_dialog_title)
        self.error_dialog.show()

    def get_sudo_password(self):
        if False:
            for i in range(10):
                print('nop')
        (sudo_password, ok_is_pressed) = QtWidgets.QInputDialog.getText(self, 'Tails Administrator password', strings.sudo_password_text, QtWidgets.QLineEdit.Password, '')
        if ok_is_pressed and sudo_password:
            return sudo_password
        else:
            return None