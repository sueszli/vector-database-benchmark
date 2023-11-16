import sys
import html
from typing import TYPE_CHECKING, Optional, Set
from PyQt5.QtCore import QObject
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QTextEdit, QMessageBox, QHBoxLayout, QVBoxLayout
from electrum.i18n import _
from electrum.base_crash_reporter import BaseCrashReporter, EarlyExceptionsQueue, CrashReportResponse
from electrum.logging import Logger
from electrum import constants
from electrum.network import Network
from .util import MessageBoxMixin, read_QIcon, WaitingDialog, font_height
if TYPE_CHECKING:
    from electrum.simple_config import SimpleConfig
    from electrum.wallet import Abstract_Wallet

class Exception_Window(BaseCrashReporter, QWidget, MessageBoxMixin, Logger):
    _active_window = None

    def __init__(self, config: 'SimpleConfig', exctype, value, tb):
        if False:
            print('Hello World!')
        BaseCrashReporter.__init__(self, exctype, value, tb)
        self.network = Network.get_instance()
        self.config = config
        QWidget.__init__(self)
        self.setWindowTitle('Electrum - ' + _('An Error Occurred'))
        self.setMinimumSize(600, 300)
        Logger.__init__(self)
        main_box = QVBoxLayout()
        heading = QLabel('<h2>' + BaseCrashReporter.CRASH_TITLE + '</h2>')
        main_box.addWidget(heading)
        main_box.addWidget(QLabel(BaseCrashReporter.CRASH_MESSAGE))
        main_box.addWidget(QLabel(BaseCrashReporter.REQUEST_HELP_MESSAGE))
        collapse_info = QPushButton(_('Show report contents'))
        collapse_info.clicked.connect(lambda : self.msg_box(QMessageBox.NoIcon, self, _('Report contents'), self.get_report_string(), rich_text=True))
        main_box.addWidget(collapse_info)
        main_box.addWidget(QLabel(BaseCrashReporter.DESCRIBE_ERROR_MESSAGE))
        self.description_textfield = QTextEdit()
        self.description_textfield.setFixedHeight(4 * font_height())
        self.description_textfield.setPlaceholderText(self.USER_COMMENT_PLACEHOLDER)
        main_box.addWidget(self.description_textfield)
        main_box.addWidget(QLabel(BaseCrashReporter.ASK_CONFIRM_SEND))
        buttons = QHBoxLayout()
        report_button = QPushButton(_('Send Bug Report'))
        report_button.clicked.connect(self.send_report)
        report_button.setIcon(read_QIcon('tab_send.png'))
        buttons.addWidget(report_button)
        never_button = QPushButton(_('Never'))
        never_button.clicked.connect(self.show_never)
        buttons.addWidget(never_button)
        close_button = QPushButton(_('Not Now'))
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        main_box.addLayout(buttons)
        self.setLayout(main_box)
        self.show()

    def send_report(self):
        if False:
            return 10

        def on_success(response: CrashReportResponse):
            if False:
                print('Hello World!')
            text = response.text
            if response.url:
                text += f" You can track further progress on <a href='{response.url}'>GitHub</a>."
            self.show_message(parent=self, title=_('Crash report'), msg=text, rich_text=True)
            self.close()

        def on_failure(exc_info):
            if False:
                for i in range(10):
                    print('nop')
            e = exc_info[1]
            self.logger.error('There was a problem with the automatic reporting', exc_info=exc_info)
            self.show_critical(parent=self, msg=_('There was a problem with the automatic reporting:') + '<br/>' + repr(e)[:120] + '<br/><br/>' + _('Please report this issue manually') + f' <a href="{constants.GIT_REPO_ISSUES_URL}">on GitHub</a>.', rich_text=True)
        proxy = self.network.proxy
        task = lambda : BaseCrashReporter.send_report(self, self.network.asyncio_loop, proxy)
        msg = _('Sending crash report...')
        WaitingDialog(self, msg, task, on_success, on_failure)

    def on_close(self):
        if False:
            print('Hello World!')
        Exception_Window._active_window = None
        self.close()

    def show_never(self):
        if False:
            while True:
                i = 10
        self.config.SHOW_CRASH_REPORTER = False
        self.close()

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        self.on_close()
        event.accept()

    def get_user_description(self):
        if False:
            return 10
        return self.description_textfield.toPlainText()

    def get_wallet_type(self):
        if False:
            while True:
                i = 10
        wallet_types = Exception_Hook._INSTANCE.wallet_types_seen
        return ','.join(wallet_types)

    def _get_traceback_str_to_display(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        traceback_str = super()._get_traceback_str_to_display()
        return html.escape(traceback_str)

def _show_window(*args):
    if False:
        while True:
            i = 10
    if not Exception_Window._active_window:
        Exception_Window._active_window = Exception_Window(*args)

class Exception_Hook(QObject, Logger):
    _report_exception = QtCore.pyqtSignal(object, object, object, object)
    _INSTANCE = None

    def __init__(self, *, config: 'SimpleConfig'):
        if False:
            print('Hello World!')
        QObject.__init__(self)
        Logger.__init__(self)
        assert self._INSTANCE is None, 'Exception_Hook is supposed to be a singleton'
        self.config = config
        self.wallet_types_seen = set()
        sys.excepthook = self.handler
        self._report_exception.connect(_show_window)
        EarlyExceptionsQueue.set_hook_as_ready()

    @classmethod
    def maybe_setup(cls, *, config: 'SimpleConfig', wallet: 'Abstract_Wallet'=None) -> None:
        if False:
            return 10
        if not config.SHOW_CRASH_REPORTER:
            EarlyExceptionsQueue.set_hook_as_ready()
            return
        if not cls._INSTANCE:
            cls._INSTANCE = Exception_Hook(config=config)
        if wallet:
            cls._INSTANCE.wallet_types_seen.add(wallet.wallet_type)

    def handler(self, *exc_info):
        if False:
            while True:
                i = 10
        self.logger.error('exception caught by crash reporter', exc_info=exc_info)
        self._report_exception.emit(self.config, *exc_info)