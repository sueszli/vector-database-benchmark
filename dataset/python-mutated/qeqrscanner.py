import os
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtGui import QGuiApplication
from electrum.util import send_exception_to_crash_reporter, UserFacingException
from electrum.simple_config import SimpleConfig
from electrum.logging import get_logger
from electrum.i18n import _
if 'ANDROID_DATA' in os.environ:
    from jnius import autoclass, cast
    from android import activity
    jpythonActivity = autoclass('org.kivy.android.PythonActivity').mActivity
    jString = autoclass('java.lang.String')
    jIntent = autoclass('android.content.Intent')

class QEQRScanner(QObject):
    _logger = get_logger(__name__)
    found = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._hint = _('Scan a QR code.')
        self._scan_data = ''

    @pyqtProperty(str)
    def hint(self):
        if False:
            i = 10
            return i + 15
        return self._hint

    @hint.setter
    def hint(self, v: str):
        if False:
            print('Hello World!')
        self._hint = v

    @pyqtProperty(str)
    def scanData(self):
        if False:
            for i in range(10):
                print('nop')
        return self._scan_data

    @scanData.setter
    def scanData(self, v: str):
        if False:
            while True:
                i = 10
        self._scan_data = v

    @pyqtSlot()
    def open(self):
        if False:
            for i in range(10):
                print('nop')
        if 'ANDROID_DATA' not in os.environ:
            self._scan_qr_non_android()
            return
        SimpleScannerActivity = autoclass('org.electrum.qr.SimpleScannerActivity')
        intent = jIntent(jpythonActivity, SimpleScannerActivity)
        intent.putExtra(jIntent.EXTRA_TEXT, jString(self._hint))

        def on_qr_result(requestCode, resultCode, intent):
            if False:
                print('Hello World!')
            try:
                if resultCode == -1:
                    contents = intent.getStringExtra(jString('text'))
                    self.scanData = contents
                    self.found.emit()
            except Exception as e:
                send_exception_to_crash_reporter(e)
            finally:
                activity.unbind(on_activity_result=on_qr_result)
        activity.bind(on_activity_result=on_qr_result)
        jpythonActivity.startActivityForResult(intent, 0)

    @pyqtSlot()
    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _scan_qr_non_android(self):
        if False:
            i = 10
            return i + 15
        data = QGuiApplication.clipboard().text()
        self.scanData = data
        self.found.emit()
        return