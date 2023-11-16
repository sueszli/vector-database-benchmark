from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QObject
from electrum.logging import get_logger
from electrum.util import bfh, format_time
from .qetypes import QEAmount
from .qewallet import QEWallet

class QELnPaymentDetails(QObject):
    _logger = get_logger(__name__)
    detailsChanged = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._wallet = None
        self._key = None
        self._label = ''
        self._date = None
        self._timestamp = 0
        self._fee = QEAmount()
        self._amount = QEAmount()
        self._status = ''
        self._phash = ''
        self._preimage = ''
    walletChanged = pyqtSignal()

    @pyqtProperty(QEWallet, notify=walletChanged)
    def wallet(self):
        if False:
            print('Hello World!')
        return self._wallet

    @wallet.setter
    def wallet(self, wallet: QEWallet):
        if False:
            return 10
        if self._wallet != wallet:
            self._wallet = wallet
            self.walletChanged.emit()
    keyChanged = pyqtSignal()

    @pyqtProperty(str, notify=keyChanged)
    def key(self):
        if False:
            print('Hello World!')
        return self._key

    @key.setter
    def key(self, key: str):
        if False:
            for i in range(10):
                print('nop')
        if self._key != key:
            self._logger.debug(f'key set -> {key}')
            self._key = key
            self.keyChanged.emit()
            self.update()
    labelChanged = pyqtSignal()

    @pyqtProperty(str, notify=labelChanged)
    def label(self):
        if False:
            i = 10
            return i + 15
        return self._label

    @pyqtSlot(str)
    def setLabel(self, label: str):
        if False:
            return 10
        if label != self._label:
            self._wallet.wallet.set_label(self._key, label)
            self._label = label
            self.labelChanged.emit()

    @pyqtProperty(str, notify=detailsChanged)
    def status(self):
        if False:
            while True:
                i = 10
        return self._status

    @pyqtProperty(str, notify=detailsChanged)
    def date(self):
        if False:
            i = 10
            return i + 15
        return self._date

    @pyqtProperty(int, notify=detailsChanged)
    def timestamp(self):
        if False:
            i = 10
            return i + 15
        return self._timestamp

    @pyqtProperty(str, notify=detailsChanged)
    def paymentHash(self):
        if False:
            return 10
        return self._phash

    @pyqtProperty(str, notify=detailsChanged)
    def preimage(self):
        if False:
            print('Hello World!')
        return self._preimage

    @pyqtProperty(QEAmount, notify=detailsChanged)
    def amount(self):
        if False:
            i = 10
            return i + 15
        return self._amount

    @pyqtProperty(QEAmount, notify=detailsChanged)
    def fee(self):
        if False:
            print('Hello World!')
        return self._fee

    def update(self):
        if False:
            print('Hello World!')
        if self._wallet is None:
            self._logger.error('wallet undefined')
            return
        tx = self._wallet.wallet.lnworker.get_lightning_history()[bfh(self._key)]
        self._logger.debug(str(tx))
        self._fee.msatsInt = 0 if not tx['fee_msat'] else int(tx['fee_msat'])
        self._amount.msatsInt = int(tx['amount_msat'])
        self._label = tx['label']
        self._date = format_time(tx['timestamp'])
        self._timestamp = tx['timestamp']
        self._status = 'settled'
        self._phash = tx['payment_hash']
        self._preimage = tx['preimage']
        self.detailsChanged.emit()