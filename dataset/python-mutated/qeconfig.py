import copy
from decimal import Decimal
from typing import TYPE_CHECKING
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QObject, QRegularExpression
from electrum.bitcoin import TOTAL_COIN_SUPPLY_LIMIT_IN_BTC
from electrum.i18n import set_language, languages
from electrum.logging import get_logger
from electrum.util import base_unit_name_to_decimal_point
from .qetypes import QEAmount
from .auth import AuthMixin, auth_protect
if TYPE_CHECKING:
    from electrum.simple_config import SimpleConfig

class QEConfig(AuthMixin, QObject):
    _logger = get_logger(__name__)

    def __init__(self, config: 'SimpleConfig', parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.config = config
    languageChanged = pyqtSignal()

    @pyqtProperty(str, notify=languageChanged)
    def language(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.LOCALIZATION_LANGUAGE

    @language.setter
    def language(self, language):
        if False:
            return 10
        if language not in languages:
            return
        if self.config.LOCALIZATION_LANGUAGE != language:
            self.config.LOCALIZATION_LANGUAGE = language
            set_language(language)
            self.languageChanged.emit()
    languagesChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=languagesChanged)
    def languagesAvailable(self):
        if False:
            i = 10
            return i + 15
        langs = copy.deepcopy(languages)
        default = langs.pop('')
        langs_sorted = sorted(list(map(lambda x: {'value': x[0], 'text': x[1]}, langs.items())), key=lambda x: x['text'])
        langs_sorted.insert(0, {'value': '', 'text': default})
        return langs_sorted
    autoConnectChanged = pyqtSignal()

    @pyqtProperty(bool, notify=autoConnectChanged)
    def autoConnect(self):
        if False:
            while True:
                i = 10
        return self.config.NETWORK_AUTO_CONNECT

    @autoConnect.setter
    def autoConnect(self, auto_connect):
        if False:
            for i in range(10):
                print('nop')
        self.config.NETWORK_AUTO_CONNECT = auto_connect
        self.autoConnectChanged.emit()

    @pyqtProperty(bool, notify=autoConnectChanged)
    def autoConnectDefined(self):
        if False:
            print('Hello World!')
        return self.config.cv.NETWORK_AUTO_CONNECT.is_set()
    baseUnitChanged = pyqtSignal()

    @pyqtProperty(str, notify=baseUnitChanged)
    def baseUnit(self):
        if False:
            return 10
        return self.config.get_base_unit()

    @baseUnit.setter
    def baseUnit(self, unit):
        if False:
            for i in range(10):
                print('nop')
        self.config.set_base_unit(unit)
        self.baseUnitChanged.emit()

    @pyqtProperty('QRegularExpression', notify=baseUnitChanged)
    def btcAmountRegex(self):
        if False:
            for i in range(10):
                print('nop')
        decimal_point = base_unit_name_to_decimal_point(self.config.get_base_unit())
        max_digits_before_dp = len(str(TOTAL_COIN_SUPPLY_LIMIT_IN_BTC)) + (base_unit_name_to_decimal_point('BTC') - decimal_point)
        exp = '[0-9]{0,%d}' % max_digits_before_dp
        if decimal_point > 0:
            exp += '\\.'
            exp += '[0-9]{0,%d}' % decimal_point
        return QRegularExpression(exp)
    thousandsSeparatorChanged = pyqtSignal()

    @pyqtProperty(bool, notify=thousandsSeparatorChanged)
    def thousandsSeparator(self):
        if False:
            return 10
        return self.config.BTC_AMOUNTS_ADD_THOUSANDS_SEP

    @thousandsSeparator.setter
    def thousandsSeparator(self, checked):
        if False:
            while True:
                i = 10
        self.config.BTC_AMOUNTS_ADD_THOUSANDS_SEP = checked
        self.config.amt_add_thousands_sep = checked
        self.thousandsSeparatorChanged.emit()
    spendUnconfirmedChanged = pyqtSignal()

    @pyqtProperty(bool, notify=spendUnconfirmedChanged)
    def spendUnconfirmed(self):
        if False:
            while True:
                i = 10
        return not self.config.WALLET_SPEND_CONFIRMED_ONLY

    @spendUnconfirmed.setter
    def spendUnconfirmed(self, checked):
        if False:
            for i in range(10):
                print('nop')
        self.config.WALLET_SPEND_CONFIRMED_ONLY = not checked
        self.spendUnconfirmedChanged.emit()
    requestExpiryChanged = pyqtSignal()

    @pyqtProperty(int, notify=requestExpiryChanged)
    def requestExpiry(self):
        if False:
            while True:
                i = 10
        return self.config.WALLET_PAYREQ_EXPIRY_SECONDS

    @requestExpiry.setter
    def requestExpiry(self, expiry):
        if False:
            while True:
                i = 10
        self.config.WALLET_PAYREQ_EXPIRY_SECONDS = expiry
        self.requestExpiryChanged.emit()
    pinCodeChanged = pyqtSignal()

    @pyqtProperty(str, notify=pinCodeChanged)
    def pinCode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.CONFIG_PIN_CODE or ''

    @pinCode.setter
    def pinCode(self, pin_code):
        if False:
            print('Hello World!')
        if pin_code == '':
            self.pinCodeRemoveAuth()
        else:
            self.config.CONFIG_PIN_CODE = pin_code
            self.pinCodeChanged.emit()

    @auth_protect(method='wallet')
    def pinCodeRemoveAuth(self):
        if False:
            while True:
                i = 10
        self.config.CONFIG_PIN_CODE = ''
        self.pinCodeChanged.emit()
    useGossipChanged = pyqtSignal()

    @pyqtProperty(bool, notify=useGossipChanged)
    def useGossip(self):
        if False:
            return 10
        return self.config.LIGHTNING_USE_GOSSIP

    @useGossip.setter
    def useGossip(self, gossip):
        if False:
            while True:
                i = 10
        self.config.LIGHTNING_USE_GOSSIP = gossip
        self.useGossipChanged.emit()
    useFallbackAddressChanged = pyqtSignal()

    @pyqtProperty(bool, notify=useFallbackAddressChanged)
    def useFallbackAddress(self):
        if False:
            while True:
                i = 10
        return self.config.WALLET_BOLT11_FALLBACK

    @useFallbackAddress.setter
    def useFallbackAddress(self, use_fallback):
        if False:
            while True:
                i = 10
        self.config.WALLET_BOLT11_FALLBACK = use_fallback
        self.useFallbackAddressChanged.emit()
    enableDebugLogsChanged = pyqtSignal()

    @pyqtProperty(bool, notify=enableDebugLogsChanged)
    def enableDebugLogs(self):
        if False:
            print('Hello World!')
        gui_setting = self.config.GUI_ENABLE_DEBUG_LOGS
        return gui_setting or bool(self.config.get('verbosity'))

    @pyqtProperty(bool, notify=enableDebugLogsChanged)
    def canToggleDebugLogs(self):
        if False:
            print('Hello World!')
        gui_setting = self.config.GUI_ENABLE_DEBUG_LOGS
        return not self.config.get('verbosity') or gui_setting

    @enableDebugLogs.setter
    def enableDebugLogs(self, enable):
        if False:
            for i in range(10):
                print('nop')
        self.config.GUI_ENABLE_DEBUG_LOGS = enable
        self.enableDebugLogsChanged.emit()
    useRecoverableChannelsChanged = pyqtSignal()

    @pyqtProperty(bool, notify=useRecoverableChannelsChanged)
    def useRecoverableChannels(self):
        if False:
            print('Hello World!')
        return self.config.LIGHTNING_USE_RECOVERABLE_CHANNELS

    @useRecoverableChannels.setter
    def useRecoverableChannels(self, useRecoverableChannels):
        if False:
            while True:
                i = 10
        self.config.LIGHTNING_USE_RECOVERABLE_CHANNELS = useRecoverableChannels
        self.useRecoverableChannelsChanged.emit()
    trustedcoinPrepayChanged = pyqtSignal()

    @pyqtProperty(int, notify=trustedcoinPrepayChanged)
    def trustedcoinPrepay(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.PLUGIN_TRUSTEDCOIN_NUM_PREPAY

    @trustedcoinPrepay.setter
    def trustedcoinPrepay(self, num_prepay):
        if False:
            print('Hello World!')
        if num_prepay != self.config.PLUGIN_TRUSTEDCOIN_NUM_PREPAY:
            self.config.PLUGIN_TRUSTEDCOIN_NUM_PREPAY = num_prepay
            self.trustedcoinPrepayChanged.emit()
    preferredRequestTypeChanged = pyqtSignal()

    @pyqtProperty(str, notify=preferredRequestTypeChanged)
    def preferredRequestType(self):
        if False:
            while True:
                i = 10
        return self.config.GUI_QML_PREFERRED_REQUEST_TYPE

    @preferredRequestType.setter
    def preferredRequestType(self, preferred_request_type):
        if False:
            i = 10
            return i + 15
        if preferred_request_type != self.config.GUI_QML_PREFERRED_REQUEST_TYPE:
            self.config.GUI_QML_PREFERRED_REQUEST_TYPE = preferred_request_type
            self.preferredRequestTypeChanged.emit()
    userKnowsPressAndHoldChanged = pyqtSignal()

    @pyqtProperty(bool, notify=userKnowsPressAndHoldChanged)
    def userKnowsPressAndHold(self):
        if False:
            print('Hello World!')
        return self.config.GUI_QML_USER_KNOWS_PRESS_AND_HOLD

    @userKnowsPressAndHold.setter
    def userKnowsPressAndHold(self, userKnowsPressAndHold):
        if False:
            print('Hello World!')
        if userKnowsPressAndHold != self.config.GUI_QML_USER_KNOWS_PRESS_AND_HOLD:
            self.config.GUI_QML_USER_KNOWS_PRESS_AND_HOLD = userKnowsPressAndHold
            self.userKnowsPressAndHoldChanged.emit()
    addresslistShowTypeChanged = pyqtSignal()

    @pyqtProperty(int, notify=addresslistShowTypeChanged)
    def addresslistShowType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.GUI_QML_ADDRESS_LIST_SHOW_TYPE

    @addresslistShowType.setter
    def addresslistShowType(self, addresslistShowType):
        if False:
            i = 10
            return i + 15
        if addresslistShowType != self.config.GUI_QML_ADDRESS_LIST_SHOW_TYPE:
            self.config.GUI_QML_ADDRESS_LIST_SHOW_TYPE = addresslistShowType
            self.addresslistShowTypeChanged.emit()
    addresslistShowUsedChanged = pyqtSignal()

    @pyqtProperty(bool, notify=addresslistShowUsedChanged)
    def addresslistShowUsed(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.GUI_QML_ADDRESS_LIST_SHOW_USED

    @addresslistShowUsed.setter
    def addresslistShowUsed(self, addresslistShowUsed):
        if False:
            return 10
        if addresslistShowUsed != self.config.GUI_QML_ADDRESS_LIST_SHOW_USED:
            self.config.GUI_QML_ADDRESS_LIST_SHOW_USED = addresslistShowUsed
            self.addresslistShowUsedChanged.emit()

    @pyqtSlot('qint64', result=str)
    @pyqtSlot(QEAmount, result=str)
    def formatSatsForEditing(self, satoshis):
        if False:
            i = 10
            return i + 15
        if isinstance(satoshis, QEAmount):
            satoshis = satoshis.satsInt
        return self.config.format_amount(satoshis, add_thousands_sep=False)

    @pyqtSlot('qint64', result=str)
    @pyqtSlot('qint64', bool, result=str)
    @pyqtSlot(QEAmount, result=str)
    @pyqtSlot(QEAmount, bool, result=str)
    def formatSats(self, satoshis, with_unit=False):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(satoshis, QEAmount):
            satoshis = satoshis.satsInt
        if with_unit:
            return self.config.format_amount_and_units(satoshis)
        else:
            return self.config.format_amount(satoshis)

    @pyqtSlot(QEAmount, result=str)
    @pyqtSlot(QEAmount, bool, result=str)
    def formatMilliSats(self, amount, with_unit=False):
        if False:
            i = 10
            return i + 15
        if isinstance(amount, QEAmount):
            msats = amount.msatsInt
        else:
            return '---'
        precision = 3
        if with_unit:
            return self.config.format_amount_and_units(msats / 1000, precision=precision)
        else:
            return self.config.format_amount(msats / 1000, precision=precision)

    def decimal_point(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.BTC_AMOUNTS_DECIMAL_POINT

    def max_precision(self):
        if False:
            return 10
        return self.decimal_point() + 0

    @pyqtSlot(str, result=QEAmount)
    def unitsToSats(self, unitAmount):
        if False:
            for i in range(10):
                print('nop')
        self._amount = QEAmount()
        try:
            x = Decimal(unitAmount)
        except Exception:
            return self._amount
        max_prec_amount = int(pow(10, self.max_precision()) * x)
        if self.max_precision() == self.decimal_point():
            self._amount = QEAmount(amount_sat=max_prec_amount)
            return self._amount
        self._logger.debug('fallthrough')
        return self._amount

    @pyqtSlot('quint64', result=float)
    def satsToUnits(self, satoshis):
        if False:
            i = 10
            return i + 15
        return satoshis / pow(10, self.config.decimal_point)