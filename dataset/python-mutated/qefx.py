from datetime import datetime
from decimal import Decimal
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QObject, QRegularExpression
from electrum.bitcoin import COIN
from electrum.exchange_rate import FxThread
from electrum.logging import get_logger
from electrum.simple_config import SimpleConfig
from .qetypes import QEAmount
from .util import QtEventListener, event_listener

class QEFX(QObject, QtEventListener):
    _logger = get_logger(__name__)
    quotesUpdated = pyqtSignal()

    def __init__(self, fxthread: FxThread, config: SimpleConfig, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.fx = fxthread
        self.config = config
        self.register_callbacks()
        self.destroyed.connect(lambda : self.on_destroy())

    def on_destroy(self):
        if False:
            print('Hello World!')
        self.unregister_callbacks()

    @event_listener
    def on_event_on_quotes(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._logger.debug('new quotes')
        self.quotesUpdated.emit()
    historyUpdated = pyqtSignal()

    @event_listener
    def on_event_on_history(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._logger.debug('new history')
        self.historyUpdated.emit()
    currenciesChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=currenciesChanged)
    def currencies(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fx.get_currencies(self.historicRates)
    rateSourcesChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=rateSourcesChanged)
    def rateSources(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fx.get_exchanges_by_ccy(self.fiatCurrency, self.historicRates)
    fiatCurrencyChanged = pyqtSignal()

    @pyqtProperty(str, notify=fiatCurrencyChanged)
    def fiatCurrency(self):
        if False:
            print('Hello World!')
        return self.fx.get_currency()

    @fiatCurrency.setter
    def fiatCurrency(self, currency):
        if False:
            return 10
        if currency != self.fiatCurrency:
            self.fx.set_currency(currency)
            self.enabled = self.enabled and currency != ''
            self.fiatCurrencyChanged.emit()
            self.rateSourcesChanged.emit()

    @pyqtProperty('QRegularExpression', notify=fiatCurrencyChanged)
    def fiatAmountRegex(self):
        if False:
            while True:
                i = 10
        decimals = self.fx.ccy_precision()
        exp = '[0-9]*'
        if decimals:
            exp += '\\.'
            exp += '[0-9]{0,%d}' % decimals
        return QRegularExpression(exp)
    historicRatesChanged = pyqtSignal()

    @pyqtProperty(bool, notify=historicRatesChanged)
    def historicRates(self):
        if False:
            return 10
        if not self.fx.config.cv.FX_HISTORY_RATES.is_set():
            self.fx.config.FX_HISTORY_RATES = True
        return self.fx.config.FX_HISTORY_RATES

    @historicRates.setter
    def historicRates(self, checked):
        if False:
            i = 10
            return i + 15
        if checked != self.historicRates:
            self.fx.config.FX_HISTORY_RATES = bool(checked)
            self.historicRatesChanged.emit()
            self.rateSourcesChanged.emit()
    rateSourceChanged = pyqtSignal()

    @pyqtProperty(str, notify=rateSourceChanged)
    def rateSource(self):
        if False:
            print('Hello World!')
        return self.fx.config_exchange()

    @rateSource.setter
    def rateSource(self, source):
        if False:
            while True:
                i = 10
        if source != self.rateSource:
            self.fx.set_exchange(source)
            self.rateSourceChanged.emit()
    enabledUpdated = pyqtSignal()

    @pyqtProperty(bool, notify=enabledUpdated)
    def enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fx.is_enabled()

    @enabled.setter
    def enabled(self, enable):
        if False:
            print('Hello World!')
        if enable != self.enabled:
            self.fx.set_enabled(enable)
            self.enabledUpdated.emit()

    @pyqtSlot(str, result=str)
    @pyqtSlot(str, bool, result=str)
    @pyqtSlot(QEAmount, result=str)
    @pyqtSlot(QEAmount, bool, result=str)
    def fiatValue(self, satoshis, plain=True):
        if False:
            while True:
                i = 10
        rate = self.fx.exchange_rate()
        if isinstance(satoshis, QEAmount):
            satoshis = satoshis.msatsInt / 1000 if satoshis.msatsInt != 0 else satoshis.satsInt
        else:
            try:
                sd = Decimal(satoshis)
            except Exception:
                return ''
        if plain:
            return self.fx.ccy_amount_str(self.fx.fiat_value(satoshis, rate), add_thousands_sep=False)
        else:
            return self.fx.value_str(satoshis, rate)

    @pyqtSlot(str, str, result=str)
    @pyqtSlot(str, str, bool, result=str)
    @pyqtSlot(QEAmount, str, result=str)
    @pyqtSlot(QEAmount, str, bool, result=str)
    def fiatValueHistoric(self, satoshis, timestamp, plain=True):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(satoshis, QEAmount):
            satoshis = satoshis.msatsInt / 1000 if satoshis.msatsInt != 0 else satoshis.satsInt
        else:
            try:
                sd = Decimal(satoshis)
            except Exception:
                return ''
        try:
            td = Decimal(timestamp)
            if td == 0:
                return ''
        except Exception:
            return ''
        dt = datetime.fromtimestamp(int(td))
        if plain:
            return self.fx.ccy_amount_str(self.fx.historical_value(satoshis, dt), add_thousands_sep=False)
        else:
            return self.fx.historical_value_str(satoshis, dt)

    @pyqtSlot(str, result=str)
    @pyqtSlot(str, bool, result=str)
    def satoshiValue(self, fiat, plain=True):
        if False:
            return 10
        rate = self.fx.exchange_rate()
        try:
            fd = Decimal(fiat)
        except Exception:
            return ''
        v = fd / Decimal(rate) * COIN
        if v.is_nan():
            return ''
        if plain:
            return str(v.to_integral_value())
        else:
            return self.config.format_amount(v)