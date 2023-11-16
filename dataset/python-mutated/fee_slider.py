import threading
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QToolTip, QComboBox
from electrum.i18n import _

class FeeComboBox(QComboBox):

    def __init__(self, fee_slider):
        if False:
            while True:
                i = 10
        QComboBox.__init__(self)
        self.config = fee_slider.config
        self.fee_slider = fee_slider
        self.addItems([_('Static'), _('ETA'), _('Mempool')])
        self.setCurrentIndex((2 if self.config.use_mempool_fees() else 1) if self.config.is_dynfee() else 0)
        self.currentIndexChanged.connect(self.on_fee_type)
        self.help_msg = '\n'.join([_('Static: the fee slider uses static values'), _('ETA: fee rate is based on average confirmation time estimates'), _('Mempool based: fee rate is targeting a depth in the memory pool')])

    def on_fee_type(self, x):
        if False:
            while True:
                i = 10
        self.config.FEE_EST_USE_MEMPOOL = x == 2
        self.config.FEE_EST_DYNAMIC = x > 0
        self.fee_slider.update()

class FeeSlider(QSlider):

    def __init__(self, window, config, callback):
        if False:
            while True:
                i = 10
        QSlider.__init__(self, Qt.Horizontal)
        self.config = config
        self.window = window
        self.callback = callback
        self.dyn = False
        self.lock = threading.RLock()
        self.update()
        self.valueChanged.connect(self.moved)
        self._active = True

    def get_fee_rate(self, pos):
        if False:
            return 10
        if self.dyn:
            fee_rate = self.config.depth_to_fee(pos) if self.config.use_mempool_fees() else self.config.eta_to_fee(pos)
        else:
            fee_rate = self.config.static_fee(pos)
        return fee_rate

    def moved(self, pos):
        if False:
            i = 10
            return i + 15
        with self.lock:
            fee_rate = self.get_fee_rate(pos)
            tooltip = self.get_tooltip(pos, fee_rate)
            QToolTip.showText(QCursor.pos(), tooltip, self)
            self.setToolTip(tooltip)
            self.callback(self.dyn, pos, fee_rate)

    def get_tooltip(self, pos, fee_rate):
        if False:
            while True:
                i = 10
        mempool = self.config.use_mempool_fees()
        (target, estimate) = self.config.get_fee_text(pos, self.dyn, mempool, fee_rate)
        if self.dyn:
            return _('Target') + ': ' + target + '\n' + _('Current rate') + ': ' + estimate
        else:
            return _('Fixed rate') + ': ' + target + '\n' + _('Estimate') + ': ' + estimate

    def get_dynfee_target(self):
        if False:
            while True:
                i = 10
        if not self.dyn:
            return ''
        pos = self.value()
        fee_rate = self.get_fee_rate(pos)
        mempool = self.config.use_mempool_fees()
        (target, estimate) = self.config.get_fee_text(pos, True, mempool, fee_rate)
        return target

    def update(self):
        if False:
            print('Hello World!')
        with self.lock:
            self.dyn = self.config.is_dynfee()
            mempool = self.config.use_mempool_fees()
            (maxp, pos, fee_rate) = self.config.get_fee_slider(self.dyn, mempool)
            self.setRange(0, maxp)
            self.setValue(pos)
            tooltip = self.get_tooltip(pos, fee_rate)
            self.setToolTip(tooltip)

    def activate(self):
        if False:
            i = 10
            return i + 15
        self._active = True
        self.setStyleSheet('')

    def deactivate(self):
        if False:
            return 10
        self._active = False
        self.setStyleSheet('\n            QSlider::groove:horizontal {\n                border: 1px solid #999999;\n                height: 8px;\n                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #B1B1B1);\n                margin: 2px 0;\n            }\n\n            QSlider::handle:horizontal {\n                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);\n                border: 1px solid #5c5c5c;\n                width: 12px;\n                margin: -2px 0;\n                border-radius: 3px;\n            }\n            ')

    def is_active(self):
        if False:
            return 10
        return self._active