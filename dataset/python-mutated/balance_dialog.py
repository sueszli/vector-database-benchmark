from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QVBoxLayout, QCheckBox, QHBoxLayout, QLineEdit, QLabel, QCompleter, QDialog, QStyledItemDelegate, QScrollArea, QWidget, QPushButton, QGridLayout, QToolButton
from PyQt5.QtCore import QRect, QEventLoop, Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QPen, QPainter, QPixmap
from electrum.i18n import _
from .util import Buttons, CloseButton, WindowModalDialog, ColorScheme, font_height, AmountLabel
if TYPE_CHECKING:
    from .main_window import ElectrumWindow
    from electrum.wallet import Abstract_Wallet
COLOR_CONFIRMED = Qt.green
COLOR_UNCONFIRMED = Qt.red
COLOR_UNMATURED = Qt.magenta
COLOR_FROZEN = ColorScheme.BLUE.as_color(True)
COLOR_LIGHTNING = Qt.yellow
COLOR_FROZEN_LIGHTNING = Qt.cyan

class PieChartObject:

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        bgcolor = self.palette().color(QPalette.Background)
        pen = QPen(Qt.gray, 1, Qt.SolidLine)
        qp = QPainter()
        qp.begin(self)
        qp.setPen(pen)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setBrush(Qt.gray)
        total = sum([x[2] for x in self._list])
        if total == 0:
            return
        alpha = 0
        s = 0
        for (name, color, amount) in self._list:
            qp.setBrush(color)
            if amount == 0:
                continue
            elif amount == total:
                qp.drawEllipse(self.R)
            else:
                delta = int(16 * 360 * amount / total)
                qp.drawPie(self.R, alpha, delta)
                alpha += delta
        qp.end()

class PieChartWidget(QWidget, PieChartObject):

    def __init__(self, size, l):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self)
        self.size = size
        self.R = QRect(0, 0, self.size, self.size)
        self.setGeometry(self.R)
        self.setMinimumWidth(self.size)
        self.setMaximumWidth(self.size)
        self.setMinimumHeight(self.size)
        self.setMaximumHeight(self.size)
        self._list = l
        self.update()

    def update_list(self, l):
        if False:
            while True:
                i = 10
        self._list = l
        self.update()

class BalanceToolButton(QToolButton, PieChartObject):

    def __init__(self):
        if False:
            print('Hello World!')
        QToolButton.__init__(self)
        self.size = max(18, font_height())
        self._list = []
        self.R = QRect(6, 3, self.size, self.size)

    def update_list(self, l):
        if False:
            print('Hello World!')
        self._list = l
        self.update()

    def setText(self, text):
        if False:
            while True:
                i = 10
        QToolButton.setText(self, '       ' + text)

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        QToolButton.paintEvent(self, event)
        PieChartObject.paintEvent(self, event)

class LegendWidget(QWidget):
    size = 20

    def __init__(self, color):
        if False:
            while True:
                i = 10
        QWidget.__init__(self)
        self.color = color
        self.R = QRect(0, 0, self.size, int(self.size * 0.75))
        self.setGeometry(self.R)
        self.setMinimumWidth(self.size)
        self.setMaximumWidth(self.size)
        self.setMinimumHeight(self.size)
        self.setMaximumHeight(self.size)

    def paintEvent(self, event):
        if False:
            print('Hello World!')
        bgcolor = self.palette().color(QPalette.Background)
        pen = QPen(Qt.gray, 1, Qt.SolidLine)
        qp = QPainter()
        qp.begin(self)
        qp.setPen(pen)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setBrush(self.color)
        qp.drawRect(self.R)
        qp.end()

class BalanceDialog(WindowModalDialog):

    def __init__(self, parent: 'ElectrumWindow', *, wallet: 'Abstract_Wallet'):
        if False:
            print('Hello World!')
        WindowModalDialog.__init__(self, parent, _('Wallet Balance'))
        self.wallet = wallet
        self.config = parent.config
        self.fx = parent.fx
        (confirmed, unconfirmed, unmatured, frozen, lightning, f_lightning) = self.wallet.get_balances_for_piechart()
        frozen_str = self.config.format_amount_and_units(frozen)
        confirmed_str = self.config.format_amount_and_units(confirmed)
        unconfirmed_str = self.config.format_amount_and_units(unconfirmed)
        unmatured_str = self.config.format_amount_and_units(unmatured)
        lightning_str = self.config.format_amount_and_units(lightning)
        f_lightning_str = self.config.format_amount_and_units(f_lightning)
        frozen_fiat_str = self.fx.format_amount_and_units(frozen) if self.fx else ''
        confirmed_fiat_str = self.fx.format_amount_and_units(confirmed) if self.fx else ''
        unconfirmed_fiat_str = self.fx.format_amount_and_units(unconfirmed) if self.fx else ''
        unmatured_fiat_str = self.fx.format_amount_and_units(unmatured) if self.fx else ''
        lightning_fiat_str = self.fx.format_amount_and_units(lightning) if self.fx else ''
        f_lightning_fiat_str = self.fx.format_amount_and_units(f_lightning) if self.fx else ''
        piechart = PieChartWidget(max(120, 9 * font_height()), [(_('Frozen'), COLOR_FROZEN, frozen), (_('Unmatured'), COLOR_UNMATURED, unmatured), (_('Unconfirmed'), COLOR_UNCONFIRMED, unconfirmed), (_('On-chain'), COLOR_CONFIRMED, confirmed), (_('Lightning'), COLOR_LIGHTNING, lightning), (_('Lightning frozen'), COLOR_FROZEN_LIGHTNING, f_lightning)])
        vbox = QVBoxLayout()
        vbox.addWidget(piechart)
        grid = QGridLayout()
        if frozen:
            grid.addWidget(LegendWidget(COLOR_FROZEN), 0, 0)
            grid.addWidget(QLabel(_('Frozen') + ':'), 0, 1)
            grid.addWidget(AmountLabel(frozen_str), 0, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(frozen_fiat_str), 0, 3, alignment=Qt.AlignRight)
        if unconfirmed:
            grid.addWidget(LegendWidget(COLOR_UNCONFIRMED), 2, 0)
            grid.addWidget(QLabel(_('Unconfirmed') + ':'), 2, 1)
            grid.addWidget(AmountLabel(unconfirmed_str), 2, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(unconfirmed_fiat_str), 2, 3, alignment=Qt.AlignRight)
        if unmatured:
            grid.addWidget(LegendWidget(COLOR_UNMATURED), 3, 0)
            grid.addWidget(QLabel(_('Unmatured') + ':'), 3, 1)
            grid.addWidget(AmountLabel(unmatured_str), 3, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(unmatured_fiat_str), 3, 3, alignment=Qt.AlignRight)
        if confirmed:
            grid.addWidget(LegendWidget(COLOR_CONFIRMED), 1, 0)
            grid.addWidget(QLabel(_('On-chain') + ':'), 1, 1)
            grid.addWidget(AmountLabel(confirmed_str), 1, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(confirmed_fiat_str), 1, 3, alignment=Qt.AlignRight)
        if lightning:
            grid.addWidget(LegendWidget(COLOR_LIGHTNING), 4, 0)
            grid.addWidget(QLabel(_('Lightning') + ':'), 4, 1)
            grid.addWidget(AmountLabel(lightning_str), 4, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(lightning_fiat_str), 4, 3, alignment=Qt.AlignRight)
        if f_lightning:
            grid.addWidget(LegendWidget(COLOR_FROZEN_LIGHTNING), 5, 0)
            grid.addWidget(QLabel(_('Lightning (frozen)') + ':'), 5, 1)
            grid.addWidget(AmountLabel(f_lightning_str), 5, 2, alignment=Qt.AlignRight)
            grid.addWidget(AmountLabel(f_lightning_fiat_str), 5, 3, alignment=Qt.AlignRight)
        vbox.addLayout(grid)
        vbox.addStretch(1)
        btn_close = CloseButton(self)
        btns = Buttons(btn_close)
        vbox.addLayout(btns)
        self.setLayout(vbox)

    def run(self):
        if False:
            i = 10
            return i + 15
        self.exec_()