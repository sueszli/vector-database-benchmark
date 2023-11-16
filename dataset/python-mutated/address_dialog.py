from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from electrum.i18n import _
from .util import WindowModalDialog, ButtonsLineEdit, ShowQRLineEdit, ColorScheme, Buttons, CloseButton
from .history_list import HistoryList, HistoryModel
from .qrtextedit import ShowQRTextEdit
if TYPE_CHECKING:
    from .main_window import ElectrumWindow

class AddressHistoryModel(HistoryModel):

    def __init__(self, window: 'ElectrumWindow', address):
        if False:
            i = 10
            return i + 15
        super().__init__(window)
        self.address = address

    def get_domain(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.address]

    def should_include_lightning_payments(self) -> bool:
        if False:
            print('Hello World!')
        return False

class AddressDialog(WindowModalDialog):

    def __init__(self, window: 'ElectrumWindow', address: str, *, parent=None):
        if False:
            while True:
                i = 10
        if parent is None:
            parent = window
        WindowModalDialog.__init__(self, parent, _('Address'))
        self.address = address
        self.window = window
        self.config = window.config
        self.wallet = window.wallet
        self.app = window.app
        self.saved = True
        self.setMinimumWidth(700)
        vbox = QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(QLabel(_('Address') + ':'))
        self.addr_e = ShowQRLineEdit(self.address, self.config, title=_('Address'))
        vbox.addWidget(self.addr_e)
        try:
            pubkeys = self.wallet.get_public_keys(address)
        except BaseException as e:
            pubkeys = None
        if pubkeys:
            vbox.addWidget(QLabel(_('Public keys') + ':'))
            for pubkey in pubkeys:
                pubkey_e = ShowQRLineEdit(pubkey, self.config, title=_('Public Key'))
                vbox.addWidget(pubkey_e)
        redeem_script = self.wallet.get_redeem_script(address)
        if redeem_script:
            vbox.addWidget(QLabel(_('Redeem Script') + ':'))
            redeem_e = ShowQRTextEdit(text=redeem_script, config=self.config)
            redeem_e.addCopyButton()
            vbox.addWidget(redeem_e)
        witness_script = self.wallet.get_witness_script(address)
        if witness_script:
            vbox.addWidget(QLabel(_('Witness Script') + ':'))
            witness_e = ShowQRTextEdit(text=witness_script, config=self.config)
            witness_e.addCopyButton()
            vbox.addWidget(witness_e)
        address_path_str = self.wallet.get_address_path_str(address)
        if address_path_str:
            vbox.addWidget(QLabel(_('Derivation path') + ':'))
            der_path_e = ButtonsLineEdit(address_path_str)
            der_path_e.addCopyButton()
            der_path_e.setReadOnly(True)
            vbox.addWidget(der_path_e)
        addr_hist_model = AddressHistoryModel(self.window, self.address)
        self.hw = HistoryList(self.window, addr_hist_model)
        self.hw.num_tx_label = QLabel('')
        addr_hist_model.set_view(self.hw)
        vbox.addWidget(self.hw.num_tx_label)
        vbox.addWidget(self.hw)
        vbox.addLayout(Buttons(CloseButton(self)))
        self.format_amount = self.window.format_amount
        addr_hist_model.refresh('address dialog constructor')

    def show_qr(self):
        if False:
            i = 10
            return i + 15
        text = self.address
        try:
            self.window.show_qrcode(text, 'Address', parent=self)
        except Exception as e:
            self.show_message(repr(e))