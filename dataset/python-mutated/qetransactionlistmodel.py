from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Any
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot
from PyQt6.QtCore import Qt, QAbstractListModel, QModelIndex
from electrum.logging import get_logger
from electrum.util import Satoshis, TxMinedInfo
from electrum.address_synchronizer import TX_HEIGHT_FUTURE, TX_HEIGHT_LOCAL
from .qetypes import QEAmount
from .util import QtEventListener, qt_event_listener
if TYPE_CHECKING:
    from electrum.wallet import Abstract_Wallet

class QETransactionListModel(QAbstractListModel, QtEventListener):
    _logger = get_logger(__name__)
    _ROLE_NAMES = ('txid', 'fee_sat', 'height', 'confirmations', 'timestamp', 'monotonic_timestamp', 'incoming', 'value', 'date', 'label', 'txpos_in_block', 'fee', 'inputs', 'outputs', 'section', 'type', 'lightning', 'payment_hash', 'key', 'complete')
    _ROLE_KEYS = range(Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + len(_ROLE_NAMES))
    _ROLE_MAP = dict(zip(_ROLE_KEYS, [bytearray(x.encode()) for x in _ROLE_NAMES]))
    _ROLE_RMAP = dict(zip(_ROLE_NAMES, _ROLE_KEYS))
    requestRefresh = pyqtSignal()

    def __init__(self, wallet: 'Abstract_Wallet', parent=None, *, onchain_domain=None, include_lightning=True):
        if False:
            return 10
        super().__init__(parent)
        self.wallet = wallet
        self.onchain_domain = onchain_domain
        self.include_lightning = include_lightning
        self.tx_history = []
        self.register_callbacks()
        self.destroyed.connect(lambda : self.on_destroy())
        self.requestRefresh.connect(lambda : self.initModel())
        self._dirty = True
        self.initModel()

    def on_destroy(self):
        if False:
            while True:
                i = 10
        self.unregister_callbacks()

    @qt_event_listener
    def on_event_verified(self, wallet, txid, info):
        if False:
            i = 10
            return i + 15
        if wallet == self.wallet:
            self._logger.debug('verified event for txid %s' % txid)
            self.on_tx_verified(txid, info)

    @qt_event_listener
    def on_event_adb_set_future_tx(self, adb, txid):
        if False:
            for i in range(10):
                print('nop')
        if adb != self.wallet.adb:
            return
        self._logger.debug(f'adb_set_future_tx event for txid {txid}')
        for (i, item) in enumerate(self.tx_history):
            if 'txid' in item and item['txid'] == txid:
                self._update_future_txitem(i)
                return

    @qt_event_listener
    def on_event_fee_histogram(self, histogram):
        if False:
            return 10
        self._logger.debug(f'fee histogram updated')
        for (i, tx_item) in enumerate(self.tx_history):
            if 'height' not in tx_item:
                continue
            if tx_item['confirmations'] > 0:
                continue
            txid = tx_item['txid']
            tx = self.wallet.db.get_transaction(txid)
            if not tx:
                continue
            txinfo = self.wallet.get_tx_info(tx)
            (status, status_str) = self.wallet.get_tx_status(txid, txinfo.tx_mined_status)
            tx_item['date'] = status_str
            index = self.index(i, 0)
            roles = [self._ROLE_RMAP['date']]
            self.dataChanged.emit(index, index, roles)

    def rowCount(self, index):
        if False:
            for i in range(10):
                print('nop')
        return len(self.tx_history)
    countChanged = pyqtSignal()

    @pyqtProperty(int, notify=countChanged)
    def count(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.tx_history)

    def roleNames(self):
        if False:
            return 10
        return self._ROLE_MAP

    def data(self, index, role):
        if False:
            return 10
        tx = self.tx_history[index.row()]
        role_index = role - Qt.ItemDataRole.UserRole
        try:
            value = tx[self._ROLE_NAMES[role_index]]
        except KeyError as e:
            self._logger.error(f'non-existing key "{self._ROLE_NAMES[role_index]}" requested')
            value = None
        if isinstance(value, (bool, list, int, str, QEAmount)) or value is None:
            return value
        if isinstance(value, Satoshis):
            return value.value
        return str(value)

    @pyqtSlot()
    def setDirty(self):
        if False:
            for i in range(10):
                print('nop')
        self._dirty = True

    def clear(self):
        if False:
            return 10
        self.beginResetModel()
        self.tx_history = []
        self.endResetModel()

    def tx_to_model(self, tx_item):
        if False:
            print('Hello World!')
        item = tx_item
        item['key'] = item['txid'] if 'txid' in item else item['payment_hash']
        if 'lightning' not in item:
            item['lightning'] = False
        if item['lightning']:
            item['value'] = QEAmount(amount_sat=item['value'].value, amount_msat=item['amount_msat'])
            if item['type'] == 'payment':
                item['incoming'] = True if item['direction'] == 'received' else False
            item['confirmations'] = 0
        else:
            item['value'] = QEAmount(amount_sat=item['value'].value)
        if 'txid' in item:
            tx = self.wallet.db.get_transaction(item['txid'])
            if tx:
                item['complete'] = tx.is_complete()
            else:
                item['complete'] = False
        if not item['timestamp']:
            txid = item['txid']
            assert txid
            tx_mined_info = self._tx_mined_info_from_tx_item(tx_item)
            item['section'] = 'local' if tx_mined_info.is_local_like() else 'mempool'
            (status, status_str) = self.wallet.get_tx_status(txid, tx_mined_info=tx_mined_info)
            item['date'] = status_str
        else:
            item['section'] = self.get_section_by_timestamp(item['timestamp'])
            item['date'] = self.format_date_by_section(item['section'], datetime.fromtimestamp(item['timestamp']))
        return item

    def get_section_by_timestamp(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        txts = datetime.fromtimestamp(timestamp)
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        if txts > today:
            return 'today'
        elif txts > today - timedelta(days=1):
            return 'yesterday'
        elif txts > today - timedelta(days=7):
            return 'lastweek'
        elif txts > today - timedelta(days=31):
            return 'lastmonth'
        else:
            return 'older'

    def format_date_by_section(self, section, date):
        if False:
            return 10
        dfmt = {'today': '%H:%M:%S', 'yesterday': '%H:%M:%S', 'lastweek': '%a, %H:%M:%S', 'lastmonth': '%a %d, %H:%M:%S', 'older': '%G-%m-%d %H:%M:%S'}
        if section not in dfmt:
            section = 'older'
        return date.strftime(dfmt[section])

    @staticmethod
    def _tx_mined_info_from_tx_item(tx_item: Dict[str, Any]) -> TxMinedInfo:
        if False:
            return 10
        tx_mined_info = TxMinedInfo(height=tx_item['height'], conf=tx_item['confirmations'], timestamp=tx_item['timestamp'], wanted_height=tx_item.get('wanted_height', None))
        return tx_mined_info

    @pyqtSlot()
    @pyqtSlot(bool)
    def initModel(self, force: bool=False):
        if False:
            print('Hello World!')
        if not self._dirty and (not force):
            return
        self._logger.debug('retrieving history')
        history = self.wallet.get_full_history(onchain_domain=self.onchain_domain, include_lightning=self.include_lightning, include_fiat=False)
        txs = []
        for (key, tx) in history.items():
            txs.append(self.tx_to_model(tx))
        self.clear()
        self.beginInsertRows(QModelIndex(), 0, len(txs) - 1)
        self.tx_history = txs
        self.tx_history.reverse()
        self.endInsertRows()
        self.countChanged.emit()
        self._dirty = False

    def on_tx_verified(self, txid, info):
        if False:
            print('Hello World!')
        for (i, tx) in enumerate(self.tx_history):
            if 'txid' in tx and tx['txid'] == txid:
                tx['height'] = info.height
                tx['confirmations'] = info.conf
                tx['timestamp'] = info.timestamp
                tx['section'] = self.get_section_by_timestamp(info.timestamp)
                tx['date'] = self.format_date_by_section(tx['section'], datetime.fromtimestamp(info.timestamp))
                index = self.index(i, 0)
                roles = [self._ROLE_RMAP[x] for x in ['section', 'height', 'confirmations', 'timestamp', 'date']]
                self.dataChanged.emit(index, index, roles)
                return

    def _update_future_txitem(self, tx_item_idx: int):
        if False:
            while True:
                i = 10
        tx_item = self.tx_history[tx_item_idx]
        if tx_item.get('height') not in (TX_HEIGHT_FUTURE, TX_HEIGHT_LOCAL):
            return
        txid = tx_item['txid']
        tx = self.wallet.db.get_transaction(txid)
        if tx is None:
            return
        txinfo = self.wallet.get_tx_info(tx)
        (status, status_str) = self.wallet.get_tx_status(txid, txinfo.tx_mined_status)
        tx_item['date'] = status_str
        tx_item['height'] = self.wallet.adb.get_tx_height(txid).height
        index = self.index(tx_item_idx, 0)
        roles = [self._ROLE_RMAP[x] for x in ['height', 'date']]
        self.dataChanged.emit(index, index, roles)

    @pyqtSlot(str, str)
    def updateTxLabel(self, key, label):
        if False:
            print('Hello World!')
        for (i, tx) in enumerate(self.tx_history):
            if tx['key'] == key:
                tx['label'] = label
                index = self.index(i, 0)
                self.dataChanged.emit(index, index, [self._ROLE_RMAP['label']])
                return

    @pyqtSlot(int)
    def updateBlockchainHeight(self, height):
        if False:
            while True:
                i = 10
        self._logger.debug('updating height to %d' % height)
        for (i, tx_item) in enumerate(self.tx_history):
            if 'height' in tx_item:
                if tx_item['height'] > 0:
                    tx_item['confirmations'] = height - tx_item['height'] + 1
                    index = self.index(i, 0)
                    roles = [self._ROLE_RMAP['confirmations']]
                    self.dataChanged.emit(index, index, roles)
                elif tx_item['height'] in (TX_HEIGHT_FUTURE, TX_HEIGHT_LOCAL):
                    self._update_future_txitem(i)