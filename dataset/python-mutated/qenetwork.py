from typing import TYPE_CHECKING
from PyQt6.QtCore import pyqtProperty, pyqtSignal, QObject
from electrum.logging import get_logger
from electrum import constants
from electrum.interface import ServerAddr
from electrum.simple_config import FEERATE_DEFAULT_RELAY
from .util import QtEventListener, event_listener
from .qeserverlistmodel import QEServerListModel
if TYPE_CHECKING:
    from .qeconfig import QEConfig
    from electrum.network import Network

class QENetwork(QObject, QtEventListener):
    _logger = get_logger(__name__)
    networkUpdated = pyqtSignal()
    blockchainUpdated = pyqtSignal()
    heightChanged = pyqtSignal([int], arguments=['height'])
    serverHeightChanged = pyqtSignal([int], arguments=['height'])
    proxySet = pyqtSignal()
    proxyChanged = pyqtSignal()
    statusChanged = pyqtSignal()
    feeHistogramUpdated = pyqtSignal()
    chaintipsChanged = pyqtSignal()
    isLaggingChanged = pyqtSignal()
    gossipUpdated = pyqtSignal()
    dataChanged = pyqtSignal()
    _height = 0
    _server = ''
    _is_connected = False
    _server_status = ''
    _network_status = ''
    _chaintips = 1
    _islagging = False
    _fee_histogram = []
    _gossipPeers = 0
    _gossipUnknownChannels = 0
    _gossipDbNodes = 0
    _gossipDbChannels = 0
    _gossipDbPolicies = 0

    def __init__(self, network: 'Network', qeconfig: 'QEConfig', parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        assert network, '--offline is not yet implemented for this GUI'
        self.network = network
        self._qeconfig = qeconfig
        self._serverListModel = None
        self._height = network.get_local_height()
        self._server_height = network.get_server_height()
        self.register_callbacks()
        self.destroyed.connect(lambda : self.on_destroy())
        self._qeconfig.useGossipChanged.connect(self.on_gossip_setting_changed)

    def on_destroy(self):
        if False:
            i = 10
            return i + 15
        self.unregister_callbacks()

    @event_listener
    def on_event_network_updated(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.networkUpdated.emit()
        self._update_status()

    @event_listener
    def on_event_blockchain_updated(self):
        if False:
            return 10
        if self._height != self.network.get_local_height():
            self._height = self.network.get_local_height()
            self._logger.debug('new height: %d' % self._height)
            self.heightChanged.emit(self._height)
        self.blockchainUpdated.emit()

    @event_listener
    def on_event_default_server_changed(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._update_status()

    @event_listener
    def on_event_proxy_set(self, *args):
        if False:
            return 10
        self._logger.debug('proxy set')
        self.proxySet.emit()
        self.proxyTorChanged.emit()

    def _update_status(self):
        if False:
            while True:
                i = 10
        server = str(self.network.get_parameters().server)
        if self._server != server:
            self._server = server
            self.statusChanged.emit()
        network_status = self.network.get_status()
        if self._network_status != network_status:
            self._logger.debug('network_status updated: %s' % network_status)
            self._network_status = network_status
            self.statusChanged.emit()
        is_connected = self.network.is_connected()
        if self._is_connected != is_connected:
            self._is_connected = is_connected
            self.statusChanged.emit()
        server_status = self.network.get_connection_status_for_GUI()
        if self._server_status != server_status:
            self._logger.debug('server_status updated: %s' % server_status)
            self._server_status = server_status
            self.statusChanged.emit()
        server_height = self.network.get_server_height()
        if self._server_height != server_height:
            self._logger.debug(f'server_height updated: {server_height}')
            self._server_height = server_height
            self.serverHeightChanged.emit(server_height)
        chains = len(self.network.get_blockchains())
        if chains != self._chaintips:
            self._logger.debug('chain tips # changed: %d', chains)
            self._chaintips = chains
            self.chaintipsChanged.emit()
        server_lag = self.network.get_local_height() - self.network.get_server_height()
        if self._islagging ^ (server_lag > 1):
            self._logger.debug('lagging changed: %s', str(server_lag > 1))
            self._islagging = server_lag > 1
            self.isLaggingChanged.emit()

    @event_listener
    def on_event_status(self, *args):
        if False:
            i = 10
            return i + 15
        self._update_status()

    @event_listener
    def on_event_fee_histogram(self, histogram):
        if False:
            print('Hello World!')
        self._logger.debug(f'fee histogram updated')
        self.update_histogram(histogram)

    def update_histogram(self, histogram):
        if False:
            i = 10
            return i + 15
        if not histogram:
            histogram = [[FEERATE_DEFAULT_RELAY / 1000, 1]]
        bytes_limit = 10 * 1000 * 1000
        bytes_current = 0
        capped_histogram = []
        for item in sorted(histogram, key=lambda x: x[0], reverse=True):
            if bytes_current >= bytes_limit:
                break
            slot = min(item[1], bytes_limit - bytes_current)
            bytes_current += slot
            capped_histogram.append([max(FEERATE_DEFAULT_RELAY / 1000, item[0]), slot, bytes_current])
        self._fee_histogram = {'histogram': capped_histogram, 'total': bytes_current, 'min_fee': capped_histogram[-1][0] if capped_histogram else FEERATE_DEFAULT_RELAY / 1000, 'max_fee': capped_histogram[0][0] if capped_histogram else FEERATE_DEFAULT_RELAY / 1000}
        self.feeHistogramUpdated.emit()

    @event_listener
    def on_event_channel_db(self, num_nodes, num_channels, num_policies):
        if False:
            return 10
        self._logger.debug(f'channel_db: {num_nodes} nodes, {num_channels} channels, {num_policies} policies')
        self._gossipDbNodes = num_nodes
        self._gossipDbChannels = num_channels
        self._gossipDbPolicies = num_policies
        self.gossipUpdated.emit()

    @event_listener
    def on_event_gossip_peers(self, num_peers):
        if False:
            return 10
        self._logger.debug(f'gossip peers {num_peers}')
        self._gossipPeers = num_peers
        self.gossipUpdated.emit()

    @event_listener
    def on_event_unknown_channels(self, unknown):
        if False:
            print('Hello World!')
        if unknown == 0 and self._gossipUnknownChannels == 0:
            return
        self._logger.debug(f'unknown channels {unknown}')
        self._gossipUnknownChannels = unknown
        self.gossipUpdated.emit()

    def on_gossip_setting_changed(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.network:
            return
        if self._qeconfig.useGossip:
            self.network.start_gossip()
        else:
            self.network.run_from_another_thread(self.network.stop_gossip())

    @pyqtProperty(int, notify=heightChanged)
    def height(self):
        if False:
            print('Hello World!')
        return self._height

    @pyqtProperty(int, notify=serverHeightChanged)
    def serverHeight(self):
        if False:
            return 10
        return self._server_height

    @pyqtProperty(str, notify=statusChanged)
    def server(self):
        if False:
            for i in range(10):
                print('nop')
        return self._server

    @server.setter
    def server(self, server: str):
        if False:
            while True:
                i = 10
        net_params = self.network.get_parameters()
        try:
            server = ServerAddr.from_str_with_inference(server)
            if not server:
                raise Exception('failed to parse')
        except Exception:
            return
        net_params = net_params._replace(server=server, auto_connect=self._qeconfig.autoConnect)
        self.network.run_from_another_thread(self.network.set_parameters(net_params))

    @pyqtProperty(str, notify=statusChanged)
    def serverWithStatus(self):
        if False:
            while True:
                i = 10
        server = self._server
        if not self.network.is_connected():
            return f'{server} (connecting...)'
        return server

    @pyqtProperty(str, notify=statusChanged)
    def status(self):
        if False:
            return 10
        return self._network_status

    @pyqtProperty(str, notify=statusChanged)
    def serverStatus(self):
        if False:
            while True:
                i = 10
        return self.network.get_connection_status_for_GUI()

    @pyqtProperty(bool, notify=statusChanged)
    def isConnected(self):
        if False:
            i = 10
            return i + 15
        return self._is_connected

    @pyqtProperty(int, notify=chaintipsChanged)
    def chaintips(self):
        if False:
            while True:
                i = 10
        return self._chaintips

    @pyqtProperty(bool, notify=isLaggingChanged)
    def isLagging(self):
        if False:
            while True:
                i = 10
        return self._islagging

    @pyqtProperty(bool, notify=dataChanged)
    def isTestNet(self):
        if False:
            for i in range(10):
                print('nop')
        return constants.net.TESTNET

    @pyqtProperty(str, notify=dataChanged)
    def networkName(self):
        if False:
            i = 10
            return i + 15
        return constants.net.__name__.replace('Bitcoin', '')

    @pyqtProperty('QVariantMap', notify=proxyChanged)
    def proxy(self):
        if False:
            for i in range(10):
                print('nop')
        net_params = self.network.get_parameters()
        return net_params.proxy if net_params.proxy else {}

    @proxy.setter
    def proxy(self, proxy_settings):
        if False:
            return 10
        net_params = self.network.get_parameters()
        if not proxy_settings['enabled']:
            proxy_settings = None
        net_params = net_params._replace(proxy=proxy_settings)
        self.network.run_from_another_thread(self.network.set_parameters(net_params))
        self.proxyChanged.emit()
    proxyTorChanged = pyqtSignal()

    @pyqtProperty(bool, notify=proxyTorChanged)
    def isProxyTor(self):
        if False:
            print('Hello World!')
        return self.network.tor_proxy

    @pyqtProperty('QVariant', notify=feeHistogramUpdated)
    def feeHistogram(self):
        if False:
            while True:
                i = 10
        return self._fee_histogram

    @pyqtProperty('QVariantMap', notify=gossipUpdated)
    def gossipInfo(self):
        if False:
            return 10
        return {'peers': self._gossipPeers, 'unknown_channels': self._gossipUnknownChannels, 'db_nodes': self._gossipDbNodes, 'db_channels': self._gossipDbChannels, 'db_policies': self._gossipDbPolicies}
    serverListModelChanged = pyqtSignal()

    @pyqtProperty(QEServerListModel, notify=serverListModelChanged)
    def serverListModel(self):
        if False:
            for i in range(10):
                print('nop')
        if self._serverListModel is None:
            self._serverListModel = QEServerListModel(self.network)
        return self._serverListModel