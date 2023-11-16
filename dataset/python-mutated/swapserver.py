import asyncio
from typing import TYPE_CHECKING
from electrum.plugin import BasePlugin, hook
from .server import SwapServer
if TYPE_CHECKING:
    from electrum.simple_config import SimpleConfig
    from electrum.daemon import Daemon
    from electrum.wallet import Abstract_Wallet

class SwapServerPlugin(BasePlugin):

    def __init__(self, parent, config: 'SimpleConfig', name):
        if False:
            while True:
                i = 10
        BasePlugin.__init__(self, parent, config, name)
        self.config = config
        self.server = None

    @hook
    def daemon_wallet_loaded(self, daemon: 'Daemon', wallet: 'Abstract_Wallet'):
        if False:
            while True:
                i = 10
        if self.server is not None:
            return
        if self.config.NETWORK_OFFLINE:
            return
        self.server = SwapServer(self.config, wallet)
        sm = wallet.lnworker.swap_manager
        for coro in [self.server.run()]:
            asyncio.run_coroutine_threadsafe(daemon.taskgroup.spawn(coro), daemon.asyncio_loop)