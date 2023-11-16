import asyncio
import hashlib
from typing import Dict, List, TYPE_CHECKING, Tuple, Set
from collections import defaultdict
import logging
from aiorpcx import run_in_thread, RPCError
from . import util
from .transaction import Transaction, PartialTransaction
from .util import make_aiohttp_session, NetworkJobOnDefaultServer, random_shuffled_copy, OldTaskGroup
from .bitcoin import address_to_scripthash, is_address
from .logging import Logger
from .interface import GracefulDisconnect, NetworkTimeout
if TYPE_CHECKING:
    from .network import Network
    from .address_synchronizer import AddressSynchronizer

class SynchronizerFailure(Exception):
    pass

def history_status(h):
    if False:
        i = 10
        return i + 15
    if not h:
        return None
    status = ''
    for (tx_hash, height) in h:
        status += tx_hash + ':%d:' % height
    return hashlib.sha256(status.encode('ascii')).digest().hex()

class SynchronizerBase(NetworkJobOnDefaultServer):
    """Subscribe over the network to a set of addresses, and monitor their statuses.
    Every time a status changes, run a coroutine provided by the subclass.
    """

    def __init__(self, network: 'Network'):
        if False:
            i = 10
            return i + 15
        self.asyncio_loop = network.asyncio_loop
        NetworkJobOnDefaultServer.__init__(self, network)

    def _reset(self):
        if False:
            i = 10
            return i + 15
        super()._reset()
        self._adding_addrs = set()
        self.requested_addrs = set()
        self._handling_addr_statuses = set()
        self.scripthash_to_address = {}
        self._processed_some_notifications = False
        self.status_queue = asyncio.Queue()

    async def _run_tasks(self, *, taskgroup):
        await super()._run_tasks(taskgroup=taskgroup)
        try:
            async with taskgroup as group:
                await group.spawn(self.handle_status())
                await group.spawn(self.main())
        finally:
            self.session.unsubscribe(self.status_queue)

    def add(self, addr):
        if False:
            while True:
                i = 10
        if not is_address(addr):
            raise ValueError(f'invalid bitcoin address {addr}')
        self._adding_addrs.add(addr)

    async def _add_address(self, addr: str):
        try:
            if not is_address(addr):
                raise ValueError(f'invalid bitcoin address {addr}')
            if addr in self.requested_addrs:
                return
            self.requested_addrs.add(addr)
            await self.taskgroup.spawn(self._subscribe_to_address, addr)
        finally:
            self._adding_addrs.discard(addr)

    async def _on_address_status(self, addr, status):
        """Handle the change of the status of an address.
        Should remove addr from self._handling_addr_statuses when done.
        """
        raise NotImplementedError()

    async def _subscribe_to_address(self, addr):
        h = address_to_scripthash(addr)
        self.scripthash_to_address[h] = addr
        self._requests_sent += 1
        try:
            async with self._network_request_semaphore:
                await self.session.subscribe('blockchain.scripthash.subscribe', [h], self.status_queue)
        except RPCError as e:
            if e.message == 'history too large':
                raise GracefulDisconnect(e, log_level=logging.ERROR) from e
            raise
        self._requests_answered += 1

    async def handle_status(self):
        while True:
            (h, status) = await self.status_queue.get()
            addr = self.scripthash_to_address[h]
            self._handling_addr_statuses.add(addr)
            self.requested_addrs.discard(addr)
            await self.taskgroup.spawn(self._on_address_status, addr, status)
            self._processed_some_notifications = True

    async def main(self):
        raise NotImplementedError()

class Synchronizer(SynchronizerBase):
    """The synchronizer keeps the wallet up-to-date with its set of
    addresses and their transactions.  It subscribes over the network
    to wallet addresses, gets the wallet to generate new addresses
    when necessary, requests the transaction history of any addresses
    we don't have the full history of, and requests binary transaction
    data of any transactions the wallet doesn't have.
    """

    def __init__(self, adb: 'AddressSynchronizer'):
        if False:
            for i in range(10):
                print('nop')
        self.adb = adb
        SynchronizerBase.__init__(self, adb.network)

    def _reset(self):
        if False:
            while True:
                i = 10
        super()._reset()
        self._init_done = False
        self.requested_tx = {}
        self.requested_histories = set()
        self._stale_histories = dict()

    def diagnostic_name(self):
        if False:
            i = 10
            return i + 15
        return self.adb.diagnostic_name()

    def is_up_to_date(self):
        if False:
            i = 10
            return i + 15
        return self._init_done and (not self._adding_addrs) and (not self.requested_addrs) and (not self._handling_addr_statuses) and (not self.requested_histories) and (not self.requested_tx) and (not self._stale_histories) and self.status_queue.empty()

    async def _on_address_status(self, addr, status):
        try:
            history = self.adb.db.get_addr_history(addr)
            if history_status(history) == status:
                return
            if (addr, status) in self.requested_histories:
                return
            self.requested_histories.add((addr, status))
            self._stale_histories.pop(addr, asyncio.Future()).cancel()
        finally:
            self._handling_addr_statuses.discard(addr)
        h = address_to_scripthash(addr)
        self._requests_sent += 1
        async with self._network_request_semaphore:
            result = await self.interface.get_history_for_scripthash(h)
        self._requests_answered += 1
        self.logger.info(f'receiving history {addr} {len(result)}')
        hist = list(map(lambda item: (item['tx_hash'], item['height']), result))
        tx_fees = [(item['tx_hash'], item.get('fee')) for item in result]
        tx_fees = dict(filter(lambda x: x[1] is not None, tx_fees))
        if history_status(hist) != status:
            self.logger.info(f"error: status mismatch: {addr}. we'll wait a bit for status update.")

            async def disconnect_if_still_stale():
                timeout = self.network.get_network_timeout_seconds(NetworkTimeout.Generic)
                await asyncio.sleep(timeout)
                raise SynchronizerFailure(f'timeout reached waiting for addr {addr}: history still stale')
            self._stale_histories[addr] = await self.taskgroup.spawn(disconnect_if_still_stale)
        else:
            self._stale_histories.pop(addr, asyncio.Future()).cancel()
            self.adb.receive_history_callback(addr, hist, tx_fees)
            await self._request_missing_txs(hist)
        self.requested_histories.discard((addr, status))

    async def _request_missing_txs(self, hist, *, allow_server_not_finding_tx=False):
        transaction_hashes = []
        for (tx_hash, tx_height) in hist:
            if tx_hash in self.requested_tx:
                continue
            tx = self.adb.db.get_transaction(tx_hash)
            if tx and (not isinstance(tx, PartialTransaction)):
                continue
            transaction_hashes.append(tx_hash)
            self.requested_tx[tx_hash] = tx_height
        if not transaction_hashes:
            return
        async with OldTaskGroup() as group:
            for tx_hash in transaction_hashes:
                await group.spawn(self._get_transaction(tx_hash, allow_server_not_finding_tx=allow_server_not_finding_tx))

    async def _get_transaction(self, tx_hash, *, allow_server_not_finding_tx=False):
        self._requests_sent += 1
        try:
            async with self._network_request_semaphore:
                raw_tx = await self.interface.get_transaction(tx_hash)
        except RPCError as e:
            if allow_server_not_finding_tx:
                self.requested_tx.pop(tx_hash)
                return
            else:
                raise
        finally:
            self._requests_answered += 1
        tx = Transaction(raw_tx)
        if tx_hash != tx.txid():
            raise SynchronizerFailure(f'received tx does not match expected txid ({tx_hash} != {tx.txid()})')
        tx_height = self.requested_tx.pop(tx_hash)
        self.adb.receive_tx_callback(tx, tx_height)
        self.logger.info(f'received tx {tx_hash} height: {tx_height} bytes: {len(raw_tx)}')

    async def main(self):
        self.adb.up_to_date_changed()
        for addr in random_shuffled_copy(self.adb.db.get_history()):
            history = self.adb.db.get_addr_history(addr)
            if history == ['*']:
                continue
            await self._request_missing_txs(history, allow_server_not_finding_tx=True)
        for addr in random_shuffled_copy(self.adb.get_addresses()):
            await self._add_address(addr)
        self._init_done = True
        prev_uptodate = False
        while True:
            await asyncio.sleep(0.1)
            for addr in self._adding_addrs.copy():
                await self._add_address(addr)
            up_to_date = self.adb.is_up_to_date()
            if up_to_date != prev_uptodate or (up_to_date and self._processed_some_notifications):
                self._processed_some_notifications = False
                self.adb.up_to_date_changed()
            prev_uptodate = up_to_date

class Notifier(SynchronizerBase):
    """Watch addresses. Every time the status of an address changes,
    an HTTP POST is sent to the corresponding URL.
    """

    def __init__(self, network):
        if False:
            print('Hello World!')
        SynchronizerBase.__init__(self, network)
        self.watched_addresses = defaultdict(list)
        self._start_watching_queue = asyncio.Queue()

    async def main(self):
        for addr in self.watched_addresses:
            await self._add_address(addr)
        while True:
            (addr, url) = await self._start_watching_queue.get()
            self.watched_addresses[addr].append(url)
            await self._add_address(addr)

    async def start_watching_addr(self, addr: str, url: str):
        await self._start_watching_queue.put((addr, url))

    async def stop_watching_addr(self, addr: str):
        self.watched_addresses.pop(addr, None)

    async def _on_address_status(self, addr, status):
        if addr not in self.watched_addresses:
            return
        self.logger.info(f'new status for addr {addr}')
        headers = {'content-type': 'application/json'}
        data = {'address': addr, 'status': status}
        for url in self.watched_addresses[addr]:
            try:
                async with make_aiohttp_session(proxy=self.network.proxy, headers=headers) as session:
                    async with session.post(url, json=data, headers=headers) as resp:
                        await resp.text()
            except Exception as e:
                self.logger.info(repr(e))
            else:
                self.logger.info(f'Got Response for {addr}')