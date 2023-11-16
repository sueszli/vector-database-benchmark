import logging
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union, cast
import attr
from typing_extensions import TypedDict
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.storage._base import SQLBaseStore
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction, make_tuple_comparison_clause
from synapse.storage.databases.main.monthly_active_users import MonthlyActiveUsersWorkerStore
from synapse.types import JsonDict, UserID
from synapse.util.caches.lrucache import LruCache
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
LAST_SEEN_GRANULARITY = 120 * 1000

@attr.s(slots=True, frozen=True, auto_attribs=True)
class DeviceLastConnectionInfo:
    """Metadata for the last connection seen for a user and device combination"""
    user_id: str
    device_id: str
    ip: Optional[str]
    user_agent: Optional[str]
    last_seen: Optional[int]

class LastConnectionInfo(TypedDict):
    """Metadata for the last connection seen for an access token and IP combination"""
    access_token: str
    ip: str
    user_agent: str
    last_seen: int

class ClientIpBackgroundUpdateStore(SQLBaseStore):

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(database, db_conn, hs)
        self.db_pool.updates.register_background_index_update('user_ips_device_index', index_name='user_ips_device_id', table='user_ips', columns=['user_id', 'device_id', 'last_seen'])
        self.db_pool.updates.register_background_index_update('user_ips_last_seen_index', index_name='user_ips_last_seen', table='user_ips', columns=['user_id', 'last_seen'])
        self.db_pool.updates.register_background_index_update('user_ips_last_seen_only_index', index_name='user_ips_last_seen_only', table='user_ips', columns=['last_seen'])
        self.db_pool.updates.register_background_update_handler('user_ips_analyze', self._analyze_user_ip)
        self.db_pool.updates.register_background_update_handler('user_ips_remove_dupes', self._remove_user_ip_dupes)
        self.db_pool.updates.register_background_index_update('user_ips_device_unique_index', index_name='user_ips_user_token_ip_unique_index', table='user_ips', columns=['user_id', 'access_token', 'ip'], unique=True)
        self.db_pool.updates.register_background_update_handler('user_ips_drop_nonunique_index', self._remove_user_ip_nonunique)
        self.db_pool.updates.register_background_update_handler('devices_last_seen', self._devices_last_seen_update)

    async def _remove_user_ip_nonunique(self, progress: JsonDict, batch_size: int) -> int:

        def f(conn: LoggingDatabaseConnection) -> None:
            if False:
                return 10
            txn = conn.cursor()
            txn.execute('DROP INDEX IF EXISTS user_ips_user_ip')
            txn.close()
        await self.db_pool.runWithConnection(f)
        await self.db_pool.updates._end_background_update('user_ips_drop_nonunique_index')
        return 1

    async def _analyze_user_ip(self, progress: JsonDict, batch_size: int) -> int:

        def user_ips_analyze(txn: LoggingTransaction) -> None:
            if False:
                for i in range(10):
                    print('nop')
            txn.execute('ANALYZE user_ips')
        await self.db_pool.runInteraction('user_ips_analyze', user_ips_analyze)
        await self.db_pool.updates._end_background_update('user_ips_analyze')
        return 1

    async def _remove_user_ip_dupes(self, progress: JsonDict, batch_size: int) -> int:
        begin_last_seen: int = progress.get('last_seen', 0)

        def get_last_seen(txn: LoggingTransaction) -> Optional[int]:
            if False:
                print('Hello World!')
            txn.execute('\n                SELECT last_seen FROM user_ips\n                WHERE last_seen > ?\n                ORDER BY last_seen\n                LIMIT 1\n                OFFSET ?\n                ', (begin_last_seen, batch_size))
            row = cast(Optional[Tuple[int]], txn.fetchone())
            if row:
                return row[0]
            else:
                return None
        end_last_seen = await self.db_pool.runInteraction('user_ips_dups_get_last_seen', get_last_seen)
        last = end_last_seen is None
        logger.info("Scanning for duplicate 'user_ips' rows in range: %s <= last_seen < %s", begin_last_seen, end_last_seen)

        def remove(txn: LoggingTransaction) -> None:
            if False:
                for i in range(10):
                    print('nop')
            args: Tuple[int, ...]
            if last:
                clause = '? <= last_seen'
                args = (begin_last_seen,)
            else:
                assert end_last_seen is not None
                clause = '? <= last_seen AND last_seen < ?'
                args = (begin_last_seen, end_last_seen)
            txn.execute('\n                SELECT user_id, access_token, ip,\n                       MAX(device_id), MAX(user_agent), MAX(last_seen),\n                       COUNT(*)\n                FROM (\n                    SELECT DISTINCT user_id, access_token, ip\n                    FROM user_ips\n                    WHERE {}\n                ) c\n                INNER JOIN user_ips USING (user_id, access_token, ip)\n                GROUP BY user_id, access_token, ip\n                HAVING count(*) > 1\n                '.format(clause), args)
            res = cast(List[Tuple[str, str, str, Optional[str], str, int, int]], txn.fetchall())
            for i in res:
                (user_id, access_token, ip, device_id, user_agent, last_seen, count) = i
                txn.execute('\n                    DELETE FROM user_ips\n                    WHERE user_id = ? AND access_token = ? AND ip = ? AND last_seen < ?\n                    ', (user_id, access_token, ip, last_seen))
                if txn.rowcount == count - 1:
                    continue
                elif txn.rowcount >= count:
                    raise Exception("We deleted more duplicate rows from 'user_ips' than expected")
                txn.execute('\n                    DELETE FROM user_ips\n                    WHERE user_id = ? AND access_token = ? AND ip = ?\n                    ', (user_id, access_token, ip))
                txn.execute('\n                    INSERT INTO user_ips\n                    (user_id, access_token, ip, device_id, user_agent, last_seen)\n                    VALUES (?, ?, ?, ?, ?, ?)\n                    ', (user_id, access_token, ip, device_id, user_agent, last_seen))
            self.db_pool.updates._background_update_progress_txn(txn, 'user_ips_remove_dupes', {'last_seen': end_last_seen})
        await self.db_pool.runInteraction('user_ips_dups_remove', remove)
        if last:
            await self.db_pool.updates._end_background_update('user_ips_remove_dupes')
        return batch_size

    async def _devices_last_seen_update(self, progress: JsonDict, batch_size: int) -> int:
        """Background update to insert last seen info into devices table"""
        last_user_id: str = progress.get('last_user_id', '')
        last_device_id: str = progress.get('last_device_id', '')

        def _devices_last_seen_update_txn(txn: LoggingTransaction) -> int:
            if False:
                return 10
            where_args: List[Union[str, int]]
            (where_clause, where_args) = make_tuple_comparison_clause([('user_id', last_user_id), ('device_id', last_device_id)])
            sql = '\n                SELECT\n                    last_seen, ip, user_agent, user_id, device_id\n                FROM (\n                    SELECT\n                        user_id, device_id, MAX(u.last_seen) AS last_seen\n                    FROM devices\n                    INNER JOIN user_ips AS u USING (user_id, device_id)\n                    WHERE %(where_clause)s\n                    GROUP BY user_id, device_id\n                    ORDER BY user_id ASC, device_id ASC\n                    LIMIT ?\n                ) c\n                INNER JOIN user_ips AS u USING (user_id, device_id, last_seen)\n            ' % {'where_clause': where_clause}
            txn.execute(sql, where_args + [batch_size])
            rows = cast(List[Tuple[int, str, str, str, str]], txn.fetchall())
            if not rows:
                return 0
            sql = '\n                UPDATE devices\n                SET last_seen = ?, ip = ?, user_agent = ?\n                WHERE user_id = ? AND device_id = ?\n            '
            txn.execute_batch(sql, rows)
            (_, _, _, user_id, device_id) = rows[-1]
            self.db_pool.updates._background_update_progress_txn(txn, 'devices_last_seen', {'last_user_id': user_id, 'last_device_id': device_id})
            return len(rows)
        updated = await self.db_pool.runInteraction('_devices_last_seen_update', _devices_last_seen_update_txn)
        if not updated:
            await self.db_pool.updates._end_background_update('devices_last_seen')
        return updated

class ClientIpWorkerStore(ClientIpBackgroundUpdateStore, MonthlyActiveUsersWorkerStore):

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(database, db_conn, hs)
        if hs.config.redis.redis_enabled:
            self._update_on_this_worker = hs.config.worker.run_background_tasks
        else:
            self._update_on_this_worker = hs.get_instance_name() == 'master'
        self.user_ips_max_age = hs.config.server.user_ips_max_age
        self.client_ip_last_seen = LruCache[Tuple[str, str, str], int](cache_name='client_ip_last_seen', max_size=50000)
        if hs.config.worker.run_background_tasks and self.user_ips_max_age:
            self._clock.looping_call(self._prune_old_user_ips, 5 * 1000)
        if self._update_on_this_worker:
            self._batch_row_update: Dict[Tuple[str, str, str], Tuple[str, Optional[str], int]] = {}
            self._client_ip_looper = self._clock.looping_call(self._update_client_ips_batch, 5 * 1000)
            self.hs.get_reactor().addSystemEventTrigger('before', 'shutdown', self._update_client_ips_batch)

    @wrap_as_background_process('prune_old_user_ips')
    async def _prune_old_user_ips(self) -> None:
        """Removes entries in user IPs older than the configured period."""
        if self.user_ips_max_age is None:
            return
        if not await self.db_pool.updates.has_completed_background_update('devices_last_seen'):
            return
        sql = '\n            DELETE FROM user_ips\n            WHERE last_seen <= (\n                SELECT COALESCE(MAX(last_seen), -1)\n                FROM (\n                    SELECT last_seen FROM user_ips\n                    WHERE last_seen <= ?\n                    ORDER BY last_seen ASC\n                    LIMIT 5000\n                ) AS u\n            )\n        '
        timestamp = self._clock.time_msec() - self.user_ips_max_age

        def _prune_old_user_ips_txn(txn: LoggingTransaction) -> None:
            if False:
                return 10
            txn.execute(sql, (timestamp,))
        await self.db_pool.runInteraction('_prune_old_user_ips', _prune_old_user_ips_txn)

    async def _get_last_client_ip_by_device_from_database(self, user_id: str, device_id: Optional[str]) -> Dict[Tuple[str, str], DeviceLastConnectionInfo]:
        """For each device_id listed, give the user_ip it was last seen on.

        The result might be slightly out of date as client IPs are inserted in batches.

        Args:
            user_id: The user to fetch devices for.
            device_id: If None fetches all devices for the user

        Returns:
            A dictionary mapping a tuple of (user_id, device_id) to DeviceLastConnectionInfo.
        """
        keyvalues = {'user_id': user_id}
        if device_id is not None:
            keyvalues['device_id'] = device_id
        res = cast(List[Tuple[str, Optional[str], Optional[str], str, Optional[int]]], await self.db_pool.simple_select_list(table='devices', keyvalues=keyvalues, retcols=('user_id', 'ip', 'user_agent', 'device_id', 'last_seen')))
        return {(user_id, device_id): DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip=ip, user_agent=user_agent, last_seen=last_seen) for (user_id, ip, user_agent, device_id, last_seen) in res}

    async def _get_user_ip_and_agents_from_database(self, user: UserID, since_ts: int=0) -> List[LastConnectionInfo]:
        """Fetch the IPs and user agents for a user since the given timestamp.

        The result might be slightly out of date as client IPs are inserted in batches.

        Args:
            user: The user for which to fetch IP addresses and user agents.
            since_ts: The timestamp after which to fetch IP addresses and user agents,
                in milliseconds.

        Returns:
            A list of dictionaries, each containing:
             * `access_token`: The access token used.
             * `ip`: The IP address used.
             * `user_agent`: The last user agent seen for this access token and IP
               address combination.
             * `last_seen`: The timestamp at which this access token and IP address
               combination was last seen, in milliseconds.

            Only the latest user agent for each access token and IP address combination
            is available.
        """
        user_id = user.to_string()

        def get_recent(txn: LoggingTransaction) -> List[Tuple[str, str, str, int]]:
            if False:
                return 10
            txn.execute('\n                SELECT access_token, ip, user_agent, last_seen FROM user_ips\n                WHERE last_seen >= ? AND user_id = ?\n                ORDER BY last_seen\n                DESC\n                ', (since_ts, user_id))
            return cast(List[Tuple[str, str, str, int]], txn.fetchall())
        rows = await self.db_pool.runInteraction(desc='get_user_ip_and_agents', func=get_recent)
        return [{'access_token': access_token, 'ip': ip, 'user_agent': user_agent, 'last_seen': last_seen} for (access_token, ip, user_agent, last_seen) in rows]

    async def insert_client_ip(self, user_id: str, access_token: str, ip: str, user_agent: str, device_id: Optional[str], now: Optional[int]=None) -> None:
        if user_agent == 'sync-v3-proxy-':
            return
        if not now:
            now = int(self._clock.time_msec())
        key = (user_id, access_token, ip)
        try:
            last_seen = self.client_ip_last_seen.get(key)
        except KeyError:
            last_seen = None
        if last_seen is not None and now - last_seen < LAST_SEEN_GRANULARITY:
            return
        self.client_ip_last_seen.set(key, now)
        if self._update_on_this_worker:
            await self.populate_monthly_active_users(user_id)
            self._batch_row_update[key] = (user_agent, device_id, now)
        else:
            self.hs.get_replication_command_handler().send_user_ip(user_id, access_token, ip, user_agent, device_id, now)

    @wrap_as_background_process('update_client_ips')
    async def _update_client_ips_batch(self) -> None:
        assert self._update_on_this_worker, 'This worker is not designated to update client IPs'
        if not self.db_pool.is_running():
            return
        to_update = self._batch_row_update
        self._batch_row_update = {}
        if to_update:
            await self.db_pool.runInteraction('_update_client_ips_batch', self._update_client_ips_batch_txn, to_update)

    def _update_client_ips_batch_txn(self, txn: LoggingTransaction, to_update: Mapping[Tuple[str, str, str], Tuple[str, Optional[str], int]]) -> None:
        if False:
            return 10
        assert self._update_on_this_worker, 'This worker is not designated to update client IPs'
        user_ips_keys = []
        user_ips_values = []
        devices_keys = []
        devices_values = []
        for entry in to_update.items():
            ((user_id, access_token, ip), (user_agent, device_id, last_seen)) = entry
            user_ips_keys.append((user_id, access_token, ip))
            user_ips_values.append((user_agent, device_id, last_seen))
            if device_id:
                devices_keys.append((user_id, device_id))
                devices_values.append((user_agent, last_seen, ip))
        self.db_pool.simple_upsert_many_txn(txn, table='user_ips', key_names=('user_id', 'access_token', 'ip'), key_values=user_ips_keys, value_names=('user_agent', 'device_id', 'last_seen'), value_values=user_ips_values)
        if devices_values:
            self.db_pool.simple_update_many_txn(txn, table='devices', key_names=('user_id', 'device_id'), key_values=devices_keys, value_names=('user_agent', 'last_seen', 'ip'), value_values=devices_values)

    async def get_last_client_ip_by_device(self, user_id: str, device_id: Optional[str]) -> Dict[Tuple[str, str], DeviceLastConnectionInfo]:
        """For each device_id listed, give the user_ip it was last seen on

        Args:
            user_id: The user to fetch devices for.
            device_id: If None fetches all devices for the user

        Returns:
            A dictionary mapping a tuple of (user_id, device_id) to DeviceLastConnectionInfo.
        """
        ret = await self._get_last_client_ip_by_device_from_database(user_id, device_id)
        if not self._update_on_this_worker:
            return ret
        for key in self._batch_row_update:
            (uid, _access_token, ip) = key
            if uid == user_id:
                (user_agent, did, last_seen) = self._batch_row_update[key]
                if did is None:
                    continue
                if not device_id or did == device_id:
                    ret[user_id, did] = DeviceLastConnectionInfo(user_id=user_id, ip=ip, user_agent=user_agent, device_id=did, last_seen=last_seen)
        return ret

    async def get_user_ip_and_agents(self, user: UserID, since_ts: int=0) -> List[LastConnectionInfo]:
        """Fetch the IPs and user agents for a user since the given timestamp.

        Args:
            user: The user for which to fetch IP addresses and user agents.
            since_ts: The timestamp after which to fetch IP addresses and user agents,
                in milliseconds.

        Returns:
            A list of dictionaries, each containing:
             * `access_token`: The access token used.
             * `ip`: The IP address used.
             * `user_agent`: The last user agent seen for this access token and IP
               address combination.
             * `last_seen`: The timestamp at which this access token and IP address
               combination was last seen, in milliseconds.

            Only the latest user agent for each access token and IP address combination
            is available.
        """
        rows_from_db = await self._get_user_ip_and_agents_from_database(user, since_ts)
        if not self._update_on_this_worker:
            return rows_from_db
        results: Dict[Tuple[str, str], LastConnectionInfo] = {(connection['access_token'], connection['ip']): connection for connection in rows_from_db}
        user_id = user.to_string()
        for key in self._batch_row_update:
            (uid, access_token, ip) = key
            if uid == user_id:
                (user_agent, _, last_seen) = self._batch_row_update[key]
                if last_seen >= since_ts:
                    results[access_token, ip] = {'access_token': access_token, 'ip': ip, 'user_agent': user_agent, 'last_seen': last_seen}
        return list(results.values())

    async def get_last_seen_for_user_id(self, user_id: str) -> Optional[int]:
        """Get the last seen timestamp for a user, if we have it."""
        return await self.db_pool.simple_select_one_onecol(table='user_ips', keyvalues={'user_id': user_id}, retcol='MAX(last_seen)', allow_none=True, desc='get_last_seen_for_user_id')