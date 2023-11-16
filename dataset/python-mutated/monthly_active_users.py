import logging
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, cast
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction, make_in_list_sql_clause
from synapse.storage.databases.main.registration import RegistrationWorkerStore
from synapse.util.caches.descriptors import cached
from synapse.util.threepids import canonicalise_email
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
LAST_SEEN_GRANULARITY = 60 * 60 * 1000

class MonthlyActiveUsersWorkerStore(RegistrationWorkerStore):

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        super().__init__(database, db_conn, hs)
        self._clock = hs.get_clock()
        self.hs = hs
        if hs.config.redis.redis_enabled:
            self._update_on_this_worker = hs.config.worker.run_background_tasks
        else:
            self._update_on_this_worker = hs.get_instance_name() == 'master'
        self._limit_usage_by_mau = hs.config.server.limit_usage_by_mau
        self._max_mau_value = hs.config.server.max_mau_value
        self._mau_stats_only = hs.config.server.mau_stats_only
        if self._update_on_this_worker:
            self.db_pool.new_transaction(db_conn, 'initialise_mau_threepids', [], [], [], self._initialise_reserved_users, hs.config.server.mau_limits_reserved_threepids[:self._max_mau_value])

    @cached(num_args=0)
    async def get_monthly_active_count(self) -> int:
        """Generates current count of monthly active users

        Returns:
            Number of current monthly active users
        """

        def _count_users(txn: LoggingTransaction) -> int:
            if False:
                print('Hello World!')
            sql = "\n                SELECT COUNT(*)\n                FROM monthly_active_users\n                    LEFT JOIN users\n                    ON monthly_active_users.user_id=users.name\n                WHERE (users.appservice_id IS NULL OR users.appservice_id = '');\n            "
            txn.execute(sql)
            (count,) = cast(Tuple[int], txn.fetchone())
            return count
        return await self.db_pool.runInteraction('count_users', _count_users)

    @cached(num_args=0)
    async def get_monthly_active_count_by_service(self) -> Mapping[str, int]:
        """Generates current count of monthly active users broken down by service.
        A service is typically an appservice but also includes native matrix users.
        Since the `monthly_active_users` table is populated from the `user_ips` table
        `config.appservice.track_appservice_user_ips` must be set to `true` for this
        method to return anything other than native matrix users.

        Returns:
            A mapping between app_service_id and the number of occurrences.

        """

        def _count_users_by_service(txn: LoggingTransaction) -> Dict[str, int]:
            if False:
                print('Hello World!')
            sql = "\n                SELECT COALESCE(appservice_id, 'native'), COUNT(*)\n                FROM monthly_active_users\n                LEFT JOIN users ON monthly_active_users.user_id=users.name\n                GROUP BY appservice_id;\n            "
            txn.execute(sql)
            result = cast(List[Tuple[str, int]], txn.fetchall())
            return dict(result)
        return await self.db_pool.runInteraction('count_users_by_service', _count_users_by_service)

    async def get_monthly_active_users_by_service(self, start_timestamp: Optional[int]=None, end_timestamp: Optional[int]=None) -> List[Tuple[str, str]]:
        """Generates list of monthly active users and their services.
        Please see "get_monthly_active_count_by_service" docstring for more details
        about services.

        Arguments:
            start_timestamp: If specified, only include users that were first active
                at or after this point
            end_timestamp: If specified, only include users that were first active
                at or before this point

        Returns:
            A list of tuples (appservice_id, user_id). "native" is emitted as the
            appservice for users that don't come from appservices (i.e. native Matrix
            users).

        """
        if start_timestamp is not None and end_timestamp is not None:
            where_clause = 'WHERE "timestamp" >= ? and "timestamp" <= ?'
            query_params = [start_timestamp, end_timestamp]
        elif start_timestamp is not None:
            where_clause = 'WHERE "timestamp" >= ?'
            query_params = [start_timestamp]
        elif end_timestamp is not None:
            where_clause = 'WHERE "timestamp" <= ?'
            query_params = [end_timestamp]
        else:
            where_clause = ''
            query_params = []

        def _list_users(txn: LoggingTransaction) -> List[Tuple[str, str]]:
            if False:
                return 10
            sql = f"\n                    SELECT COALESCE(appservice_id, 'native'), user_id\n                    FROM monthly_active_users\n                    LEFT JOIN users ON monthly_active_users.user_id=users.name\n                    {where_clause};\n                "
            txn.execute(sql, query_params)
            return cast(List[Tuple[str, str]], txn.fetchall())
        return await self.db_pool.runInteraction('list_users', _list_users)

    async def get_registered_reserved_users(self) -> List[str]:
        """Of the reserved threepids defined in config, retrieve those that are associated
        with registered users

        Returns:
            User IDs of actual users that are reserved
        """
        users = []
        for tp in self.hs.config.server.mau_limits_reserved_threepids[:self.hs.config.server.max_mau_value]:
            user_id = await self.hs.get_datastores().main.get_user_id_by_threepid(tp['medium'], canonicalise_email(tp['address']))
            if user_id:
                users.append(user_id)
        return users

    @cached(num_args=1)
    async def user_last_seen_monthly_active(self, user_id: str) -> Optional[int]:
        """
        Checks if a given user is part of the monthly active user group

        Arguments:
            user_id: user to add/update

        Return:
            Timestamp since last seen, None if never seen
        """
        return await self.db_pool.simple_select_one_onecol(table='monthly_active_users', keyvalues={'user_id': user_id}, retcol='timestamp', allow_none=True, desc='user_last_seen_monthly_active')

    @wrap_as_background_process('reap_monthly_active_users')
    async def reap_monthly_active_users(self) -> None:
        """Cleans out monthly active user table to ensure that no stale
        entries exist.
        """

        def _reap_users(txn: LoggingTransaction, reserved_users: List[str]) -> None:
            if False:
                i = 10
                return i + 15
            '\n            Args:\n                reserved_users: reserved users to preserve\n            '
            thirty_days_ago = int(self._clock.time_msec()) - 1000 * 60 * 60 * 24 * 30
            (in_clause, in_clause_args) = make_in_list_sql_clause(self.database_engine, 'user_id', reserved_users)
            txn.execute('DELETE FROM monthly_active_users WHERE timestamp < ? AND NOT %s' % (in_clause,), [thirty_days_ago] + in_clause_args)
            if self._limit_usage_by_mau:
                num_of_non_reserved_users_to_remove = max(self._max_mau_value - len(reserved_users), 0)
                sql = '\n                    DELETE FROM monthly_active_users\n                    WHERE user_id NOT IN (\n                        SELECT user_id FROM monthly_active_users\n                        WHERE NOT %s\n                        ORDER BY timestamp DESC\n                        LIMIT ?\n                    )\n                    AND NOT %s\n                ' % (in_clause, in_clause)
                query_args = in_clause_args + [num_of_non_reserved_users_to_remove] + in_clause_args
                txn.execute(sql, query_args)
            self._invalidate_all_cache_and_stream(txn, self.user_last_seen_monthly_active)
            self._invalidate_cache_and_stream(txn, self.get_monthly_active_count, ())
        reserved_users = await self.get_registered_reserved_users()
        await self.db_pool.runInteraction('reap_monthly_active_users', _reap_users, reserved_users)

    def _initialise_reserved_users(self, txn: LoggingTransaction, threepids: List[dict]) -> None:
        if False:
            return 10
        'Ensures that reserved threepids are accounted for in the MAU table, should\n        be called on start up.\n\n        Args:\n            txn:\n            threepids: List of threepid dicts to reserve\n        '
        assert self._update_on_this_worker, 'This worker is not designated to update MAUs'
        for tp in threepids:
            user_id = self.get_user_id_by_threepid_txn(txn, tp['medium'], tp['address'])
            if user_id:
                is_support = self.is_support_user_txn(txn, user_id)
                if not is_support:
                    self.db_pool.simple_upsert_txn(txn, table='monthly_active_users', keyvalues={'user_id': user_id}, values={'timestamp': int(self._clock.time_msec())})
            else:
                logger.warning('mau limit reserved threepid %s not found in db' % tp)

    async def upsert_monthly_active_user(self, user_id: str) -> None:
        """Updates or inserts the user into the monthly active user table, which
        is used to track the current MAU usage of the server

        Args:
            user_id: user to add/update
        """
        assert self._update_on_this_worker, 'This worker is not designated to update MAUs'
        is_support = await self.is_support_user(user_id)
        if is_support:
            return
        await self.db_pool.runInteraction('upsert_monthly_active_user', self.upsert_monthly_active_user_txn, user_id)

    def upsert_monthly_active_user_txn(self, txn: LoggingTransaction, user_id: str) -> None:
        if False:
            while True:
                i = 10
        "Updates or inserts monthly active user member\n\n        We consciously do not call is_support_txn from this method because it\n        is not possible to cache the response. is_support_txn will be false in\n        almost all cases, so it seems reasonable to call it only for\n        upsert_monthly_active_user and to call is_support_txn manually\n        for cases where upsert_monthly_active_user_txn is called directly,\n        like _initialise_reserved_users\n\n        In short, don't call this method with support users. (Support users\n        should not appear in the MAU stats).\n\n        Args:\n            txn:\n            user_id: user to add/update\n        "
        assert self._update_on_this_worker, 'This worker is not designated to update MAUs'
        self.db_pool.simple_upsert_txn(txn, table='monthly_active_users', keyvalues={'user_id': user_id}, values={'timestamp': int(self._clock.time_msec())})
        self._invalidate_cache_and_stream(txn, self.get_monthly_active_count, ())
        self._invalidate_cache_and_stream(txn, self.get_monthly_active_count_by_service, ())
        self._invalidate_cache_and_stream(txn, self.user_last_seen_monthly_active, (user_id,))

    async def populate_monthly_active_users(self, user_id: str) -> None:
        """Checks on the state of monthly active user limits and optionally
        add the user to the monthly active tables

        Args:
            user_id: the user_id to query
        """
        assert self._update_on_this_worker, 'This worker is not designated to update MAUs'
        if self._limit_usage_by_mau or self._mau_stats_only:
            is_guest = await self.is_guest(user_id)
            if is_guest:
                return
            is_trial = await self.is_trial_user(user_id)
            if is_trial:
                return
            last_seen_timestamp = await self.user_last_seen_monthly_active(user_id)
            now = self.hs.get_clock().time_msec()
            if last_seen_timestamp is None:
                if not self._limit_usage_by_mau:
                    await self.upsert_monthly_active_user(user_id)
                else:
                    count = await self.get_monthly_active_count()
                    if count < self._max_mau_value:
                        await self.upsert_monthly_active_user(user_id)
            elif now - last_seen_timestamp > LAST_SEEN_GRANULARITY:
                await self.upsert_monthly_active_user(user_id)