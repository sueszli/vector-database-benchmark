import logging
from typing import TYPE_CHECKING, Optional
from synapse.events.utils import prune_event_dict
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.storage._base import SQLBaseStore
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction
from synapse.storage.databases.main.cache import CacheInvalidationWorkerStore
from synapse.storage.databases.main.events_worker import EventsWorkerStore
from synapse.util import json_encoder
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class CensorEventsStore(EventsWorkerStore, CacheInvalidationWorkerStore, SQLBaseStore):

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(database, db_conn, hs)
        if hs.config.worker.run_background_tasks and self.hs.config.server.redaction_retention_period is not None:
            hs.get_clock().looping_call(self._censor_redactions, 5 * 60 * 1000)

    @wrap_as_background_process('_censor_redactions')
    async def _censor_redactions(self) -> None:
        """Censors all redactions older than the configured period that haven't
        been censored yet.

        By censor we mean update the event_json table with the redacted event.
        """
        if self.hs.config.server.redaction_retention_period is None:
            return
        if not await self.db_pool.updates.has_completed_background_update('redactions_have_censored_ts_idx'):
            return
        before_ts = self._clock.time_msec() - self.hs.config.server.redaction_retention_period
        sql = '\n            SELECT redactions.event_id, redacts FROM redactions\n            LEFT JOIN events AS original_event ON (\n                redacts = original_event.event_id\n            )\n            WHERE NOT have_censored\n            AND redactions.received_ts <= ?\n            ORDER BY redactions.received_ts ASC\n            LIMIT ?\n        '
        rows = await self.db_pool.execute('_censor_redactions_fetch', sql, before_ts, 100)
        updates = []
        for (redaction_id, event_id) in rows:
            redaction_event = await self.get_event(redaction_id, allow_none=True)
            original_event = await self.get_event(event_id, allow_rejected=True, allow_none=True)
            if redaction_event and original_event and original_event.internal_metadata.is_redacted():
                pruned_json: Optional[str] = json_encoder.encode(prune_event_dict(original_event.room_version, original_event.get_dict()))
            else:
                pruned_json = None
            updates.append((redaction_id, event_id, pruned_json))

        def _update_censor_txn(txn: LoggingTransaction) -> None:
            if False:
                print('Hello World!')
            for (redaction_id, event_id, pruned_json) in updates:
                if pruned_json:
                    self._censor_event_txn(txn, event_id, pruned_json)
                self.db_pool.simple_update_one_txn(txn, table='redactions', keyvalues={'event_id': redaction_id}, updatevalues={'have_censored': True})
        await self.db_pool.runInteraction('_update_censor_txn', _update_censor_txn)

    def _censor_event_txn(self, txn: LoggingTransaction, event_id: str, pruned_json: str) -> None:
        if False:
            print('Hello World!')
        'Censor an event by replacing its JSON in the event_json table with the\n        provided pruned JSON.\n\n        Args:\n            txn: The database transaction.\n            event_id: The ID of the event to censor.\n            pruned_json: The pruned JSON\n        '
        self.db_pool.simple_update_one_txn(txn, table='event_json', keyvalues={'event_id': event_id}, updatevalues={'json': pruned_json})

    async def expire_event(self, event_id: str) -> None:
        """Retrieve and expire an event that has expired, and delete its associated
        expiry timestamp. If the event can't be retrieved, delete its associated
        timestamp so we don't try to expire it again in the future.

        Args:
             event_id: The ID of the event to delete.
        """
        event = await self.get_event(event_id)

        def delete_expired_event_txn(txn: LoggingTransaction) -> None:
            if False:
                print('Hello World!')
            self._delete_event_expiry_txn(txn, event_id)
            if not event:
                logger.warning("Can't expire event %s because we don't have it.", event_id)
                return
            pruned_json = json_encoder.encode(prune_event_dict(event.room_version, event.get_dict()))
            self._censor_event_txn(txn, event.event_id, pruned_json)
            self.invalidate_get_event_cache_after_txn(txn, event.event_id)
            self._send_invalidation_to_replication(txn, '_get_event_cache', (event.event_id,))
        await self.db_pool.runInteraction('delete_expired_event', delete_expired_event_txn)

    def _delete_event_expiry_txn(self, txn: LoggingTransaction, event_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Delete the expiry timestamp associated with an event ID without deleting the\n        actual event.\n\n        Args:\n            txn: The transaction to use to perform the deletion.\n            event_id: The event ID to delete the associated expiry timestamp of.\n        '
        self.db_pool.simple_delete_txn(txn=txn, table='event_expiry', keyvalues={'event_id': event_id})