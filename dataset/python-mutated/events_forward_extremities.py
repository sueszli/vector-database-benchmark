import logging
from typing import List, Optional, Tuple, cast
from synapse.api.errors import SynapseError
from synapse.storage.database import LoggingTransaction
from synapse.storage.databases.main import CacheInvalidationWorkerStore
from synapse.storage.databases.main.event_federation import EventFederationWorkerStore
logger = logging.getLogger(__name__)

class EventForwardExtremitiesStore(EventFederationWorkerStore, CacheInvalidationWorkerStore):

    async def delete_forward_extremities_for_room(self, room_id: str) -> int:
        """Delete any extra forward extremities for a room.

        Invalidates the "get_latest_event_ids_in_room" cache if any forward
        extremities were deleted.

        Returns count deleted.
        """

        def delete_forward_extremities_for_room_txn(txn: LoggingTransaction) -> int:
            if False:
                print('Hello World!')
            sql = '\n                SELECT event_id FROM event_forward_extremities\n                INNER JOIN events USING (room_id, event_id)\n                WHERE room_id = ?\n                ORDER BY stream_ordering DESC\n                LIMIT 1\n            '
            txn.execute(sql, (room_id,))
            rows = txn.fetchall()
            try:
                event_id = rows[0][0]
                logger.debug('Found event_id %s as the forward extremity to keep for room %s', event_id, room_id)
            except KeyError:
                msg = 'No forward extremity event found for room %s' % room_id
                logger.warning(msg)
                raise SynapseError(400, msg)
            sql = '\n                DELETE FROM event_forward_extremities\n                WHERE event_id != ? AND room_id = ?\n            '
            txn.execute(sql, (event_id, room_id))
            deleted_count = txn.rowcount
            logger.info('Deleted %s extra forward extremities for room %s', deleted_count, room_id)
            if deleted_count > 0:
                self._invalidate_cache_and_stream(txn, self.get_latest_event_ids_in_room, (room_id,))
            return deleted_count
        return await self.db_pool.runInteraction('delete_forward_extremities_for_room', delete_forward_extremities_for_room_txn)

    async def get_forward_extremities_for_room(self, room_id: str) -> List[Tuple[str, int, int, Optional[int]]]:
        """
        Get list of forward extremities for a room.

        Returns:
            A list of tuples of event_id, state_group, depth, and received_ts.
        """

        def get_forward_extremities_for_room_txn(txn: LoggingTransaction) -> List[Tuple[str, int, int, Optional[int]]]:
            if False:
                for i in range(10):
                    print('nop')
            sql = '\n                SELECT event_id, state_group, depth, received_ts\n                FROM event_forward_extremities\n                INNER JOIN event_to_state_groups USING (event_id)\n                INNER JOIN events USING (room_id, event_id)\n                WHERE room_id = ?\n            '
            txn.execute(sql, (room_id,))
            return cast(List[Tuple[str, int, int, Optional[int]]], txn.fetchall())
        return await self.db_pool.runInteraction('get_forward_extremities_for_room', get_forward_extremities_for_room_txn)