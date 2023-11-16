import logging
from collections import defaultdict
from typing import Collection, Dict, Set
from twisted.internet import defer
from twisted.internet.defer import Deferred
from synapse.logging.context import PreserveLoggingContext, make_deferred_yieldable
from synapse.logging.opentracing import trace_with_opname
from synapse.storage.databases.main.events_worker import EventsWorkerStore
from synapse.storage.databases.main.room import RoomWorkerStore
from synapse.util import unwrapFirstError
from synapse.util.cancellation import cancellable
logger = logging.getLogger(__name__)

class PartialStateEventsTracker:
    """Keeps track of which events have partial state, after a partial-state join"""

    def __init__(self, store: EventsWorkerStore):
        if False:
            return 10
        self._store = store
        self._observers: Dict[str, Set[Deferred[None]]] = defaultdict(set)

    def notify_un_partial_stated(self, event_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Notify that we now have full state for a given event\n\n        Called by the state-resynchronization loop whenever we resynchronize the state\n        for a particular event. Unblocks any callers to await_full_state() for that\n        event.\n\n        Args:\n            event_id: the event that now has full state.\n        '
        observers = self._observers.pop(event_id, None)
        if not observers:
            return
        logger.info('Notifying %i things waiting for un-partial-stating of event %s', len(observers), event_id)
        with PreserveLoggingContext():
            for o in observers:
                o.callback(None)

    @trace_with_opname('PartialStateEventsTracker.await_full_state')
    @cancellable
    async def await_full_state(self, event_ids: Collection[str]) -> None:
        """Wait for all the given events to have full state.

        Args:
            event_ids: the list of event ids that we want full state for
        """
        partial_state_event_ids = [ev for (ev, p) in (await self._store.get_partial_state_events(event_ids)).items() if p]
        if not partial_state_event_ids:
            return
        logger.info('Awaiting un-partial-stating of events %s', partial_state_event_ids, stack_info=True)
        observers: Dict[str, Deferred[None]] = {event_id: Deferred() for event_id in partial_state_event_ids}
        for (event_id, observer) in observers.items():
            self._observers[event_id].add(observer)
        try:
            for (event_id, partial) in (await self._store.get_partial_state_events(observers.keys())).items():
                if not partial and (not observers[event_id].called):
                    observers[event_id].callback(None)
            await make_deferred_yieldable(defer.gatherResults(observers.values(), consumeErrors=True)).addErrback(unwrapFirstError)
            logger.info('Events %s all un-partial-stated', observers.keys())
        finally:
            for (event_id, observer) in observers.items():
                observer_set = self._observers.get(event_id)
                if observer_set:
                    observer_set.discard(observer)
                    if not observer_set:
                        del self._observers[event_id]

class PartialCurrentStateTracker:
    """Keeps track of which rooms have partial state, after partial-state joins"""

    def __init__(self, store: RoomWorkerStore):
        if False:
            while True:
                i = 10
        self._store = store
        self._observers: Dict[str, Set[Deferred[None]]] = defaultdict(set)

    def notify_un_partial_stated(self, room_id: str) -> None:
        if False:
            while True:
                i = 10
        'Notify that we now have full current state for a given room\n\n        Unblocks any callers to await_full_state() for that room.\n\n        Args:\n            room_id: the room that now has full current state.\n        '
        observers = self._observers.pop(room_id, None)
        if not observers:
            return
        logger.info('Notifying %i things waiting for un-partial-stating of room %s', len(observers), room_id)
        with PreserveLoggingContext():
            for o in observers:
                o.callback(None)

    @trace_with_opname('PartialCurrentStateTracker.await_full_state')
    @cancellable
    async def await_full_state(self, room_id: str) -> None:
        d: Deferred[None] = Deferred()
        self._observers.setdefault(room_id, set()).add(d)
        try:
            has_partial_state = await self._store.is_partial_state_room(room_id)
            if not has_partial_state:
                return
            logger.info('Awaiting un-partial-stating of room %s', room_id, stack_info=True)
            await make_deferred_yieldable(d)
            logger.info('Room has un-partial-stated')
        finally:
            ds = self._observers.get(room_id)
            if ds is not None:
                ds.discard(d)
                if not ds:
                    self._observers.pop(room_id, None)