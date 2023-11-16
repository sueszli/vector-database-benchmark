import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.internet.interfaces import IDelayedCall
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.push import Pusher, PusherConfig, PusherConfigException, ThrottleParams
from synapse.push.mailer import Mailer
from synapse.push.push_types import EmailReason
from synapse.storage.databases.main.event_push_actions import EmailPushAction
from synapse.util.threepids import validate_email
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
DELAY_BEFORE_MAIL_MS = 10 * 60 * 1000
THROTTLE_START_MS = 10 * 60 * 1000
THROTTLE_MAX_MS = 24 * 60 * 60 * 1000
THROTTLE_MULTIPLIER = 144
THROTTLE_RESET_AFTER_MS = 12 * 60 * 60 * 1000
INCLUDE_ALL_UNREAD_NOTIFS = False

class EmailPusher(Pusher):
    """
    A pusher that sends email notifications about events (approximately)
    when they happen.
    This shares quite a bit of code with httpusher: it would be good to
    factor out the common parts
    """

    def __init__(self, hs: 'HomeServer', pusher_config: PusherConfig, mailer: Mailer):
        if False:
            return 10
        super().__init__(hs, pusher_config)
        self.mailer = mailer
        self.store = self.hs.get_datastores().main
        self.email = pusher_config.pushkey
        self.timed_call: Optional[IDelayedCall] = None
        self.throttle_params: Dict[str, ThrottleParams] = {}
        self._inited = False
        self._is_processing = False
        try:
            validate_email(self.email)
        except ValueError:
            raise PusherConfigException('Invalid email')

    def on_started(self, should_check_for_notifs: bool) -> None:
        if False:
            i = 10
            return i + 15
        "Called when this pusher has been started.\n\n        Args:\n            should_check_for_notifs: Whether we should immediately\n                check for push to send. Set to False only if it's known there\n                is nothing to send\n        "
        if should_check_for_notifs and self.mailer is not None:
            self._start_processing()

    def on_stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.timed_call:
            try:
                self.timed_call.cancel()
            except (AlreadyCalled, AlreadyCancelled):
                pass
            self.timed_call = None

    def on_new_receipts(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def on_timer(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.timed_call = None
        self._start_processing()

    def _start_processing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._is_processing:
            return
        run_as_background_process('emailpush.process', self._process)

    def _pause_processing(self) -> None:
        if False:
            return 10
        'Used by tests to temporarily pause processing of events.\n\n        Asserts that its not currently processing.\n        '
        assert not self._is_processing
        self._is_processing = True

    def _resume_processing(self) -> None:
        if False:
            print('Hello World!')
        'Used by tests to resume processing of events after pausing.'
        assert self._is_processing
        self._is_processing = False
        self._start_processing()

    async def _process(self) -> None:
        assert not self._is_processing
        try:
            self._is_processing = True
            if not self._inited:
                assert self.pusher_id is not None
                self.throttle_params = await self.store.get_throttle_params_by_room(self.pusher_id)
                self._inited = True
            while True:
                starting_max_ordering = self.max_stream_ordering
                try:
                    await self._unsafe_process()
                except Exception:
                    logger.exception('Exception processing notifs')
                if self.max_stream_ordering == starting_max_ordering:
                    break
        finally:
            self._is_processing = False

    async def _unsafe_process(self) -> None:
        """
        Main logic of the push loop without the wrapper function that sets
        up logging, measures and guards against multiple instances of it
        being run.
        """
        start = 0 if INCLUDE_ALL_UNREAD_NOTIFS else self.last_stream_ordering
        unprocessed = await self.store.get_unread_push_actions_for_user_in_range_for_email(self.user_id, start, self.max_stream_ordering)
        soonest_due_at: Optional[int] = None
        if not unprocessed:
            await self.save_last_stream_ordering_and_success(self.max_stream_ordering)
            return
        for push_action in unprocessed:
            received_at = push_action.received_ts
            if received_at is None:
                received_at = 0
            notif_ready_at = received_at + DELAY_BEFORE_MAIL_MS
            room_ready_at = self.room_ready_to_notify_at(push_action.room_id)
            should_notify_at = max(notif_ready_at, room_ready_at)
            if should_notify_at <= self.clock.time_msec():
                reason: EmailReason = {'room_id': push_action.room_id, 'now': self.clock.time_msec(), 'received_at': received_at, 'delay_before_mail_ms': DELAY_BEFORE_MAIL_MS, 'last_sent_ts': self.get_room_last_sent_ts(push_action.room_id), 'throttle_ms': self.get_room_throttle_ms(push_action.room_id)}
                await self.send_notification(unprocessed, reason)
                await self.save_last_stream_ordering_and_success(max((ea.stream_ordering for ea in unprocessed)))
                for ea in unprocessed:
                    await self.sent_notif_update_throttle(ea.room_id, ea)
                break
            else:
                if soonest_due_at is None or should_notify_at < soonest_due_at:
                    soonest_due_at = should_notify_at
                if self.timed_call is not None:
                    try:
                        self.timed_call.cancel()
                    except (AlreadyCalled, AlreadyCancelled):
                        pass
                    self.timed_call = None
        if soonest_due_at is not None:
            self.timed_call = self.hs.get_reactor().callLater(self.seconds_until(soonest_due_at), self.on_timer)

    async def save_last_stream_ordering_and_success(self, last_stream_ordering: int) -> None:
        self.last_stream_ordering = last_stream_ordering
        pusher_still_exists = await self.store.update_pusher_last_stream_ordering_and_success(self.app_id, self.email, self.user_id, last_stream_ordering, self.clock.time_msec())
        if not pusher_still_exists:
            self.on_stop()

    def seconds_until(self, ts_msec: int) -> float:
        if False:
            print('Hello World!')
        secs = (ts_msec - self.clock.time_msec()) / 1000
        return max(secs, 0)

    def get_room_throttle_ms(self, room_id: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        if room_id in self.throttle_params:
            return self.throttle_params[room_id].throttle_ms
        else:
            return 0

    def get_room_last_sent_ts(self, room_id: str) -> int:
        if False:
            while True:
                i = 10
        if room_id in self.throttle_params:
            return self.throttle_params[room_id].last_sent_ts
        else:
            return 0

    def room_ready_to_notify_at(self, room_id: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines whether throttling should prevent us from sending an email\n        for the given room\n\n        Returns:\n            The timestamp when we are next allowed to send an email notif\n            for this room\n        '
        last_sent_ts = self.get_room_last_sent_ts(room_id)
        throttle_ms = self.get_room_throttle_ms(room_id)
        may_send_at = last_sent_ts + throttle_ms
        return may_send_at

    async def sent_notif_update_throttle(self, room_id: str, notified_push_action: EmailPushAction) -> None:
        time_of_previous_notifs = await self.store.get_time_of_last_push_action_before(notified_push_action.stream_ordering)
        time_of_this_notifs = notified_push_action.received_ts
        if time_of_previous_notifs is not None and time_of_this_notifs is not None:
            gap = time_of_this_notifs - time_of_previous_notifs
        else:
            gap = 0
        current_throttle_ms = self.get_room_throttle_ms(room_id)
        if gap > THROTTLE_RESET_AFTER_MS:
            new_throttle_ms = THROTTLE_START_MS
        elif current_throttle_ms == 0:
            new_throttle_ms = THROTTLE_START_MS
        else:
            new_throttle_ms = min(current_throttle_ms * THROTTLE_MULTIPLIER, THROTTLE_MAX_MS)
        self.throttle_params[room_id] = ThrottleParams(self.clock.time_msec(), new_throttle_ms)
        assert self.pusher_id is not None
        await self.store.set_throttle_params(self.pusher_id, room_id, self.throttle_params[room_id])

    async def send_notification(self, push_actions: List[EmailPushAction], reason: EmailReason) -> None:
        logger.info('Sending notif email for user %r', self.user_id)
        await self.mailer.send_notification_mail(self.app_id, self.user_id, self.email, push_actions, reason)