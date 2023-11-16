from typing import Optional, Tuple
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import MAIN_TIMELINE, RelationTypes
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.databases.main.event_push_actions import NotifCounts
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class EventPushActionsStoreTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main
        persist_events_store = hs.get_datastores().persist_events
        assert persist_events_store is not None
        self.persist_events_store = persist_events_store

    def _create_users_and_room(self) -> Tuple[str, str, str, str, str]:
        if False:
            print('Hello World!')
        '\n        Creates two users and a shared room.\n\n        Returns:\n            Tuple of (user 1 ID, user 1 token, user 2 ID, user 2 token, room ID).\n        '
        user_id = self.register_user('user1235', 'pass')
        token = self.login('user1235', 'pass')
        other_id = self.register_user('other', 'pass')
        other_token = self.login('other', 'pass')
        room_id = self.helper.create_room_as(user_id, tok=token)
        self.helper.join(room_id, other_id, tok=other_token)
        return (user_id, token, other_id, other_token, room_id)

    def test_get_unread_push_actions_for_user_in_range(self) -> None:
        if False:
            return 10
        'Test getting unread push actions for HTTP and email pushers.'
        (user_id, token, _, other_token, room_id) = self._create_users_and_room()
        first_event_id = self.helper.send_event(room_id, type='m.room.message', content={'msgtype': 'm.text', 'body': 'msg'}, tok=other_token)['event_id']
        second_event_id = self.helper.send_event(room_id, type='m.room.message', content={'msgtype': 'm.text', 'body': user_id, 'm.relates_to': {'rel_type': RelationTypes.THREAD, 'event_id': first_event_id}}, tok=other_token)['event_id']
        http_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_http(user_id, 0, 1000, 20))
        self.assertEqual(2, len(http_actions))
        email_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_email(user_id, 0, 1000, 20))
        self.assertEqual(2, len(email_actions))
        self.get_success(self.store.insert_receipt(room_id, 'm.read', user_id=user_id, event_ids=[first_event_id], thread_id=None, data={}))
        http_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_http(user_id, 0, 1000, 20))
        self.assertEqual(1, len(http_actions))
        email_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_email(user_id, 0, 1000, 20))
        self.assertEqual(1, len(email_actions))
        self.get_success(self.store.insert_receipt(room_id, 'm.read', user_id=user_id, event_ids=[second_event_id], thread_id=first_event_id, data={}))
        http_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_http(user_id, 0, 1000, 20))
        self.assertEqual([], http_actions)
        email_actions = self.get_success(self.store.get_unread_push_actions_for_user_in_range_for_email(user_id, 0, 1000, 20))
        self.assertEqual([], email_actions)

    def test_count_aggregation(self) -> None:
        if False:
            print('Hello World!')
        (user_id, token, _, other_token, room_id) = self._create_users_and_room()
        last_event_id = ''

        def _assert_counts(notif_count: int, highlight_count: int) -> None:
            if False:
                print('Hello World!')
            counts = self.get_success(self.store.db_pool.runInteraction('get-unread-counts', self.store._get_unread_counts_by_receipt_txn, room_id, user_id))
            self.assertEqual(counts.main_timeline, NotifCounts(notify_count=notif_count, unread_count=0, highlight_count=highlight_count))
            self.assertEqual(counts.threads, {})
            aggregate_counts = self.get_success(self.store.db_pool.runInteraction('get-aggregate-unread-counts', self.store._get_unread_counts_by_room_for_user_txn, user_id))
            self.assertEqual(aggregate_counts[room_id], notif_count)

        def _create_event(highlight: bool=False) -> str:
            if False:
                while True:
                    i = 10
            result = self.helper.send_event(room_id, type='m.room.message', content={'msgtype': 'm.text', 'body': user_id if highlight else 'msg'}, tok=other_token)
            nonlocal last_event_id
            last_event_id = result['event_id']
            return last_event_id

        def _rotate() -> None:
            if False:
                i = 10
                return i + 15
            self.get_success(self.store._rotate_notifs())

        def _mark_read(event_id: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.get_success(self.store.insert_receipt(room_id, 'm.read', user_id=user_id, event_ids=[event_id], thread_id=None, data={}))
        _assert_counts(0, 0)
        _create_event()
        _assert_counts(1, 0)
        _rotate()
        _assert_counts(1, 0)
        event_id = _create_event()
        _assert_counts(2, 0)
        _rotate()
        _assert_counts(2, 0)
        _create_event()
        _mark_read(event_id)
        _assert_counts(1, 0)
        _mark_read(last_event_id)
        _assert_counts(0, 0)
        _create_event()
        _assert_counts(1, 0)
        _rotate()
        _assert_counts(1, 0)
        self.pump(60 * 60 * 24)
        self.get_success(self.store._remove_old_push_actions_that_have_rotated())
        result = self.get_success(self.store.db_pool.simple_select_list(table='event_push_actions', keyvalues={'1': 1}, retcols=('event_id',), desc=''))
        self.assertEqual(result, [])
        _assert_counts(1, 0)
        _mark_read(last_event_id)
        _assert_counts(0, 0)
        event_id = _create_event(True)
        _assert_counts(1, 1)
        _rotate()
        _assert_counts(1, 1)
        _create_event()
        _rotate()
        _assert_counts(2, 1)
        _mark_read(event_id)
        _assert_counts(1, 0)
        _mark_read(last_event_id)
        _assert_counts(0, 0)
        _create_event(True)
        _assert_counts(1, 1)
        _mark_read(last_event_id)
        _assert_counts(0, 0)
        _rotate()
        _assert_counts(0, 0)

    def test_count_aggregation_threads(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        This is essentially the same test as test_count_aggregation, but adds\n        events to the main timeline and to a thread.\n        '
        (user_id, token, _, other_token, room_id) = self._create_users_and_room()
        thread_id: str
        last_event_id = ''

        def _assert_counts(notif_count: int, highlight_count: int, thread_notif_count: int, thread_highlight_count: int) -> None:
            if False:
                i = 10
                return i + 15
            counts = self.get_success(self.store.db_pool.runInteraction('get-unread-counts', self.store._get_unread_counts_by_receipt_txn, room_id, user_id))
            self.assertEqual(counts.main_timeline, NotifCounts(notify_count=notif_count, unread_count=0, highlight_count=highlight_count))
            if thread_notif_count or thread_highlight_count:
                self.assertEqual(counts.threads, {thread_id: NotifCounts(notify_count=thread_notif_count, unread_count=0, highlight_count=thread_highlight_count)})
            else:
                self.assertEqual(counts.threads, {})
            aggregate_counts = self.get_success(self.store.db_pool.runInteraction('get-aggregate-unread-counts', self.store._get_unread_counts_by_room_for_user_txn, user_id))
            self.assertEqual(aggregate_counts[room_id], notif_count + thread_notif_count)

        def _create_event(highlight: bool=False, thread_id: Optional[str]=None) -> str:
            if False:
                for i in range(10):
                    print('nop')
            content: JsonDict = {'msgtype': 'm.text', 'body': user_id if highlight else 'msg'}
            if thread_id:
                content['m.relates_to'] = {'rel_type': 'm.thread', 'event_id': thread_id}
            result = self.helper.send_event(room_id, type='m.room.message', content=content, tok=other_token)
            nonlocal last_event_id
            last_event_id = result['event_id']
            return last_event_id

        def _rotate() -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.get_success(self.store._rotate_notifs())

        def _mark_read(event_id: str, thread_id: str=MAIN_TIMELINE) -> None:
            if False:
                while True:
                    i = 10
            self.get_success(self.store.insert_receipt(room_id, 'm.read', user_id=user_id, event_ids=[event_id], thread_id=thread_id, data={}))
        _assert_counts(0, 0, 0, 0)
        thread_id = _create_event()
        _assert_counts(1, 0, 0, 0)
        _rotate()
        _assert_counts(1, 0, 0, 0)
        _create_event(thread_id=thread_id)
        _assert_counts(1, 0, 1, 0)
        _rotate()
        _assert_counts(1, 0, 1, 0)
        _create_event()
        _assert_counts(2, 0, 1, 0)
        _rotate()
        _assert_counts(2, 0, 1, 0)
        event_id = _create_event(thread_id=thread_id)
        _assert_counts(2, 0, 2, 0)
        _rotate()
        _assert_counts(2, 0, 2, 0)
        _create_event()
        _create_event(thread_id=thread_id)
        _mark_read(event_id)
        _assert_counts(1, 0, 3, 0)
        _mark_read(event_id, thread_id)
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _create_event()
        _create_event(thread_id=thread_id)
        _assert_counts(1, 0, 1, 0)
        _rotate()
        _assert_counts(1, 0, 1, 0)
        self.get_success(self.store._remove_old_push_actions_that_have_rotated())
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _create_event(True)
        _assert_counts(1, 1, 0, 0)
        _rotate()
        _assert_counts(1, 1, 0, 0)
        event_id = _create_event(True, thread_id)
        _assert_counts(1, 1, 1, 1)
        _rotate()
        _assert_counts(1, 1, 1, 1)
        _create_event()
        _rotate()
        _assert_counts(2, 1, 1, 1)
        _create_event(thread_id=thread_id)
        _rotate()
        _assert_counts(2, 1, 2, 1)
        _mark_read(event_id)
        _assert_counts(1, 0, 2, 1)
        _mark_read(event_id, thread_id)
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id)
        _assert_counts(0, 0, 1, 0)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _create_event(True)
        _create_event(True, thread_id)
        _assert_counts(1, 1, 1, 1)
        _mark_read(last_event_id)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _rotate()
        _assert_counts(0, 0, 0, 0)

    def test_count_aggregation_mixed(self) -> None:
        if False:
            return 10
        '\n        This is essentially the same test as test_count_aggregation_threads, but\n        sends both unthreaded and threaded receipts.\n        '
        (user_id, token, _, other_token, room_id) = self._create_users_and_room()
        thread_id: str
        last_event_id = ''

        def _assert_counts(notif_count: int, highlight_count: int, thread_notif_count: int, thread_highlight_count: int) -> None:
            if False:
                i = 10
                return i + 15
            counts = self.get_success(self.store.db_pool.runInteraction('get-unread-counts', self.store._get_unread_counts_by_receipt_txn, room_id, user_id))
            self.assertEqual(counts.main_timeline, NotifCounts(notify_count=notif_count, unread_count=0, highlight_count=highlight_count))
            if thread_notif_count or thread_highlight_count:
                self.assertEqual(counts.threads, {thread_id: NotifCounts(notify_count=thread_notif_count, unread_count=0, highlight_count=thread_highlight_count)})
            else:
                self.assertEqual(counts.threads, {})
            aggregate_counts = self.get_success(self.store.db_pool.runInteraction('get-aggregate-unread-counts', self.store._get_unread_counts_by_room_for_user_txn, user_id))
            self.assertEqual(aggregate_counts[room_id], notif_count + thread_notif_count)

        def _create_event(highlight: bool=False, thread_id: Optional[str]=None) -> str:
            if False:
                print('Hello World!')
            content: JsonDict = {'msgtype': 'm.text', 'body': user_id if highlight else 'msg'}
            if thread_id:
                content['m.relates_to'] = {'rel_type': 'm.thread', 'event_id': thread_id}
            result = self.helper.send_event(room_id, type='m.room.message', content=content, tok=other_token)
            nonlocal last_event_id
            last_event_id = result['event_id']
            return last_event_id

        def _rotate() -> None:
            if False:
                i = 10
                return i + 15
            self.get_success(self.store._rotate_notifs())

        def _mark_read(event_id: str, thread_id: Optional[str]=None) -> None:
            if False:
                return 10
            self.get_success(self.store.insert_receipt(room_id, 'm.read', user_id=user_id, event_ids=[event_id], thread_id=thread_id, data={}))
        _assert_counts(0, 0, 0, 0)
        thread_id = _create_event()
        _assert_counts(1, 0, 0, 0)
        _rotate()
        _assert_counts(1, 0, 0, 0)
        _create_event(thread_id=thread_id)
        _assert_counts(1, 0, 1, 0)
        _rotate()
        _assert_counts(1, 0, 1, 0)
        _create_event()
        _assert_counts(2, 0, 1, 0)
        _rotate()
        _assert_counts(2, 0, 1, 0)
        event_id = _create_event(thread_id=thread_id)
        _assert_counts(2, 0, 2, 0)
        _rotate()
        _assert_counts(2, 0, 2, 0)
        _create_event()
        _create_event(thread_id=thread_id)
        _mark_read(event_id)
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id, MAIN_TIMELINE)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _create_event()
        _create_event(thread_id=thread_id)
        _assert_counts(1, 0, 1, 0)
        _rotate()
        _assert_counts(1, 0, 1, 0)
        self.get_success(self.store._remove_old_push_actions_that_have_rotated())
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id)
        _assert_counts(0, 0, 0, 0)
        _create_event(True)
        _assert_counts(1, 1, 0, 0)
        _rotate()
        _assert_counts(1, 1, 0, 0)
        event_id = _create_event(True, thread_id)
        _assert_counts(1, 1, 1, 1)
        _rotate()
        _assert_counts(1, 1, 1, 1)
        _create_event()
        _rotate()
        _assert_counts(2, 1, 1, 1)
        _create_event(thread_id=thread_id)
        _rotate()
        _assert_counts(2, 1, 2, 1)
        _mark_read(event_id)
        _assert_counts(1, 0, 1, 0)
        _mark_read(event_id, MAIN_TIMELINE)
        _assert_counts(1, 0, 1, 0)
        _mark_read(last_event_id, MAIN_TIMELINE)
        _assert_counts(0, 0, 1, 0)
        _mark_read(last_event_id, thread_id)
        _assert_counts(0, 0, 0, 0)
        _create_event(True)
        _create_event(True, thread_id)
        _assert_counts(1, 1, 1, 1)
        _mark_read(last_event_id)
        _assert_counts(0, 0, 0, 0)
        _rotate()
        _assert_counts(0, 0, 0, 0)

    def test_recursive_thread(self) -> None:
        if False:
            print('Hello World!')
        '\n        Events related to events in a thread should still be considered part of\n        that thread.\n        '
        user_id = self.register_user('user1235', 'pass')
        token = self.login('user1235', 'pass')
        other_id = self.register_user('other', 'pass')
        other_token = self.login('other', 'pass')
        room_id = self.helper.create_room_as(user_id, tok=token)
        self.helper.join(room_id, other_id, tok=other_token)
        self.get_success(self.store.add_push_rule(user_id, 'related_events', priority_class=5, conditions=[{'kind': 'event_match', 'key': 'type', 'pattern': 'm.reaction'}], actions=['notify']))

        def _create_event(type: str, content: JsonDict) -> str:
            if False:
                for i in range(10):
                    print('nop')
            result = self.helper.send_event(room_id, type=type, content=content, tok=other_token)
            return result['event_id']

        def _assert_counts(notif_count: int, thread_notif_count: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            counts = self.get_success(self.store.db_pool.runInteraction('get-unread-counts', self.store._get_unread_counts_by_receipt_txn, room_id, user_id))
            self.assertEqual(counts.main_timeline, NotifCounts(notify_count=notif_count, unread_count=0, highlight_count=0))
            if thread_notif_count:
                self.assertEqual(counts.threads, {thread_id: NotifCounts(notify_count=thread_notif_count, unread_count=0, highlight_count=0)})
            else:
                self.assertEqual(counts.threads, {})
        thread_id = _create_event('m.room.message', {'msgtype': 'm.text', 'body': 'msg'})
        _assert_counts(1, 0)
        reply_id = _create_event('m.room.message', {'msgtype': 'm.text', 'body': 'msg', 'm.relates_to': {'rel_type': 'm.thread', 'event_id': thread_id}})
        _assert_counts(1, 1)
        _create_event(type='m.reaction', content={'m.relates_to': {'rel_type': 'm.annotation', 'event_id': reply_id, 'key': 'A'}})
        _assert_counts(1, 2)

    def test_find_first_stream_ordering_after_ts(self) -> None:
        if False:
            i = 10
            return i + 15

        def add_event(so: int, ts: int) -> None:
            if False:
                while True:
                    i = 10
            self.get_success(self.store.db_pool.simple_insert('events', {'stream_ordering': so, 'received_ts': ts, 'event_id': 'event%i' % so, 'type': '', 'room_id': '', 'content': '', 'processed': True, 'outlier': False, 'topological_ordering': 0, 'depth': 0}))
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(11))
        self.assertEqual(r, 0)
        add_event(2, 10)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(9))
        self.assertEqual(r, 2)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(10))
        self.assertEqual(r, 2)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(11))
        self.assertEqual(r, 3)
        for (stream_ordering, ts) in ((3, 110), (4, 120), (5, 120), (10, 130), (20, 140)):
            add_event(stream_ordering, ts)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(110))
        self.assertEqual(r, 3, 'First event after 110ms should be 3, was %i' % r)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(120))
        self.assertEqual(r, 4, 'First event after 120ms should be 4, was %i' % r)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(129))
        self.assertEqual(r, 10, 'First event after 129ms should be 10, was %i' % r)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(140))
        self.assertEqual(r, 20, 'First event after 14ms should be 20, was %i' % r)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(160))
        self.assertEqual(r, 21)
        add_event(0, 5)
        r = self.get_success(self.store.find_first_stream_ordering_after_ts(1))
        self.assertEqual(r, 0)