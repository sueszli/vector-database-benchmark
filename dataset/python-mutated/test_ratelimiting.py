from synapse.api.ratelimiting import LimitExceededError, Ratelimiter
from synapse.appservice import ApplicationService
from synapse.config.ratelimiting import RatelimitSettings
from synapse.types import create_requester
from tests import unittest

class TestRatelimiter(unittest.HomeserverTestCase):

    def test_allowed_via_can_do_action(self) -> None:
        if False:
            while True:
                i = 10
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', _time_now_s=0))
        self.assertTrue(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', _time_now_s=5))
        self.assertFalse(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', _time_now_s=10))
        self.assertTrue(allowed)
        self.assertEqual(20.0, time_allowed)

    def test_allowed_appservice_ratelimited_via_can_requester_do_action(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        appservice = ApplicationService(token='fake_token', id='foo', rate_limited=True, sender='@as:example.com')
        as_requester = create_requester('@user:example.com', app_service=appservice)
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=0))
        self.assertTrue(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=5))
        self.assertFalse(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=10))
        self.assertTrue(allowed)
        self.assertEqual(20.0, time_allowed)

    def test_allowed_appservice_via_can_requester_do_action(self) -> None:
        if False:
            print('Hello World!')
        appservice = ApplicationService(token='fake_token', id='foo', rate_limited=False, sender='@as:example.com')
        as_requester = create_requester('@user:example.com', app_service=appservice)
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=0))
        self.assertTrue(allowed)
        self.assertEqual(-1, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=5))
        self.assertTrue(allowed)
        self.assertEqual(-1, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(as_requester, _time_now_s=10))
        self.assertTrue(allowed)
        self.assertEqual(-1, time_allowed)

    def test_allowed_via_ratelimit(self) -> None:
        if False:
            while True:
                i = 10
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        self.get_success_or_raise(limiter.ratelimit(None, key='test_id', _time_now_s=0))
        with self.assertRaises(LimitExceededError) as context:
            self.get_success_or_raise(limiter.ratelimit(None, key='test_id', _time_now_s=5))
        self.assertEqual(context.exception.retry_after_ms, 5000)
        self.get_success_or_raise(limiter.ratelimit(None, key='test_id', _time_now_s=10))

    def test_allowed_via_can_do_action_and_overriding_parameters(self) -> None:
        if False:
            print('Hello World!')
        'Test that we can override options of can_do_action that would otherwise fail\n        an action\n        '
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, ('test_id',), _time_now_s=0))
        self.assertTrue(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, ('test_id',), _time_now_s=1))
        self.assertFalse(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, ('test_id',), _time_now_s=1, rate_hz=10.0))
        self.assertTrue(allowed)
        self.assertEqual(1.1, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, ('test_id',), _time_now_s=1, burst_count=10))
        self.assertTrue(allowed)
        self.assertEqual(1.0, time_allowed)

    def test_allowed_via_ratelimit_and_overriding_parameters(self) -> None:
        if False:
            print('Hello World!')
        'Test that we can override options of the ratelimit method that would otherwise\n        fail an action\n        '
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        self.get_success_or_raise(limiter.ratelimit(None, key=('test_id',), _time_now_s=0))
        with self.assertRaises(LimitExceededError) as context:
            self.get_success_or_raise(limiter.ratelimit(None, key=('test_id',), _time_now_s=1))
        self.assertEqual(context.exception.retry_after_ms, 9000)
        self.get_success_or_raise(limiter.ratelimit(None, key=('test_id',), _time_now_s=1, rate_hz=10.0))
        self.get_success_or_raise(limiter.ratelimit(None, key=('test_id',), _time_now_s=1, burst_count=10))

    def test_pruning(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=1))
        self.get_success_or_raise(limiter.can_do_action(None, key='test_id_1', _time_now_s=0))
        self.assertIn('test_id_1', limiter.actions)
        self.get_success_or_raise(limiter.can_do_action(None, key='test_id_2', _time_now_s=10))
        self.assertNotIn('test_id_1', limiter.actions)

    def test_db_user_override(self) -> None:
        if False:
            i = 10
            return i + 15
        "Test that users that have ratelimiting disabled in the DB aren't\n        ratelimited.\n        "
        store = self.hs.get_datastores().main
        user_id = '@user:test'
        requester = create_requester(user_id)
        self.get_success(store.db_pool.simple_insert(table='ratelimit_override', values={'user_id': user_id, 'messages_per_second': None, 'burst_count': None}, desc='test_db_user_override'))
        limiter = Ratelimiter(store=store, clock=self.clock, cfg=RatelimitSettings('', per_second=0.1, burst_count=1))
        for _ in range(20):
            self.get_success_or_raise(limiter.ratelimit(requester, _time_now_s=0))

    def test_multiple_actions(self) -> None:
        if False:
            i = 10
            return i + 15
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings(key='', per_second=0.1, burst_count=3))
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', n_actions=4, _time_now_s=0))
        self.assertFalse(allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', n_actions=3, _time_now_s=0))
        self.assertTrue(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', n_actions=1, _time_now_s=0))
        self.assertFalse(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', update=False, n_actions=1, _time_now_s=10))
        self.assertTrue(allowed)
        self.assertEqual(20.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', n_actions=2, _time_now_s=10))
        self.assertFalse(allowed)
        self.assertEqual(10.0, time_allowed)
        (allowed, time_allowed) = self.get_success_or_raise(limiter.can_do_action(None, key='test_id', n_actions=2, _time_now_s=20))
        self.assertTrue(allowed)
        self.assertEqual(30.0, time_allowed)

    def test_rate_limit_burst_only_given_once(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Regression test against a bug that meant that you could build up\n        extra tokens by timing requests.\n        '
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings('', per_second=0.1, burst_count=3))

        def consume_at(time: float) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=time))
            return success
        self.assertTrue(consume_at(0.0))
        self.assertTrue(consume_at(0.1))
        self.assertTrue(consume_at(0.2))
        self.assertTrue(consume_at(10.1))
        self.assertFalse(consume_at(11.1))

    def test_record_action_which_doesnt_fill_bucket(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings('', per_second=0.1, burst_count=3))
        limiter.record_action(requester=None, key='a', n_actions=2, _time_now_s=0.0)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=0.0))
        self.assertTrue(success)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=0.0))
        self.assertFalse(success)

    def test_record_action_which_fills_bucket(self) -> None:
        if False:
            print('Hello World!')
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings('', per_second=0.1, burst_count=3))
        limiter.record_action(requester=None, key='a', n_actions=3, _time_now_s=0.0)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=0.0))
        self.assertFalse(success)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=10.0))
        self.assertTrue(success)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=10.0))
        self.assertFalse(success)

    def test_record_action_which_overfills_bucket(self) -> None:
        if False:
            i = 10
            return i + 15
        limiter = Ratelimiter(store=self.hs.get_datastores().main, clock=self.clock, cfg=RatelimitSettings('', per_second=0.1, burst_count=3))
        limiter.record_action(requester=None, key='a', n_actions=4, _time_now_s=0.0)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=0.0))
        self.assertFalse(success)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=10.0))
        self.assertFalse(success)
        (success, _) = self.get_success_or_raise(limiter.can_do_action(requester=None, key='a', _time_now_s=20.0))
        self.assertTrue(success)