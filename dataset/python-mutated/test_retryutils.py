from unittest import mock
from synapse.notifier import Notifier
from synapse.replication.tcp.handler import ReplicationCommandHandler
from synapse.util.retryutils import NotRetryingDestination, get_retry_limiter
from tests.unittest import HomeserverTestCase

class RetryLimiterTestCase(HomeserverTestCase):

    def test_new_destination(self) -> None:
        if False:
            i = 10
            return i + 15
        'A happy-path case with a new destination and a successful operation'
        store = self.hs.get_datastores().main
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store))
        self.pump(1)
        with limiter:
            pass
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNone(new_timings)

    def test_limiter(self) -> None:
        if False:
            i = 10
            return i + 15
        'General test case which walks through the process of a failing request'
        store = self.hs.get_datastores().main
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store))
        min_retry_interval_ms = self.hs.config.federation.destination_min_retry_interval_ms
        retry_multiplier = self.hs.config.federation.destination_retry_multiplier
        self.pump(1)
        try:
            with limiter:
                self.pump(1)
                failure_ts = self.clock.time_msec()
                raise AssertionError('argh')
        except AssertionError:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        assert new_timings is not None
        self.assertEqual(new_timings.failure_ts, failure_ts)
        self.assertEqual(new_timings.retry_last_ts, failure_ts)
        self.assertEqual(new_timings.retry_interval, min_retry_interval_ms)
        self.get_failure(get_retry_limiter('test_dest', self.clock, store), NotRetryingDestination)
        self.pump(min_retry_interval_ms)
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store))
        self.pump(1)
        try:
            with limiter:
                self.pump(1)
                retry_ts = self.clock.time_msec()
                raise AssertionError('argh')
        except AssertionError:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        assert new_timings is not None
        self.assertEqual(new_timings.failure_ts, failure_ts)
        self.assertEqual(new_timings.retry_last_ts, retry_ts)
        self.assertGreaterEqual(new_timings.retry_interval, min_retry_interval_ms * retry_multiplier * 0.5)
        self.assertLessEqual(new_timings.retry_interval, min_retry_interval_ms * retry_multiplier * 2.0)
        self.reactor.advance(min_retry_interval_ms * retry_multiplier * 2.0)
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store))
        self.pump(1)
        with limiter:
            self.pump(1)
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNone(new_timings)

    def test_notifier_replication(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure the notifier/replication client is called only when expected.'
        store = self.hs.get_datastores().main
        notifier = mock.Mock(spec=Notifier)
        replication_client = mock.Mock(spec=ReplicationCommandHandler)
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store, notifier=notifier, replication_client=replication_client))
        self.pump(1)
        with limiter:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNone(new_timings)
        notifier.notify_remote_server_up.assert_not_called()
        replication_client.send_remote_server_up.assert_not_called()
        self.pump(1)
        try:
            with limiter:
                raise AssertionError('argh')
        except AssertionError:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNotNone(new_timings)
        notifier.notify_remote_server_up.assert_not_called()
        replication_client.send_remote_server_up.assert_not_called()
        self.pump(1)
        try:
            with limiter:
                raise AssertionError('argh')
        except AssertionError:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNotNone(new_timings)
        notifier.notify_remote_server_up.assert_not_called()
        replication_client.send_remote_server_up.assert_not_called()
        self.pump(1)
        with limiter:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        self.assertIsNone(new_timings)
        notifier.notify_remote_server_up.assert_called_once_with('test_dest')
        replication_client.send_remote_server_up.assert_called_once_with('test_dest')

    def test_max_retry_interval(self) -> None:
        if False:
            while True:
                i = 10
        'Test that `destination_max_retry_interval` setting works as expected'
        store = self.hs.get_datastores().main
        destination_max_retry_interval_ms = self.hs.config.federation.destination_max_retry_interval_ms
        self.get_success(get_retry_limiter('test_dest', self.clock, store))
        self.pump(1)
        failure_ts = self.clock.time_msec()
        self.get_success(store.set_destination_retry_timings('test_dest', failure_ts=failure_ts, retry_last_ts=failure_ts, retry_interval=destination_max_retry_interval_ms))
        self.get_failure(get_retry_limiter('test_dest', self.clock, store), NotRetryingDestination)
        self.reactor.advance(destination_max_retry_interval_ms / 1000 + 1)
        limiter = self.get_success(get_retry_limiter('test_dest', self.clock, store))
        self.pump(1)
        try:
            with limiter:
                self.pump(1)
                raise AssertionError('argh')
        except AssertionError:
            pass
        self.pump()
        new_timings = self.get_success(store.get_destination_retry_timings('test_dest'))
        assert new_timings is not None
        self.assertEqual(new_timings.retry_interval, destination_max_retry_interval_ms)
        self.get_failure(get_retry_limiter('test_dest', self.clock, store), NotRetryingDestination)