import resource
from unittest import mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.app.phone_stats_home import phone_stats_home
from synapse.rest import admin
from synapse.rest.client import login, sync
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class PhoneHomeStatsTestCase(HomeserverTestCase):

    def test_performance_frozen_clock(self) -> None:
        if False:
            print('Hello World!')
        "\n        If time doesn't move, don't error out.\n        "
        past_stats = [(int(self.hs.get_clock().time()), resource.getrusage(resource.RUSAGE_SELF))]
        stats: JsonDict = {}
        self.get_success(phone_stats_home(self.hs, stats, past_stats))
        self.assertEqual(stats['cpu_average'], 0)

    def test_performance_100(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        1 second of usage over 1 second is 100% CPU usage.\n        '
        real_res = resource.getrusage(resource.RUSAGE_SELF)
        old_resource = mock.Mock(spec=real_res)
        old_resource.ru_utime = real_res.ru_utime - 1
        old_resource.ru_stime = real_res.ru_stime
        old_resource.ru_maxrss = real_res.ru_maxrss
        past_stats = [(self.hs.get_clock().time(), old_resource)]
        stats: JsonDict = {}
        self.reactor.advance(1)
        self.get_success(phone_stats_home(self.hs, stats, past_stats))
        self.assertApproximates(stats['cpu_average'], 100, tolerance=2.5)

class CommonMetricsTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, sync.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.metrics_manager = hs.get_common_usage_metrics_manager()
        self.get_success(self.metrics_manager.setup())

    def test_dau(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the daily active users count is correctly updated.'
        self._assert_metric_value('daily_active_users', 0)
        self.register_user('user', 'password')
        tok = self.login('user', 'password')
        self.make_request('GET', '/sync', access_token=tok)
        self.pump(1)
        self._assert_metric_value('daily_active_users', 1)

    def _assert_metric_value(self, metric_name: str, expected: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Compare the given value to the current value of the common usage metric with\n        the given name.\n\n        Args:\n            metric_name: The metric to look up.\n            expected: Expected value for this metric.\n        '
        metrics = self.get_success(self.metrics_manager.get_metrics())
        value = getattr(metrics, metric_name)
        self.assertEqual(value, expected)