"""Unit tests for Superset"""
from unittest import TestCase
from unittest.mock import Mock, patch
from superset.stats_logger import StatsdStatsLogger

class TestStatsdStatsLogger(TestCase):

    def verify_client_calls(self, logger, client):
        if False:
            for i in range(10):
                print('nop')
        logger.incr('foo1')
        client.incr.assert_called_once()
        client.incr.assert_called_with('foo1')
        logger.decr('foo2')
        client.decr.assert_called_once()
        client.decr.assert_called_with('foo2')
        logger.gauge('foo3', 2.21)
        client.gauge.assert_called_once()
        client.gauge.assert_called_with('foo3', 2.21)
        logger.timing('foo4', 1.234)
        client.timing.assert_called_once()
        client.timing.assert_called_with('foo4', 1.234)

    def test_init_with_statsd_client(self):
        if False:
            return 10
        client = Mock()
        stats_logger = StatsdStatsLogger(statsd_client=client)
        self.verify_client_calls(stats_logger, client)

    def test_init_with_params(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('superset.stats_logger.StatsClient') as MockStatsdClient:
            mock_client = MockStatsdClient.return_value
            stats_logger = StatsdStatsLogger()
            self.verify_client_calls(stats_logger, mock_client)