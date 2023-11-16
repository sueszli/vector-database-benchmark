from __future__ import annotations
from unittest import mock
from airflow.executors.sequential_executor import SequentialExecutor

class TestSequentialExecutor:

    def test_supports_pickling(self):
        if False:
            for i in range(10):
                print('nop')
        assert not SequentialExecutor.supports_pickling

    def test_supports_sentry(self):
        if False:
            i = 10
            return i + 15
        assert not SequentialExecutor.supports_sentry

    def test_is_local_default_value(self):
        if False:
            for i in range(10):
                print('nop')
        assert SequentialExecutor.is_local

    def test_is_production_default_value(self):
        if False:
            print('Hello World!')
        assert not SequentialExecutor.is_production

    def test_serve_logs_default_value(self):
        if False:
            for i in range(10):
                print('nop')
        assert SequentialExecutor.serve_logs

    def test_is_single_threaded_default_value(self):
        if False:
            i = 10
            return i + 15
        assert SequentialExecutor.is_single_threaded

    @mock.patch('airflow.executors.sequential_executor.SequentialExecutor.sync')
    @mock.patch('airflow.executors.base_executor.BaseExecutor.trigger_tasks')
    @mock.patch('airflow.executors.base_executor.Stats.gauge')
    def test_gauge_executor_metrics(self, mock_stats_gauge, mock_trigger_tasks, mock_sync):
        if False:
            while True:
                i = 10
        executor = SequentialExecutor()
        executor.heartbeat()
        calls = [mock.call('executor.open_slots', value=mock.ANY, tags={'status': 'open', 'name': 'SequentialExecutor'}), mock.call('executor.queued_tasks', value=mock.ANY, tags={'status': 'queued', 'name': 'SequentialExecutor'}), mock.call('executor.running_tasks', value=mock.ANY, tags={'status': 'running', 'name': 'SequentialExecutor'})]
        mock_stats_gauge.assert_has_calls(calls)