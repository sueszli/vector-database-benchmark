from __future__ import annotations
import logging
import time
from unittest import mock
from unittest.mock import ANY
import pytest
from opentelemetry.metrics import MeterProvider
from airflow.exceptions import InvalidStatsNameException
from airflow.metrics.otel_logger import OTEL_NAME_MAX_LENGTH, UP_DOWN_COUNTERS, MetricsMap, SafeOtelLogger, _generate_key_name, _is_up_down_counter, full_name
from airflow.metrics.validators import BACK_COMPAT_METRIC_NAMES, MetricNameLengthExemptionWarning
INVALID_STAT_NAME_CASES = [(None, 'can not be None'), (42, 'is not a string'), ('X' * OTEL_NAME_MAX_LENGTH, 'too long'), ('test/$tats', 'contains invalid characters')]

@pytest.fixture
def name():
    if False:
        print('Hello World!')
    return 'test_stats_run'

class TestOtelMetrics:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.meter = mock.Mock(MeterProvider)
        self.stats = SafeOtelLogger(otel_provider=self.meter)
        self.map = self.stats.metrics_map.map
        self.logger = logging.getLogger(__name__)

    def test_is_up_down_counter_positive(self):
        if False:
            while True:
                i = 10
        udc = next(iter(UP_DOWN_COUNTERS))
        assert _is_up_down_counter(udc)

    def test_is_up_down_counter_negative(self):
        if False:
            while True:
                i = 10
        assert not _is_up_down_counter('this_is_not_a_udc')

    def test_exemption_list_has_not_grown(self):
        if False:
            return 10
        assert len(BACK_COMPAT_METRIC_NAMES) <= 26, "This test exists solely to ensure that nobody is adding names to the exemption list. There are 26 names which are potentially too long for OTel and that number should only ever go down as these names are deprecated.  If this test is failing, please adjust your new stat's name; do not add as exemption without a very good reason."

    @pytest.mark.parametrize('invalid_stat_combo', [*[pytest.param(('prefix', name), id=f'Stat name {msg}.') for (name, msg) in INVALID_STAT_NAME_CASES], *[pytest.param((prefix, 'name'), id=f'Stat prefix {msg}.') for (prefix, msg) in INVALID_STAT_NAME_CASES]])
    def test_invalid_stat_names_are_caught(self, invalid_stat_combo):
        if False:
            while True:
                i = 10
        prefix = invalid_stat_combo[0]
        name = invalid_stat_combo[1]
        self.stats.prefix = prefix
        with pytest.raises(InvalidStatsNameException):
            self.stats.incr(name)
        self.meter.assert_not_called()

    def test_old_name_exception_works(self, caplog):
        if False:
            print('Hello World!')
        name = 'task_instance_created_OperatorNameWhichIsSuperLongAndExceedsTheOpenTelemetryCharacterLimit'
        assert len(name) > OTEL_NAME_MAX_LENGTH
        with pytest.warns(MetricNameLengthExemptionWarning):
            self.stats.incr(name)
        self.meter.get_meter().create_counter.assert_called_once_with(name=full_name(name)[:OTEL_NAME_MAX_LENGTH])

    def test_incr_new_metric(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.stats.incr(name)
        self.meter.get_meter().create_counter.assert_called_once_with(name=full_name(name))

    def test_incr_new_metric_with_tags(self, name):
        if False:
            print('Hello World!')
        tags = {'hello': 'world'}
        key = _generate_key_name(full_name(name), tags)
        self.stats.incr(name, tags=tags)
        self.meter.get_meter().create_counter.assert_called_once_with(name=full_name(name))
        self.map[key].add.assert_called_once_with(1, attributes=tags)

    def test_incr_existing_metric(self, name):
        if False:
            while True:
                i = 10
        self.stats.incr(name)
        self.stats.incr(name)
        assert self.map[full_name(name)].add.call_count == 2
        self.meter.get_meter().create_counter.assert_called_once_with(name=full_name(name))

    @mock.patch('random.random', side_effect=[0.1, 0.9])
    def test_incr_with_rate_limit_works(self, mock_random, name):
        if False:
            while True:
                i = 10
        self.stats.incr(name, rate=0.5)
        self.stats.incr(name, rate=0.5)
        with pytest.raises(ValueError):
            self.stats.incr(name, rate=-0.5)
        assert mock_random.call_count == 2
        assert self.map[full_name(name)].add.call_count == 1

    def test_decr_existing_metric(self, name):
        if False:
            for i in range(10):
                print('nop')
        expected_calls = [mock.call(1, attributes=None), mock.call(-1, attributes=None)]
        self.stats.incr(name)
        self.stats.decr(name)
        self.map[full_name(name)].add.assert_has_calls(expected_calls)
        assert self.map[full_name(name)].add.call_count == len(expected_calls)

    @mock.patch('random.random', side_effect=[0.1, 0.9])
    def test_decr_with_rate_limit_works(self, mock_random, name):
        if False:
            print('Hello World!')
        expected_calls = [mock.call(1, attributes=None), mock.call(-1, attributes=None)]
        self.stats.incr(name)
        self.stats.decr(name, rate=0.5)
        self.stats.decr(name, rate=0.5)
        with pytest.raises(ValueError):
            self.stats.decr(name, rate=-0.5)
        assert mock_random.call_count == 2
        self.map[full_name(name)].add.assert_has_calls(expected_calls)
        self.map[full_name(name)].add.call_count == 2

    def test_gauge_new_metric(self, name):
        if False:
            return 10
        self.stats.gauge(name, value=1)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        assert self.map[full_name(name)].value == 1

    def test_gauge_new_metric_with_tags(self, name):
        if False:
            while True:
                i = 10
        tags = {'hello': 'world'}
        key = _generate_key_name(full_name(name), tags)
        self.stats.gauge(name, value=1, tags=tags)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        self.map[key].attributes == tags

    def test_gauge_existing_metric(self, name):
        if False:
            i = 10
            return i + 15
        self.stats.gauge(name, value=1)
        self.stats.gauge(name, value=2)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        assert self.map[full_name(name)].value == 2

    def test_gauge_existing_metric_with_delta(self, name):
        if False:
            print('Hello World!')
        self.stats.gauge(name, value=1)
        self.stats.gauge(name, value=2, delta=True)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        assert self.map[full_name(name)].value == 3

    @mock.patch('random.random', side_effect=[0.1, 0.9])
    @mock.patch.object(MetricsMap, 'set_gauge_value')
    def test_gauge_with_rate_limit_works(self, mock_set_value, mock_random, name):
        if False:
            print('Hello World!')
        self.stats.gauge(name, value=1, rate=0.5)
        self.stats.gauge(name, value=1, rate=0.5)
        with pytest.raises(ValueError):
            self.stats.gauge(name, value=1, rate=-0.5)
        assert mock_random.call_count == 2
        assert mock_set_value.call_count == 1

    def test_gauge_value_is_correct(self, name):
        if False:
            while True:
                i = 10
        self.stats.gauge(name, value=1)
        assert self.map[full_name(name)].value == 1

    def test_timing_new_metric(self, name):
        if False:
            print('Hello World!')
        self.stats.timing(name, dt=123)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)

    def test_timing_new_metric_with_tags(self, name):
        if False:
            for i in range(10):
                print('nop')
        tags = {'hello': 'world'}
        key = _generate_key_name(full_name(name), tags)
        self.stats.timing(name, dt=1, tags=tags)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        self.map[key].attributes == tags

    def test_timing_existing_metric(self, name):
        if False:
            return 10
        self.stats.timing(name, dt=1)
        self.stats.timing(name, dt=2)
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)
        assert self.map[full_name(name)].value == 2

    @mock.patch.object(time, 'perf_counter', side_effect=[0.0, 3.14])
    def test_timer_with_name_returns_float_and_stores_value(self, mock_time, name):
        if False:
            i = 10
            return i + 15
        with self.stats.timer(name) as timer:
            pass
        assert isinstance(timer.duration, float)
        assert timer.duration == 3.14
        assert mock_time.call_count == 2
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)

    @mock.patch.object(time, 'perf_counter', side_effect=[0.0, 3.14])
    def test_timer_no_name_returns_float_but_does_not_store_value(self, mock_time, name):
        if False:
            return 10
        with self.stats.timer() as timer:
            pass
        assert isinstance(timer.duration, float)
        assert timer.duration == 3.14
        assert mock_time.call_count == 2
        self.meter.get_meter().create_observable_gauge.assert_not_called()

    @mock.patch.object(time, 'perf_counter', side_effect=[0.0, 3.14])
    def test_timer_start_and_stop_manually_send_false(self, mock_time, name):
        if False:
            print('Hello World!')
        timer = self.stats.timer(name)
        timer.start()
        timer.stop(send=False)
        assert isinstance(timer.duration, float)
        assert timer.duration == 3.14
        assert mock_time.call_count == 2
        self.meter.get_meter().create_observable_gauge.assert_not_called()

    @mock.patch.object(time, 'perf_counter', side_effect=[0.0, 3.14])
    def test_timer_start_and_stop_manually_send_true(self, mock_time, name):
        if False:
            print('Hello World!')
        timer = self.stats.timer(name)
        timer.start()
        timer.stop(send=True)
        assert isinstance(timer.duration, float)
        assert timer.duration == 3.14
        assert mock_time.call_count == 2
        self.meter.get_meter().create_observable_gauge.assert_called_once_with(name=full_name(name), callbacks=ANY)