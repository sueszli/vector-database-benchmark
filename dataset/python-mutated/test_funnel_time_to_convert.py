from posthog.constants import INSIGHT_FUNNELS, TRENDS_LINEAR, FunnelOrderType
from posthog.models.filters import Filter
from posthog.queries.funnels.funnel_time_to_convert import ClickhouseFunnelTimeToConvert
from posthog.test.base import APIBaseTest, ClickhouseTestMixin, _create_event, _create_person, snapshot_clickhouse_queries
FORMAT_TIME = '%Y-%m-%d %H:%M:%S'
FORMAT_TIME_DAY_END = '%Y-%m-%d 23:59:59'

class TestFunnelTimeToConvert(ClickhouseTestMixin, APIBaseTest):
    maxDiff = None

    @snapshot_clickhouse_queries
    def test_auto_bin_count_single_step(self):
        if False:
            print('Hello World!')
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step two', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step three', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step two', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step two', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_from_step': 0, 'funnel_to_step': 1, 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(2220.0, 2), (42510.0, 0), (82800.0, 1)], 'average_conversion_time': 29540})

    def test_auto_bin_count_single_step_duplicate_events(self):
        if False:
            for i in range(10):
                print('nop')
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_from_step': 0, 'funnel_to_step': 1, 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step one', 'order': 1}, {'id': 'step one', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(2220.0, 2), (42510.0, 0), (82800.0, 1)], 'average_conversion_time': 29540})

    def test_custom_bin_count_single_step(self):
        if False:
            return 10
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step two', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step three', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step two', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step two', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_from_step': 0, 'funnel_to_step': 1, 'funnel_window_days': 7, 'bin_count': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(2220.0, 2), (13732.0, 0), (25244.0, 0), (36756.0, 0), (48268.0, 0), (59780.0, 0), (71292.0, 1), (82804.0, 0)], 'average_conversion_time': 29540})

    @snapshot_clickhouse_queries
    def test_auto_bin_count_total(self):
        if False:
            return 10
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step two', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step three', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step two', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step two', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(10800.0, 1), (10860.0, 0)], 'average_conversion_time': 10800.0})
        funnel_trends_steps_specified = ClickhouseFunnelTimeToConvert(Filter(data={**filter._data, 'funnel_from_step': 0, 'funnel_to_step': 2}), self.team)
        results_steps_specified = funnel_trends_steps_specified.run()
        self.assertEqual(results, results_steps_specified)

    @snapshot_clickhouse_queries
    def test_basic_unordered(self):
        if False:
            while True:
                i = 10
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_event(event='step three', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step two', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step two', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='step two', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_from_step': 0, 'funnel_to_step': 1, 'funnel_window_days': 7, 'funnel_order_type': FunnelOrderType.UNORDERED, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(2220.0, 2), (42510.0, 0), (82800.0, 1)], 'average_conversion_time': 29540})

    @snapshot_clickhouse_queries
    def test_basic_strict(self):
        if False:
            while True:
                i = 10
        _create_person(distinct_ids=['user a'], team=self.team)
        _create_person(distinct_ids=['user b'], team=self.team)
        _create_person(distinct_ids=['user c'], team=self.team)
        _create_person(distinct_ids=['user d'], team=self.team)
        _create_event(event='step one', distinct_id='user a', team=self.team, timestamp='2021-06-08 18:00:00')
        _create_event(event='step two', distinct_id='user a', team=self.team, timestamp='2021-06-08 19:00:00')
        _create_event(event='step three', distinct_id='user a', team=self.team, timestamp='2021-06-08 21:00:00')
        _create_event(event='step one', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:00:00')
        _create_event(event='step two', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:37:00')
        _create_event(event='blah', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:38:00')
        _create_event(event='step three', distinct_id='user b', team=self.team, timestamp='2021-06-09 13:39:00')
        _create_event(event='step one', distinct_id='user c', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='step two', distinct_id='user c', team=self.team, timestamp='2021-06-12 06:00:00')
        _create_event(event='step one', distinct_id='user d', team=self.team, timestamp='2021-06-11 07:00:00')
        _create_event(event='blah', distinct_id='user d', team=self.team, timestamp='2021-06-12 07:00:00')
        _create_event(event='step two', distinct_id='user d', team=self.team, timestamp='2021-06-12 09:00:00')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_from_step': 0, 'funnel_to_step': 1, 'funnel_window_days': 7, 'funnel_order_type': FunnelOrderType.STRICT, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTimeToConvert(filter, self.team)
        results = funnel_trends.run()
        self.assertEqual(results, {'bins': [(2220.0, 2), (42510.0, 0), (82800.0, 1)], 'average_conversion_time': 29540})