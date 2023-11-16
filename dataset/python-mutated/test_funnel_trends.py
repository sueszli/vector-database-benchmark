from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from freezegun.api import freeze_time
from posthog.constants import INSIGHT_FUNNELS, TRENDS_LINEAR, FunnelOrderType
from posthog.models.cohort import Cohort
from posthog.models.filters import Filter
from posthog.queries.funnels.funnel_trends import ClickhouseFunnelTrends
from posthog.queries.funnels.funnel_trends_persons import ClickhouseFunnelTrendsActors
from posthog.test.base import APIBaseTest, ClickhouseTestMixin, _create_person, snapshot_clickhouse_queries
from posthog.test.test_journeys import journeys_for
FORMAT_TIME = '%Y-%m-%d %H:%M:%S'
FORMAT_TIME_DAY_END = '%Y-%m-%d 23:59:59'

class TestFunnelTrends(ClickhouseTestMixin, APIBaseTest):
    maxDiff = None

    def _get_actors_at_step(self, filter, entrance_period_start, drop_off):
        if False:
            print('Hello World!')
        person_filter = filter.shallow_clone({'entrance_period_start': entrance_period_start, 'drop_off': drop_off})
        funnel_query_builder = ClickhouseFunnelTrendsActors(person_filter, self.team)
        (_, serialized_result, _) = funnel_query_builder.get_actors()
        return serialized_result

    def _create_sample_data(self):
        if False:
            for i in range(10):
                print('nop')
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4)}], 'user_three': [{'event': 'step one', 'timestamp': datetime(2021, 5, 6)}], 'user_four': [{'event': 'step none', 'timestamp': datetime(2021, 5, 6)}], 'user_five': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 3)}], 'user_six': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_seven': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4)}], 'user_eight': []}, self.team)

    def test_no_event_in_period(self):
        if False:
            i = 10
            return i + 15
        journeys_for({'user a': [{'event': 'Step one', 'timestamp': datetime(2021, 6, 6, 21)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTrends(filter, self.team)
        results = funnel_trends._exec_query()
        formatted_results = funnel_trends._format_results(results)
        self.assertEqual(len(results), 7)
        self.assertEqual(formatted_results[0]['days'][0], '2021-06-07')

    def test_only_one_user_reached_one_step(self):
        if False:
            i = 10
            return i + 15
        journeys_for({'user a': [{'event': 'step one', 'timestamp': datetime(2021, 6, 7, 19)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-06-07 00:00:00', 'date_to': '2021-06-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        funnel_trends = ClickhouseFunnelTrends(filter, self.team)
        results = funnel_trends._exec_query()
        self.assertEqual(results, [{'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 1, 'timestamp': datetime(2021, 6, 7, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 8, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 9, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 10, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 11, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 12, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}, {'reached_to_step_count': 0, 'conversion_rate': 0, 'reached_from_step_count': 0, 'timestamp': datetime(2021, 6, 13, 0, 0).replace(tzinfo=ZoneInfo('UTC'))}])
        funnel_trends_persons_existent_dropped_off_results = self._get_actors_at_step(filter, '2021-06-07 00:00:00', True)
        self.assertEqual(len(funnel_trends_persons_existent_dropped_off_results), 1)
        self.assertEqual([person['distinct_ids'] for person in funnel_trends_persons_existent_dropped_off_results], [['user a']])
        funnel_trends_persons_nonexistent_converted_results = self._get_actors_at_step(filter, '2021-06-07 00:00:00', False)
        self.assertEqual(len(funnel_trends_persons_nonexistent_converted_results), 0)
        funnel_trends_persons_nonexistent_converted_results = self._get_actors_at_step(filter, '2021-06-08 00:00:00', True)
        self.assertEqual(len(funnel_trends_persons_nonexistent_converted_results), 0)

    def test_hour_interval(self):
        if False:
            for i in range(10):
                print('nop')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'hour', 'date_from': '2021-05-01 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        with freeze_time('2021-05-06T23:40:59Z'):
            results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 144)

    def test_day_interval(self):
        if False:
            i = 10
            return i + 15
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-07 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 0)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 2)}]}, self.team)
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(7, len(results))
        persons = self._get_actors_at_step(filter, '2021-05-01 00:00:00', False)
        self.assertEqual([person['distinct_ids'] for person in persons], [['user_one']])

    @snapshot_clickhouse_queries
    def test_week_interval(self):
        if False:
            return 10
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'week', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-07 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 0)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 2)}]}, self.team)
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        persons = self._get_actors_at_step(filter, '2021-04-25 00:00:00', False)
        self.assertEqual(2, len(results))
        self.assertEqual([person['distinct_ids'] for person in persons], [['user_one']])

    def test_month_interval(self):
        if False:
            i = 10
            return i + 15
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'month', 'date_from': '2020-01-01 00:00:00', 'date_to': '2020-07-01 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2020, 5, 1, 0)}, {'event': 'step two', 'timestamp': datetime(2020, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2020, 5, 1, 2)}]}, self.team)
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(results, [{'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 1, 1)}, {'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 2, 1)}, {'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 3, 1)}, {'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 4, 1)}, {'conversion_rate': 100.0, 'reached_from_step_count': 1, 'reached_to_step_count': 1, 'timestamp': date(2020, 5, 1)}, {'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 6, 1)}, {'conversion_rate': 0.0, 'reached_from_step_count': 0, 'reached_to_step_count': 0, 'timestamp': date(2020, 7, 1)}])
        persons = self._get_actors_at_step(filter, '2020-05-01 00:00:00', False)
        self.assertEqual([person['distinct_ids'] for person in persons], [['user_one']])

    def test_all_date_range(self):
        if False:
            print('Hello World!')
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': 'all', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 0)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 2)}]}, self.team)
        with freeze_time('2021-05-20T13:01:01Z'):
            results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(20, len(results))
        persons = self._get_actors_at_step(filter, '2021-05-01 00:00:00', False)
        self.assertEqual([person['distinct_ids'] for person in persons], [['user_one']])

    def test_all_results_for_day_interval(self):
        if False:
            print('Hello World!')
        self._create_sample_data()
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-07 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        saturday = results[0]
        self.assertEqual(3, saturday['reached_to_step_count'])
        self.assertEqual(3, saturday['reached_from_step_count'])
        self.assertEqual(100, saturday['conversion_rate'])
        sunday = results[1]
        self.assertEqual(0, sunday['reached_to_step_count'])
        self.assertEqual(2, sunday['reached_from_step_count'])
        self.assertEqual(0, sunday['conversion_rate'])
        monday = results[2]
        self.assertEqual(0, monday['reached_to_step_count'])
        self.assertEqual(0, monday['reached_from_step_count'])
        self.assertEqual(0, monday['conversion_rate'])
        tuesday = results[3]
        self.assertEqual(0, tuesday['reached_to_step_count'])
        self.assertEqual(0, tuesday['reached_from_step_count'])
        self.assertEqual(0, tuesday['conversion_rate'])
        wednesday = results[4]
        self.assertEqual(0, wednesday['reached_to_step_count'])
        self.assertEqual(0, wednesday['reached_from_step_count'])
        self.assertEqual(0, wednesday['conversion_rate'])
        thursday = results[5]
        self.assertEqual(0, thursday['reached_to_step_count'])
        self.assertEqual(1, thursday['reached_from_step_count'])
        self.assertEqual(0, thursday['conversion_rate'])
        friday = results[6]
        self.assertEqual(0, friday['reached_to_step_count'])
        self.assertEqual(0, friday['reached_from_step_count'])
        self.assertEqual(0, friday['conversion_rate'])

    def test_window_size_one_day(self):
        if False:
            print('Hello World!')
        self._create_sample_data()
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-07 00:00:00', 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        saturday = results[0]
        self.assertEqual(1, saturday['reached_to_step_count'])
        self.assertEqual(3, saturday['reached_from_step_count'])
        self.assertEqual(33.33, saturday['conversion_rate'])
        sunday = results[1]
        self.assertEqual(0, sunday['reached_to_step_count'])
        self.assertEqual(2, sunday['reached_from_step_count'])
        self.assertEqual(0, sunday['conversion_rate'])
        monday = results[2]
        self.assertEqual(0, monday['reached_to_step_count'])
        self.assertEqual(0, monday['reached_from_step_count'])
        self.assertEqual(0, monday['conversion_rate'])
        tuesday = results[3]
        self.assertEqual(0, tuesday['reached_to_step_count'])
        self.assertEqual(0, tuesday['reached_from_step_count'])
        self.assertEqual(0, tuesday['conversion_rate'])
        wednesday = results[4]
        self.assertEqual(0, wednesday['reached_to_step_count'])
        self.assertEqual(0, wednesday['reached_from_step_count'])
        self.assertEqual(0, wednesday['conversion_rate'])
        thursday = results[5]
        self.assertEqual(0, thursday['reached_to_step_count'])
        self.assertEqual(1, thursday['reached_from_step_count'])
        self.assertEqual(0, thursday['conversion_rate'])
        friday = results[6]
        self.assertEqual(0, friday['reached_to_step_count'])
        self.assertEqual(0, friday['reached_from_step_count'])
        self.assertEqual(0, friday['conversion_rate'])

    def test_period_not_final(self):
        if False:
            while True:
                i = 10
        now = datetime.now()
        journeys_for({'user_eight': [{'event': 'step one', 'timestamp': now}, {'event': 'step two', 'timestamp': now + timedelta(minutes=1)}, {'event': 'step three', 'timestamp': now + timedelta(minutes=2)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': (now - timedelta(1)).strftime(FORMAT_TIME), 'date_to': now.strftime(FORMAT_TIME_DAY_END), 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 2)
        day = results[0]
        self.assertEqual(day['reached_from_step_count'], 0)
        self.assertEqual(day['reached_to_step_count'], 0)
        self.assertEqual(day['conversion_rate'], 0)
        self.assertEqual(day['timestamp'].replace(tzinfo=ZoneInfo('UTC')), (datetime(now.year, now.month, now.day) - timedelta(1)).replace(tzinfo=ZoneInfo('UTC')))
        day = results[1]
        self.assertEqual(day['reached_from_step_count'], 1)
        self.assertEqual(day['reached_to_step_count'], 1)
        self.assertEqual(day['conversion_rate'], 100)
        self.assertEqual(day['timestamp'].replace(tzinfo=ZoneInfo('UTC')), datetime(now.year, now.month, now.day).replace(tzinfo=ZoneInfo('UTC')))

    def test_two_runs_by_single_user_in_one_period(self):
        if False:
            i = 10
            return i + 15
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 0)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 1, 13)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 14)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 15)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-01 23:59:59', 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 1)
        day = results[0]
        self.assertEqual(day['reached_from_step_count'], 1)
        self.assertEqual(day['reached_to_step_count'], 1)
        self.assertEqual(day['conversion_rate'], 100)

    def test_steps_performed_in_period_but_in_reverse(self):
        if False:
            return 10
        journeys_for({'user_one': [{'event': 'step three', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 1, 3)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-01 23:59:59', 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 1)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 1)
        self.assertEqual(day_1['reached_to_step_count'], 0)
        self.assertEqual(day_1['conversion_rate'], 0)

    def test_one_person_in_multiple_periods_and_windows(self):
        if False:
            i = 10
            return i + 15
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 3)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 3, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3, 2)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 4, 11)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4, 12)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 4, 13)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 4, 18)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-04 23:59:59', 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 4)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 1)
        self.assertEqual(day_1['reached_to_step_count'], 1)
        self.assertEqual(day_1['conversion_rate'], 100)
        day_2 = results[1]
        self.assertEqual(day_2['reached_from_step_count'], 0)
        self.assertEqual(day_2['reached_to_step_count'], 0)
        self.assertEqual(day_2['conversion_rate'], 0)
        day_3 = results[2]
        self.assertEqual(day_3['reached_from_step_count'], 1)
        self.assertEqual(day_3['reached_to_step_count'], 0)
        self.assertEqual(day_3['conversion_rate'], 0)
        day_4 = results[3]
        self.assertEqual(day_4['reached_from_step_count'], 2)
        self.assertEqual(day_4['reached_to_step_count'], 1)
        self.assertEqual(day_4['conversion_rate'], 50)
        funnel_trends_persons_existent_dropped_off_results = self._get_actors_at_step(filter, '2021-05-04 00:00:00', True)
        self.assertEqual(len(funnel_trends_persons_existent_dropped_off_results), 1)
        self.assertEqual([person['distinct_ids'] for person in funnel_trends_persons_existent_dropped_off_results], [['user_two']])
        funnel_trends_persons_existent_dropped_off_results = self._get_actors_at_step(filter, '2021-05-04 00:00:00', False)
        self.assertEqual(len(funnel_trends_persons_existent_dropped_off_results), 1)
        self.assertEqual([person['distinct_ids'] for person in funnel_trends_persons_existent_dropped_off_results], [['user_one']])

    def test_from_second_step(self):
        if False:
            print('Hello World!')
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 2, 3)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}], 'user_three': [{'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 2, 3)}], 'user_four': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-02 23:59:59', 'funnel_window_days': 3, 'funnel_from_step': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 2)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 1)
        self.assertEqual(day_1['reached_to_step_count'], 1)
        self.assertEqual(day_1['conversion_rate'], 100)
        day_2 = results[1]
        self.assertEqual(day_2['reached_from_step_count'], 1)
        self.assertEqual(day_2['reached_to_step_count'], 0)
        self.assertEqual(day_2['conversion_rate'], 0)

    def test_to_second_step(self):
        if False:
            return 10
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 2, 3)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}], 'user_three': [{'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 2, 3)}], 'user_four': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 2, 2)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-02 23:59:59', 'funnel_window_days': 3, 'funnel_to_step': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 2)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 2)
        self.assertEqual(day_1['reached_to_step_count'], 1)
        self.assertEqual(day_1['conversion_rate'], 50)
        day_2 = results[1]
        self.assertEqual(day_2['reached_from_step_count'], 1)
        self.assertEqual(day_2['reached_to_step_count'], 1)
        self.assertEqual(day_2['conversion_rate'], 100)

    def test_one_person_in_multiple_periods_and_windows_in_unordered_funnel(self):
        if False:
            return 10
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 3)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3, 1)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 3, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 4, 11)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 4, 12)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4, 13)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 4, 18)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-04 23:59:59', 'funnel_window_days': 1, 'funnel_order_type': FunnelOrderType.UNORDERED, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 4)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 1)
        self.assertEqual(day_1['reached_to_step_count'], 1)
        self.assertEqual(day_1['conversion_rate'], 100)
        day_2 = results[1]
        self.assertEqual(day_2['reached_from_step_count'], 0)
        self.assertEqual(day_2['reached_to_step_count'], 0)
        self.assertEqual(day_2['conversion_rate'], 0)
        day_3 = results[2]
        self.assertEqual(day_3['reached_from_step_count'], 1)
        self.assertEqual(day_3['reached_to_step_count'], 0)
        self.assertEqual(day_3['conversion_rate'], 0)
        day_4 = results[3]
        self.assertEqual(day_4['reached_from_step_count'], 2)
        self.assertEqual(day_4['reached_to_step_count'], 1)
        self.assertEqual(day_4['conversion_rate'], 50)
        funnel_trends_persons_existent_dropped_off_results = self._get_actors_at_step(filter, '2021-05-04 00:00:00', True)
        self.assertEqual(len(funnel_trends_persons_existent_dropped_off_results), 1)
        self.assertEqual([person['distinct_ids'] for person in funnel_trends_persons_existent_dropped_off_results], [['user_two']])
        funnel_trends_persons_existent_dropped_off_results = self._get_actors_at_step(filter, '2021-05-04 00:00:00', False)
        self.assertEqual(len(funnel_trends_persons_existent_dropped_off_results), 1)
        self.assertEqual([person['distinct_ids'] for person in funnel_trends_persons_existent_dropped_off_results], [['user_one']])

    def test_one_person_in_multiple_periods_and_windows_in_strict_funnel(self):
        if False:
            return 10
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 3)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 3, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3, 2)}, {'event': 'blah', 'timestamp': datetime(2021, 5, 3, 2, 30)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 3, 3)}, {'event': 'step one', 'timestamp': datetime(2021, 5, 4, 11)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4, 12)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 4, 13)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 4, 18)}, {'event': 'blah', 'timestamp': datetime(2021, 5, 4, 18, 20)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4, 19)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-04 23:59:59', 'funnel_order_type': FunnelOrderType.STRICT, 'funnel_window_days': 1, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.assertEqual(len(results), 4)
        day_1 = results[0]
        self.assertEqual(day_1['reached_from_step_count'], 1)
        self.assertEqual(day_1['reached_to_step_count'], 1)
        self.assertEqual(day_1['conversion_rate'], 100)
        day_2 = results[1]
        self.assertEqual(day_2['reached_from_step_count'], 0)
        self.assertEqual(day_2['reached_to_step_count'], 0)
        self.assertEqual(day_2['conversion_rate'], 0)
        day_3 = results[2]
        self.assertEqual(day_3['reached_from_step_count'], 1)
        self.assertEqual(day_3['reached_to_step_count'], 0)
        self.assertEqual(day_3['conversion_rate'], 0)
        day_4 = results[3]
        self.assertEqual(day_4['reached_from_step_count'], 2)
        self.assertEqual(day_4['reached_to_step_count'], 1)
        self.assertEqual(day_4['conversion_rate'], 50)

    def test_funnel_step_breakdown_event(self):
        if False:
            for i in range(10):
                print('nop')
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1), 'properties': {'$browser': 'Chrome'}}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3), 'properties': {'$browser': 'Chrome'}}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5), 'properties': {'$browser': 'Chrome'}}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2), 'properties': {'$browser': 'Chrome'}}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3), 'properties': {'$browser': 'Chrome'}}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5), 'properties': {'$browser': 'Chrome'}}], 'user_three': [{'event': 'step one', 'timestamp': datetime(2021, 5, 3), 'properties': {'$browser': 'Safari'}}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4), 'properties': {'$browser': 'Safari'}}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5), 'properties': {'$browser': 'Safari'}}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}], 'breakdown_type': 'event', 'breakdown': '$browser'})
        funnel_trends = ClickhouseFunnelTrends(filter, self.team)
        result = funnel_trends.run()
        self.assertEqual(len(result), 2)
        for res in result:
            if res['breakdown_value'] == ['Chrome']:
                self.assertEqual(res['data'], [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif res['breakdown_value'] == ['Safari']:
                self.assertEqual(res['data'], [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                self.fail(msg='Invalid breakdown value')

    def test_funnel_step_breakdown_person(self):
        if False:
            i = 10
            return i + 15
        _create_person(distinct_ids=['user_one'], team=self.team, properties={'$browser': 'Chrome'})
        _create_person(distinct_ids=['user_two'], team=self.team, properties={'$browser': 'Chrome'})
        _create_person(distinct_ids=['user_three'], team=self.team, properties={'$browser': 'Safari'})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_three': [{'event': 'step one', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}], 'breakdown_type': 'person', 'breakdown': '$browser'})
        funnel_trends = ClickhouseFunnelTrends(filter, self.team)
        result = funnel_trends.run()
        self.assertEqual(len(result), 2)
        for res in result:
            if res['breakdown_value'] == ['Chrome']:
                self.assertEqual(res['data'], [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif res['breakdown_value'] == ['Safari']:
                self.assertEqual(res['data'], [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                self.fail(msg='Invalid breakdown value')

    def test_funnel_trend_cohort_breakdown(self):
        if False:
            while True:
                i = 10
        _create_person(distinct_ids=['user_one'], team=self.team, properties={'key': 'value'})
        _create_person(distinct_ids=['user_two'], team=self.team, properties={'key': 'value'})
        _create_person(distinct_ids=['user_three'], team=self.team, properties={'$browser': 'Safari'})
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 2)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}], 'user_three': [{'event': 'step one', 'timestamp': datetime(2021, 5, 3)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 4)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 5)}]}, self.team)
        cohort = Cohort.objects.create(team=self.team, name='test_cohort', groups=[{'properties': [{'key': 'key', 'value': 'value', 'type': 'person'}]}])
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'date_to': '2021-05-13 23:59:59', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}], 'breakdown_type': 'cohort', 'breakdown': [cohort.pk]})
        funnel_trends = ClickhouseFunnelTrends(filter, self.team)
        result = funnel_trends.run()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['data'], [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @snapshot_clickhouse_queries
    def test_timezones_trends(self):
        if False:
            while True:
                i = 10
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 10)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 11)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 12)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 2)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 3)}], 'user_three': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 1)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 10)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 11)}], 'user_eight': []}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'interval': 'day', 'date_from': '2021-04-30 00:00:00', 'date_to': '2021-05-07 00:00:00', 'funnel_window_days': 7, 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        self.team.timezone = 'US/Pacific'
        self.team.save()
        results_pacific = ClickhouseFunnelTrends(filter, self.team)._exec_query()
        saturday = results[1]
        self.assertEqual(3, saturday['reached_to_step_count'])
        self.assertEqual(3, saturday['reached_from_step_count'])
        self.assertEqual(100.0, saturday['conversion_rate'])
        friday_pacific = results_pacific[0]
        self.assertEqual(2, friday_pacific['reached_to_step_count'])
        self.assertEqual(2, friday_pacific['reached_from_step_count'])
        self.assertEqual(100.0, friday_pacific['conversion_rate'])
        saturday_pacific = results_pacific[1]
        self.assertEqual(1, saturday_pacific['reached_to_step_count'])
        self.assertEqual(1, saturday_pacific['reached_from_step_count'])

    def test_trend_for_hour_based_conversion_window(self):
        if False:
            print('Hello World!')
        journeys_for({'user_one': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 10)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 11)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 12)}], 'user_two': [{'event': 'step one', 'timestamp': datetime(2021, 5, 1, 10)}, {'event': 'step two', 'timestamp': datetime(2021, 5, 1, 11)}, {'event': 'step three', 'timestamp': datetime(2021, 5, 1, 14)}]}, self.team)
        filter = Filter(data={'insight': INSIGHT_FUNNELS, 'display': TRENDS_LINEAR, 'funnel_window_interval': 3, 'funnel_window_interval_unit': 'hour', 'interval': 'day', 'date_from': '2021-05-01 00:00:00', 'events': [{'id': 'step one', 'order': 0}, {'id': 'step two', 'order': 1}, {'id': 'step three', 'order': 2}]})
        with freeze_time('2021-05-06T23:40:59Z'):
            results = ClickhouseFunnelTrends(filter, self.team)._exec_query()
            conversion_rates = [row['conversion_rate'] for row in results]
            self.assertEqual(conversion_rates, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0])