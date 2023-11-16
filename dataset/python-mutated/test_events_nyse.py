from functools import partial
from unittest import TestCase
from datetime import timedelta
import pandas as pd
from nose_parameterized import parameterized
from zipline.utils.events import NDaysBeforeLastTradingDayOfWeek, AfterOpen, BeforeClose
from zipline.utils.events import NthTradingDayOfWeek
from .test_events import StatelessRulesTests, StatefulRulesTests, minutes_for_days
T = partial(pd.Timestamp, tz='UTC')

class TestStatelessRulesNYSE(StatelessRulesTests, TestCase):
    CALENDAR_STRING = 'NYSE'
    HALF_SESSION = pd.Timestamp('2014-07-03', tz='UTC')
    FULL_SESSION = pd.Timestamp('2014-09-24', tz='UTC')

    def test_edge_cases_for_TradingDayOfWeek(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that we account for midweek holidays. Monday 01/20 is a holiday.\n        Ensure that the trigger date for that week is adjusted\n        appropriately, or thrown out if not enough trading days. Also, test\n        that if we start the simulation on a day where we miss the trigger\n        for that week, that the trigger is recalculated for next week.\n        '
        rule = NthTradingDayOfWeek(0)
        rule.cal = self.cal
        expected = {'2013-12-30': True, '2013-12-31': False, '2014-01-02': False, '2014-01-06': True, '2014-01-21': True, '2014-01-22': False}
        results = {x: rule.should_trigger(self.cal.next_open(T(x))) for x in expected.keys()}
        self.assertEquals(expected, results)
        rule = NthTradingDayOfWeek(1)
        rule.cal = self.cal
        expected = {'2013-12-31': True, '2014-01-02': False, '2014-01-22': True, '2014-01-23': False}
        results = {x: rule.should_trigger(self.cal.next_open(T(x))) for x in expected.keys()}
        self.assertEquals(expected, results)
        rule = NDaysBeforeLastTradingDayOfWeek(0)
        rule.cal = self.cal
        expected = {'2014-01-03': True, '2014-01-02': False, '2014-01-24': True, '2014-01-23': False}
        results = {x: rule.should_trigger(self.cal.next_open(T(x))) for x in expected.keys()}
        self.assertEquals(expected, results)

    @parameterized.expand([('week_start',), ('week_end',)])
    def test_week_and_time_composed_rule(self, rule_type):
        if False:
            for i in range(10):
                print('nop')
        week_rule = NthTradingDayOfWeek(0) if rule_type == 'week_start' else NDaysBeforeLastTradingDayOfWeek(4)
        time_rule = AfterOpen(minutes=60)
        week_rule.cal = self.cal
        time_rule.cal = self.cal
        composed_rule = week_rule & time_rule
        should_trigger = composed_rule.should_trigger
        week_minutes = self.cal.minutes_for_sessions_in_range(pd.Timestamp('2014-01-06', tz='UTC'), pd.Timestamp('2014-01-10', tz='UTC'))
        dt = pd.Timestamp('2014-01-06 14:30:00', tz='UTC')
        trigger_day_offset = 0
        trigger_minute_offset = 60
        n_triggered = 0
        for m in week_minutes:
            if should_trigger(m):
                self.assertEqual(m, dt + timedelta(days=trigger_day_offset) + timedelta(minutes=trigger_minute_offset))
                n_triggered += 1
        self.assertEqual(n_triggered, 1)

    def test_offset_too_far(self):
        if False:
            return 10
        minute_groups = minutes_for_days(self.cal, ordered_days=True)
        after_open_rule = AfterOpen(hours=11, minutes=11)
        after_open_rule.cal = self.cal
        before_close_rule = BeforeClose(hours=11, minutes=5)
        before_close_rule.cal = self.cal
        for session_minutes in minute_groups:
            for minute in session_minutes:
                self.assertFalse(after_open_rule.should_trigger(minute))
                self.assertFalse(before_close_rule.should_trigger(minute))

class TestStatefulRulesNYSE(StatefulRulesTests, TestCase):
    CALENDAR_STRING = 'NYSE'