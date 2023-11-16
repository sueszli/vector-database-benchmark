from unittest import TestCase
import pandas as pd
from .test_events import StatefulRulesTests, StatelessRulesTests, minutes_for_days
from zipline.utils.events import AfterOpen

class TestStatelessRulesCMES(StatelessRulesTests, TestCase):
    CALENDAR_STRING = 'CMES'
    HALF_SESSION = pd.Timestamp('2014-07-04', tz='UTC')
    FULL_SESSION = pd.Timestamp('2014-09-24', tz='UTC')

    def test_far_after_open(self):
        if False:
            i = 10
            return i + 15
        minute_groups = minutes_for_days(self.cal, ordered_days=True)
        after_open = AfterOpen(hours=9, minutes=25)
        after_open.cal = self.cal
        for session_minutes in minute_groups:
            for (i, minute) in enumerate(session_minutes):
                if i != 564:
                    self.assertFalse(after_open.should_trigger(minute))
                else:
                    self.assertTrue(after_open.should_trigger(minute))

class TestStatefulRulesCMES(StatefulRulesTests, TestCase):
    CALENDAR_STRING = 'CMES'