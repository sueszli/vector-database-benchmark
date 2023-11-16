from posthog.models.filters.mixins.funnel import FunnelWindowDaysMixin
from posthog.test.base import BaseTest

class TestFilterMixins(BaseTest):

    def test_funnel_window_days_to_microseconds(self):
        if False:
            for i in range(10):
                print('nop')
        one_day = FunnelWindowDaysMixin.microseconds_from_days(1)
        two_days = FunnelWindowDaysMixin.microseconds_from_days(2)
        three_days = FunnelWindowDaysMixin.microseconds_from_days(3)
        self.assertEqual(86400000000, one_day)
        self.assertEqual(172800000000, two_days)
        self.assertEqual(259200000000, three_days)

    def test_funnel_window_days_to_milliseconds(self):
        if False:
            while True:
                i = 10
        one_day = FunnelWindowDaysMixin.milliseconds_from_days(1)
        self.assertEqual(one_day, 86400000)