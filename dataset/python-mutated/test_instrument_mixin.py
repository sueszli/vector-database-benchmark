from rqalpha.utils.testing import DataProxyFixture, RQAlphaTestCase

class InstrumentMixinTestCase(DataProxyFixture, RQAlphaTestCase):

    def init_fixture(self):
        if False:
            i = 10
            return i + 15
        super(InstrumentMixinTestCase, self).init_fixture()

    def test_get_trading_period(self):
        if False:
            for i in range(10):
                print('nop')
        from datetime import time
        from rqalpha.utils import TimeRange
        rb_time_range = self.data_proxy.get_trading_period(['RB1912'])
        self.assertSetEqual(set(rb_time_range), {TimeRange(start=time(21, 1), end=time(23, 0)), TimeRange(start=time(9, 1), end=time(10, 15)), TimeRange(start=time(10, 31), end=time(11, 30)), TimeRange(start=time(13, 31), end=time(15, 0))})
        merged_time_range = self.data_proxy.get_trading_period(['AG1912', 'TF1912'], [TimeRange(start=time(9, 31), end=time(11, 30)), TimeRange(start=time(13, 1), end=time(15, 0))])
        self.assertSetEqual(set(merged_time_range), {TimeRange(start=time(21, 1), end=time(23, 59)), TimeRange(start=time(0, 0), end=time(2, 30)), TimeRange(start=time(9, 1), end=time(11, 30)), TimeRange(start=time(13, 1), end=time(15, 15))})

    def test_is_night_trading(self):
        if False:
            i = 10
            return i + 15
        assert not self.data_proxy.is_night_trading(['TF1912'])
        assert self.data_proxy.is_night_trading(['AG1912', '000001.XSHE'])