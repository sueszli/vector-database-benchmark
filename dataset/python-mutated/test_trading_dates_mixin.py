from rqalpha.utils.testing import DataProxyFixture, RQAlphaTestCase

class TradingDateMixinTestCase(DataProxyFixture, RQAlphaTestCase):

    def init_fixture(self):
        if False:
            for i in range(10):
                print('nop')
        super(TradingDateMixinTestCase, self).init_fixture()

    def test_count_trading_dates(self):
        if False:
            return 10
        from datetime import date
        assert self.data_proxy.count_trading_dates(date(2018, 11, 1), date(2018, 11, 12)) == 8
        assert self.data_proxy.count_trading_dates(date(2018, 11, 3), date(2018, 11, 12)) == 6
        assert self.data_proxy.count_trading_dates(date(2018, 11, 3), date(2018, 11, 18)) == 10