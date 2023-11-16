from AlgorithmImports import *
from custom_data import *

class SecurityHistoryTest:

    def __init__(self, start_date, security_type, symbol):
        if False:
            while True:
                i = 10
        self.qb = QuantBook()
        self.qb.SetStartDate(start_date)
        self.symbol = self.qb.AddSecurity(security_type, symbol).Symbol
        self.column = 'close'

    def __str__(self):
        if False:
            print('Hello World!')
        return '{} on {}'.format(self.symbol.ID, self.qb.StartDate)

    def test_period_overload(self, period):
        if False:
            return 10
        history = self.qb.History([self.symbol], period)
        return history[self.column].unstack(level=0)

    def test_daterange_overload(self, end):
        if False:
            i = 10
            return i + 15
        start = end - timedelta(1)
        history = self.qb.History([self.symbol], start, end)
        return history[self.column].unstack(level=0)

class OptionHistoryTest(SecurityHistoryTest):

    def test_daterange_overload(self, end, start=None):
        if False:
            print('Hello World!')
        if start is None:
            start = end - timedelta(1)
        history = self.qb.GetOptionHistory(self.symbol, start, end)
        return history.GetAllData()

class FutureHistoryTest(SecurityHistoryTest):

    def test_daterange_overload(self, end, start=None, maxFilter=182):
        if False:
            while True:
                i = 10
        if start is None:
            start = end - timedelta(1)
        self.qb.Securities[self.symbol].SetFilter(0, maxFilter)
        history = self.qb.GetFutureHistory(self.symbol, start, end)
        return history.GetAllData()

class FutureContractHistoryTest:

    def __init__(self, start_date, security_type, symbol):
        if False:
            return 10
        self.qb = QuantBook()
        self.qb.SetStartDate(start_date)
        self.symbol = symbol
        self.column = 'close'

    def test_daterange_overload(self, end):
        if False:
            return 10
        start = end - timedelta(1)
        history = self.qb.GetFutureHistory(self.symbol, start, end)
        return history.GetAllData()

class OptionContractHistoryTest(FutureContractHistoryTest):

    def test_daterange_overload(self, end):
        if False:
            while True:
                i = 10
        start = end - timedelta(1)
        history = self.qb.GetOptionHistory(self.symbol, start, end)
        return history.GetAllData()

class CustomDataHistoryTest(SecurityHistoryTest):

    def __init__(self, start_date, security_type, symbol):
        if False:
            return 10
        self.qb = QuantBook()
        self.qb.SetStartDate(start_date)
        if security_type == 'Nifty':
            type = Nifty
            self.column = 'close'
        elif security_type == 'CustomPythonData':
            type = CustomPythonData
            self.column = 'close'
        else:
            raise
        self.symbol = self.qb.AddData(type, symbol, Resolution.Daily).Symbol

class MultipleSecuritiesHistoryTest(SecurityHistoryTest):

    def __init__(self, start_date, security_type, symbol):
        if False:
            for i in range(10):
                print('nop')
        self.qb = QuantBook()
        self.qb.SetStartDate(start_date)
        self.qb.AddEquity('SPY', Resolution.Daily)
        self.qb.AddForex('EURUSD', Resolution.Daily)
        self.qb.AddCrypto('BTCUSD', Resolution.Daily)

    def test_period_overload(self, period):
        if False:
            return 10
        history = self.qb.History(self.qb.Securities.Keys, period)
        return history['close'].unstack(level=0)

class FundamentalHistoryTest:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.qb = QuantBook()

    def getFundamentals(self, ticker, selector, start, end):
        if False:
            print('Hello World!')
        return self.qb.GetFundamental(ticker, selector, start, end)