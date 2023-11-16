from AlgorithmImports import *

class NamedArgumentsRegression(QCAlgorithm):
    """Regression algorithm that makes use of PythonNet kwargs"""

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(month=10, day=8, year=2013)
        self.SetEndDate(month=10, day=17, year=2013)
        self.SetCash(startingCash=100000)
        if self.StartDate.year != 2013 or self.StartDate.month != 10 or self.StartDate.day != 8:
            raise AssertionError(f'Start date was incorrect! Expected 10/8/2013 Recieved {self.StartDate}')
        if self.EndDate.year != 2013 or self.EndDate.month != 10 or self.EndDate.day != 17:
            raise AssertionError(f'End date was incorrect! Expected 10/17/2013 Recieved {self.EndDate}')
        if self.Portfolio.Cash != 100000:
            raise AssertionError(f'Portfolio cash was incorrect! Expected 100000 Recieved {self.Portfolio.Cash}')
        symbol = self.AddEquity(resolution=Resolution.Daily, ticker='SPY').Symbol
        for config in self.SubscriptionManager.SubscriptionDataConfigService.GetSubscriptionDataConfigs(symbol):
            if config.Resolution != Resolution.Daily:
                raise AssertionError(f'Resolution was not correct on security')

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings(symbol='SPY', percentage=1)
            self.Debug(message='Purchased Stock')