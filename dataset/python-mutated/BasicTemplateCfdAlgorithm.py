from AlgorithmImports import *

class BasicTemplateCfdAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetAccountCurrency('EUR')
        self.SetStartDate(2019, 2, 20)
        self.SetEndDate(2019, 2, 21)
        self.SetCash('EUR', 100000)
        self.symbol = self.AddCfd('DE30EUR').Symbol
        history = self.History(self.symbol, 60, Resolution.Daily)
        self.Log(f'Received {len(history)} bars from CFD historical data call.')

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n        Arguments:\n            slice: Slice object keyed by symbol containing the stock data\n        '
        if data.QuoteBars.ContainsKey(self.symbol):
            quoteBar = data.QuoteBars[self.symbol]
            self.Log(f'{quoteBar.EndTime} :: {quoteBar.Close}')
        if not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Debug('{} {}'.format(self.Time, orderEvent.ToString()))