from AlgorithmImports import *

class FundamentalRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2014, 3, 25)
        self.SetEndDate(2014, 4, 7)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddEquity('SPY')
        self.AddEquity('AAPL')
        ibm = Symbol.Create('IBM', SecurityType.Equity, Market.USA)
        ibmFundamental = self.Fundamentals(ibm)
        if self.Time != self.StartDate or self.Time != ibmFundamental.EndTime:
            raise ValueError(f'Unexpected Fundamental time {ibmFundamental.EndTime}')
        if ibmFundamental.Price == 0:
            raise ValueError(f'Unexpected Fundamental IBM price!')
        nb = Symbol.Create('NB', SecurityType.Equity, Market.USA)
        fundamentals = self.Fundamentals([nb, ibm])
        if len(fundamentals) != 2:
            raise ValueError(f'Unexpected Fundamental count {len(fundamentals)}! Expected 2')
        history = self.History(Fundamental, TimeSpan(1, 0, 0, 0))
        if len(history) != 2:
            raise ValueError(f'Unexpected Fundamental history count {len(history)}! Expected 2')
        for ticker in ['AAPL', 'SPY']:
            data = history.loc[ticker]
            if data['value'][0] == 0:
                raise ValueError(f'Unexpected {data} fundamental data')
        history2 = self.History(Fundamentals, TimeSpan(1, 0, 0, 0))
        if len(history2) != 1:
            raise ValueError(f'Unexpected Fundamentals history count {len(history2)}! Expected 1')
        data = history2['data'][0]
        if len(data) < 7000:
            raise ValueError(f'Unexpected Fundamentals data count {len(data)}! Expected > 7000')
        for fundamental in data:
            if type(fundamental) is not Fundamental:
                raise ValueError(f'Unexpected Fundamentals data type! {fundamental}')
        self.AddUniverse(self.SelectionFunction)
        self.changes = None
        self.numberOfSymbolsFundamental = 2

    def SelectionFunction(self, fundamental):
        if False:
            while True:
                i = 10
        sortedByDollarVolume = sorted([x for x in fundamental if x.Price > 1], key=lambda x: x.DollarVolume, reverse=True)
        sortedByPeRatio = sorted(sortedByDollarVolume, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
        return [x.Symbol for x in sortedByPeRatio[:self.numberOfSymbolsFundamental]]

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if self.changes is None:
            return
        for security in self.changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
                self.Debug('Liquidated Stock: ' + str(security.Symbol.Value))
        for security in self.changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.02)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        self.changes = changes