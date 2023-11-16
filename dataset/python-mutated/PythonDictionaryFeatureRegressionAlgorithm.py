from AlgorithmImports import *

class PythonDictionaryFeatureRegressionAlgorithm(QCAlgorithm):
    """Example algorithm showing that Slice, Securities and Portfolio behave as a Python Dictionary"""

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.spySymbol = self.AddEquity('SPY').Symbol
        self.ibmSymbol = self.AddEquity('IBM').Symbol
        self.aigSymbol = self.AddEquity('AIG').Symbol
        self.aaplSymbol = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        dateRules = self.DateRules.On(2013, 10, 7)
        self.Schedule.On(dateRules, self.TimeRules.At(13, 0), self.TestSecuritiesDictionary)
        self.Schedule.On(dateRules, self.TimeRules.At(14, 0), self.TestPortfolioDictionary)
        self.Schedule.On(dateRules, self.TimeRules.At(15, 0), self.TestSliceDictionary)

    def TestSliceDictionary(self):
        if False:
            i = 10
            return i + 15
        slice = self.CurrentSlice
        symbols = ', '.join([f'{x}' for x in slice.keys()])
        sliceData = ', '.join([f'{x}' for x in slice.values()])
        sliceBars = ', '.join([f'{x}' for x in slice.Bars.values()])
        if 'SPY' not in slice:
            raise Exception('SPY (string) is not in Slice')
        if self.spySymbol not in slice:
            raise Exception('SPY (Symbol) is not in Slice')
        spy = slice.get(self.spySymbol)
        if spy is None:
            raise Exception('SPY is not in Slice')
        for (symbol, bar) in slice.Bars.items():
            self.Plot(symbol, 'Price', bar.Close)

    def TestSecuritiesDictionary(self):
        if False:
            print('Hello World!')
        symbols = ', '.join([f'{x}' for x in self.Securities.keys()])
        leverages = ', '.join([str(x.GetLastData()) for x in self.Securities.values()])
        if 'IBM' not in self.Securities:
            raise Exception('IBM (string) is not in Securities')
        if self.ibmSymbol not in self.Securities:
            raise Exception('IBM (Symbol) is not in Securities')
        ibm = self.Securities.get(self.ibmSymbol)
        if ibm is None:
            raise Exception('ibm is None')
        aapl = self.Securities.get(self.aaplSymbol)
        if aapl is not None:
            raise Exception('aapl is not None')
        for (symbol, security) in self.Securities.items():
            self.Plot(symbol, 'Price', security.Price)

    def TestPortfolioDictionary(self):
        if False:
            i = 10
            return i + 15
        symbols = ', '.join([f'{x}' for x in self.Portfolio.keys()])
        leverages = ', '.join([f'{x.Symbol}: {x.Leverage}' for x in self.Portfolio.values()])
        if 'AIG' not in self.Securities:
            raise Exception('AIG (string) is not in Portfolio')
        if self.aigSymbol not in self.Securities:
            raise Exception('AIG (Symbol) is not in Portfolio')
        aig = self.Portfolio.get(self.aigSymbol)
        if aig is None:
            raise Exception('aig is None')
        aapl = self.Portfolio.get(self.aaplSymbol)
        if aapl is not None:
            raise Exception('aapl is not None')
        for (symbol, holdings) in self.Portfolio.items():
            msg = f'{symbol}: {holdings.Leverage}'

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        portfolioCopy = self.Portfolio.copy()
        try:
            self.Portfolio.clear()
        except Exception as e:
            self.Debug(e)
        bar = self.Securities.pop('SPY')
        length = len(self.Securities)
        if length != 2:
            raise Exception(f'After popping SPY, Securities should have 2 elements, {length} found')
        securitiesCopy = self.Securities.copy()
        self.Securities.clear()

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1 / 3)
            self.SetHoldings('IBM', 1 / 3)
            self.SetHoldings('AIG', 1 / 3)