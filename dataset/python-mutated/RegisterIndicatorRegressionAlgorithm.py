from AlgorithmImports import *
from CustomDataRegressionAlgorithm import Bitcoin

class RegisterIndicatorRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 9)
        SP500 = Symbol.Create(Futures.Indices.SP500EMini, SecurityType.Future, Market.CME)
        self._symbol = _symbol = self.FutureChainProvider.GetFutureContractList(SP500, self.StartDate + timedelta(days=1))[0]
        self.AddFutureContract(_symbol)
        self._indicators = []
        self._selectorCalled = [False, False, False, False, False, False]
        indicator = CustomIndicator()
        consolidator = self.ResolveConsolidator(_symbol, Resolution.Minute, QuoteBar)
        self.RegisterIndicator(_symbol, indicator, consolidator)
        self._indicators.append(indicator)
        indicator2 = CustomIndicator()
        consolidator = self.ResolveConsolidator(_symbol, timedelta(minutes=1), QuoteBar)
        self.RegisterIndicator(_symbol, indicator2, consolidator, lambda bar: self.SetSelectorCalled(0) and bar)
        self._indicators.append(indicator2)
        indicator3 = SimpleMovingAverage(10)
        consolidator = self.ResolveConsolidator(_symbol, timedelta(minutes=1), QuoteBar)
        self.RegisterIndicator(_symbol, indicator3, consolidator, lambda bar: self.SetSelectorCalled(1) and bar.Ask.High - bar.Bid.Low)
        self._indicators.append(indicator3)
        movingAverage = SimpleMovingAverage(10)
        self.RegisterIndicator(_symbol, movingAverage, Resolution.Minute, lambda bar: self.SetSelectorCalled(2) and bar.Volume)
        self._indicators.append(movingAverage)
        movingAverage2 = SimpleMovingAverage(10)
        self.RegisterIndicator(_symbol, movingAverage2, Resolution.Minute)
        self._indicators.append(movingAverage2)
        movingAverage3 = SimpleMovingAverage(10)
        self.RegisterIndicator(_symbol, movingAverage3, timedelta(minutes=1))
        self._indicators.append(movingAverage3)
        movingAverage4 = SimpleMovingAverage(10)
        self.RegisterIndicator(_symbol, movingAverage4, timedelta(minutes=1), lambda bar: self.SetSelectorCalled(3) and bar.Volume)
        self._indicators.append(movingAverage4)
        symbolCustom = self.AddData(Bitcoin, 'BTC', Resolution.Minute).Symbol
        smaCustomData = SimpleMovingAverage(1)
        self.RegisterIndicator(symbolCustom, smaCustomData, timedelta(minutes=1), lambda bar: self.SetSelectorCalled(4) and bar.Volume)
        self._indicators.append(smaCustomData)
        smaCustomData2 = SimpleMovingAverage(1)
        self.RegisterIndicator(symbolCustom, smaCustomData2, Resolution.Minute)
        self._indicators.append(smaCustomData2)
        smaCustomData3 = SimpleMovingAverage(1)
        consolidator = self.ResolveConsolidator(symbolCustom, timedelta(minutes=1))
        self.RegisterIndicator(symbolCustom, smaCustomData3, consolidator, lambda bar: self.SetSelectorCalled(5) and bar.Volume)
        self._indicators.append(smaCustomData3)

    def SetSelectorCalled(self, position):
        if False:
            for i in range(10):
                print('nop')
        self._selectorCalled[position] = True
        return True

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            self.SetHoldings(self._symbol, 0.5)

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if any((not wasCalled for wasCalled in self._selectorCalled)):
            raise ValueError('All selectors should of been called')
        if any((not indicator.IsReady for indicator in self._indicators)):
            raise ValueError('All indicators should be ready')
        self.Log(f'Total of {len(self._indicators)} are ready')

class CustomIndicator(PythonIndicator):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.Name = 'Jose'
        self.Value = 0

    def Update(self, input):
        if False:
            print('Hello World!')
        self.Value = input.Ask.High
        return True