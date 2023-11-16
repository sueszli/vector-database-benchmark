from AlgorithmImports import *

class TickDataFilteringAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)
        self.SetCash(25000)
        spy = self.AddEquity('SPY', Resolution.Tick)
        spy.SetDataFilter(TickExchangeDataFilter(self))

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not data.ContainsKey('SPY'):
            return
        spyTickList = data['SPY']
        for tick in spyTickList:
            self.Debug(tick.Exchange)
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

class TickExchangeDataFilter(SecurityDataFilter):

    def __init__(self, algo: IAlgorithm):
        if False:
            i = 10
            return i + 15
        self.algo = algo
        super().__init__()

    def Filter(self, asset: Security, data: BaseData):
        if False:
            return 10
        if isinstance(data, Tick):
            if data.Exchange == str(Exchange.ARCA):
                return True
        return False