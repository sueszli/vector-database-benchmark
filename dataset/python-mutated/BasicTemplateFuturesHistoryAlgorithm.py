from AlgorithmImports import *

class BasicTemplateFuturesHistoryAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 9)
        self.SetCash(1000000)
        extendedMarketHours = self.GetExtendedMarketHours()
        futureES = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute, extendedMarketHours=extendedMarketHours)
        futureES.SetFilter(timedelta(0), timedelta(182))
        futureGC = self.AddFuture(Futures.Metals.Gold, Resolution.Minute, extendedMarketHours=extendedMarketHours)
        futureGC.SetFilter(timedelta(0), timedelta(182))
        self.SetBenchmark(lambda x: 1000000)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.MakeHistoryCall)
        self._successCount = 0

    def MakeHistoryCall(self):
        if False:
            return 10
        history = self.History(self.Securities.keys(), 10, Resolution.Minute)
        if len(history) < 10:
            raise Exception(f'Empty history at {self.Time}')
        self._successCount += 1

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self._successCount < self.GetExpectedHistoryCallCount():
            raise Exception(f'Scheduled Event did not assert history call as many times as expected: {self._successCount}/49')

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if self.Portfolio.Invested:
            return
        for chain in slice.FutureChains:
            for contract in chain.Value:
                self.Log(f'{contract.Symbol.Value},' + f'Bid={contract.BidPrice} ' + f'Ask={contract.AskPrice} ' + f'Last={contract.LastPrice} ' + f'OI={contract.OpenInterest}')

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        for change in changes.AddedSecurities:
            history = self.History(change.Symbol, 10, Resolution.Minute).sort_index(level='time', ascending=False)[:3]
            for (index, row) in history.iterrows():
                self.Log(f'History: {index[1]} : {index[2]:%m/%d/%Y %I:%M:%S %p} > {row.close}')

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        self.Log(f'{orderEvent}')

    def GetExtendedMarketHours(self):
        if False:
            while True:
                i = 10
        return False

    def GetExpectedHistoryCallCount(self):
        if False:
            for i in range(10):
                print('nop')
        return 42