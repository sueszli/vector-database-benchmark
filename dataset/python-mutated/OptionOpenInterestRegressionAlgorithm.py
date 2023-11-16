from AlgorithmImports import *

class OptionOpenInterestRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetCash(1000000)
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 6)
        option = self.AddOption('TWX')
        option.SetFilter(-10, 10, timedelta(0), timedelta(365 * 2))
        self.SetBenchmark('TWX')

    def OnData(self, slice):
        if False:
            return 10
        if not self.Portfolio.Invested:
            for chain in slice.OptionChains:
                for contract in chain.Value:
                    if float(contract.Symbol.ID.StrikePrice) == 72.5 and contract.Symbol.ID.OptionRight == OptionRight.Call and (contract.Symbol.ID.Date == datetime(2016, 1, 15)):
                        history = self.History(OpenInterest, contract.Symbol, timedelta(1))['openinterest']
                        if len(history.index) == 0 or 0 in history.values:
                            raise ValueError('Regression test failed: open interest history request is empty')
                        security = self.Securities[contract.Symbol]
                        openInterestCache = security.Cache.GetData[OpenInterest]()
                        if openInterestCache == None:
                            raise ValueError("Regression test failed: current open interest isn't in the security cache")
                        if slice.Time.date() == datetime(2014, 6, 5).date() and (contract.OpenInterest != 50 or security.OpenInterest != 50):
                            raise ValueError('Regression test failed: current open interest was not correctly loaded and is not equal to 50')
                        if slice.Time.date() == datetime(2014, 6, 6).date() and (contract.OpenInterest != 70 or security.OpenInterest != 70):
                            raise ValueError('Regression test failed: current open interest was not correctly loaded and is not equal to 70')
                        if slice.Time.date() == datetime(2014, 6, 6).date():
                            self.MarketOrder(contract.Symbol, 1)
                            self.MarketOnCloseOrder(contract.Symbol, -1)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Log(str(orderEvent))