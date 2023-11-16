from AlgorithmImports import *

class FutureOptionBuySellCallIntradayRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2020, 1, 5)
        self.SetEndDate(2020, 6, 30)
        self.es20h20 = self.AddFutureContract(Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, datetime(2020, 3, 20)), Resolution.Minute).Symbol
        self.es19m20 = self.AddFutureContract(Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, datetime(2020, 6, 19)), Resolution.Minute).Symbol
        self.esOptions = [self.AddFutureOptionContract(i, Resolution.Minute).Symbol for i in self.OptionChainProvider.GetOptionContractList(self.es19m20, self.Time) + self.OptionChainProvider.GetOptionContractList(self.es20h20, self.Time) if i.ID.StrikePrice == 3200.0 and i.ID.OptionRight == OptionRight.Call]
        self.expectedContracts = [Symbol.CreateOption(self.es20h20, Market.CME, OptionStyle.American, OptionRight.Call, 3200.0, datetime(2020, 3, 20)), Symbol.CreateOption(self.es19m20, Market.CME, OptionStyle.American, OptionRight.Call, 3200.0, datetime(2020, 6, 19))]
        for esOption in self.esOptions:
            if esOption not in self.expectedContracts:
                raise AssertionError(f'Contract {esOption} was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.es19m20, 1), self.ScheduleCallbackBuy)

    def ScheduleCallbackBuy(self):
        if False:
            return 10
        self.MarketOrder(self.esOptions[0], 1)
        self.MarketOrder(self.esOptions[1], -1)

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if self.Portfolio.Invested:
            raise AssertionError(f"Expected no holdings at end of algorithm, but are invested in: {', '.join([str(i.ID) for i in self.Portfolio.Keys])}")