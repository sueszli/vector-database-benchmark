from AlgorithmImports import *

class IndexOptionBuySellCallIntradayRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 31)
        spx = self.AddIndex('SPX', Resolution.Minute).Symbol
        spxOptions = list(sorted([self.AddIndexOptionContract(i, Resolution.Minute).Symbol for i in self.OptionChainProvider.GetOptionContractList(spx, self.Time) if (i.ID.StrikePrice == 3700 or i.ID.StrikePrice == 3800) and i.ID.OptionRight == OptionRight.Call and (i.ID.Date.year == 2021) and (i.ID.Date.month == 1)], key=lambda x: x.ID.StrikePrice))
        expectedContract3700 = Symbol.CreateOption(spx, Market.USA, OptionStyle.European, OptionRight.Call, 3700, datetime(2021, 1, 15))
        expectedContract3800 = Symbol.CreateOption(spx, Market.USA, OptionStyle.European, OptionRight.Call, 3800, datetime(2021, 1, 15))
        if len(spxOptions) != 2:
            raise Exception(f'Expected 2 index options symbols from chain provider, found {spxOptions.Count}')
        if spxOptions[0] != expectedContract3700:
            raise Exception(f'Contract {expectedContract3700} was not found in the chain, found instead: {spxOptions[0]}')
        if spxOptions[1] != expectedContract3800:
            raise Exception(f'Contract {expectedContract3800} was not found in the chain, found instead: {spxOptions[1]}')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(spx, 1), lambda : self.AfterMarketOpenTrade(spxOptions))
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.Noon, lambda : self.Liquidate())

    def AfterMarketOpenTrade(self, spxOptions):
        if False:
            print('Hello World!')
        self.MarketOrder(spxOptions[0], 1)
        self.MarketOrder(spxOptions[1], -1)

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.Portfolio.Invested:
            raise Exception(f"Expected no holdings at end of algorithm, but are invested in: {', '.join(self.Portfolio.Keys)}")