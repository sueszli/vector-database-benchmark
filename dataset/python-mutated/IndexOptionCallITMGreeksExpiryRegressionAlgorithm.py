from AlgorithmImports import *

class IndexOptionCallITMGreeksExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.onDataCalls = 0
        self.invested = False
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 31)
        spx = self.AddIndex('SPX', Resolution.Minute)
        spx.VolatilityModel = StandardDeviationOfReturnsVolatilityModel(60, Resolution.Minute, timedelta(minutes=1))
        self.spx = spx.Symbol
        self.spxOption = list(self.OptionChainProvider.GetOptionContractList(self.spx, self.Time))
        self.spxOption = [i for i in self.spxOption if i.ID.StrikePrice <= 3200 and i.ID.OptionRight == OptionRight.Call and (i.ID.Date.year == 2021) and (i.ID.Date.month == 1)]
        self.spxOption = list(sorted(self.spxOption, key=lambda x: x.ID.StrikePrice, reverse=True))[0]
        self.spxOption = self.AddIndexOptionContract(self.spxOption, Resolution.Minute)
        self.spxOption.PriceModel = OptionPriceModels.BlackScholes()
        self.expectedOptionContract = Symbol.CreateOption(self.spx, Market.USA, OptionStyle.European, OptionRight.Call, 3200, datetime(2021, 1, 15))
        if self.spxOption.Symbol != self.expectedOptionContract:
            raise Exception(f'Contract {self.expectedOptionContract} was not found in the chain')

    def OnData(self, data: Slice):
        if False:
            i = 10
            return i + 15
        if self.invested or self.onDataCalls < 40:
            self.onDataCalls += 1
            return
        self.onDataCalls += 1
        if data.OptionChains.Count == 0:
            return
        if all([any([c.Symbol not in data for c in o.Contracts.Values]) for o in data.OptionChains.Values]):
            return
        if len(list(list(data.OptionChains.Values)[0].Contracts.Values)) == 0:
            raise Exception(f'No contracts found in the option {list(data.OptionChains.Keys)[0]}')
        deltas = [i.Greeks.Delta for i in self.SortByMaxVolume(data)]
        gammas = [i.Greeks.Gamma for i in self.SortByMaxVolume(data)]
        lambda_ = [i.Greeks.Lambda for i in self.SortByMaxVolume(data)]
        rho = [i.Greeks.Rho for i in self.SortByMaxVolume(data)]
        theta = [i.Greeks.Theta for i in self.SortByMaxVolume(data)]
        vega = [i.Greeks.Vega for i in self.SortByMaxVolume(data)]
        if any([i for i in deltas if i == 0]):
            raise Exception('Option contract Delta was equal to zero')
        if any([i for i in gammas if i == 0]):
            raise AggregateException('Option contract Gamma was equal to zero')
        if any([i for i in lambda_ if lambda_ == 0]):
            raise AggregateException('Option contract Lambda was equal to zero')
        if any([i for i in rho if i == 0]):
            raise Exception('Option contract Rho was equal to zero')
        if any([i for i in theta if i == 0]):
            raise Exception('Option contract Theta was equal to zero')
        if any([i for i in vega if vega == 0]):
            raise AggregateException('Option contract Vega was equal to zero')
        if not self.invested:
            self.SetHoldings(list(list(data.OptionChains.Values)[0].Contracts.Values)[0].Symbol, 1)
            self.invested = True

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if self.Portfolio.Invested:
            raise Exception(f"Expected no holdings at end of algorithm, but are invested in: {', '.join(self.Portfolio.Keys)}")
        if not self.invested:
            raise Exception(f'Never checked greeks, maybe we have no option data?')

    def SortByMaxVolume(self, data: Slice):
        if False:
            print('Hello World!')
        chain = [i for i in sorted(list(data.OptionChains.Values), key=lambda x: sum([j.Volume for j in x.Contracts.Values]), reverse=True)][0]
        return chain.Contracts.Values