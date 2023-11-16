from AlgorithmImports import *

class BasicTemplateFuturesWithExtendedMarketAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 10)
        self.SetCash(1000000)
        self.contractSymbol = None
        self.futureSP500 = self.AddFuture(Futures.Indices.SP500EMini, extendedMarketHours=True)
        self.futureGold = self.AddFuture(Futures.Metals.Gold, extendedMarketHours=True)
        self.futureSP500.SetFilter(timedelta(0), timedelta(182))
        self.futureGold.SetFilter(0, 182)
        benchmark = self.AddEquity('SPY')
        self.SetBenchmark(benchmark.Symbol)
        seeder = FuncSecuritySeeder(self.GetLastKnownPrices)
        self.SetSecurityInitializer(lambda security: seeder.SeedSecurity(security))

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            for chain in slice.FutureChains:
                contracts = list(filter(lambda x: x.Expiry > self.Time + timedelta(90), chain.Value))
                if len(contracts) == 0:
                    continue
                front = sorted(contracts, key=lambda x: x.Expiry, reverse=True)[0]
                self.contractSymbol = front.Symbol
                self.MarketOrder(front.Symbol, 1)
        else:
            self.Liquidate()

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        buyingPowerModel = self.Securities[self.contractSymbol].BuyingPowerModel
        name = type(buyingPowerModel).__name__
        if name != 'FutureMarginModel':
            raise Exception(f'Invalid buying power model. Found: {name}. Expected: FutureMarginModel')
        initialOvernight = buyingPowerModel.InitialOvernightMarginRequirement
        maintenanceOvernight = buyingPowerModel.MaintenanceOvernightMarginRequirement
        initialIntraday = buyingPowerModel.InitialIntradayMarginRequirement
        maintenanceIntraday = buyingPowerModel.MaintenanceIntradayMarginRequirement

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        for addedSecurity in changes.AddedSecurities:
            if addedSecurity.Symbol.SecurityType == SecurityType.Future and (not addedSecurity.Symbol.IsCanonical()) and (not addedSecurity.HasData):
                raise Exception(f'Future contracts did not work up as expected: {addedSecurity.Symbol}')