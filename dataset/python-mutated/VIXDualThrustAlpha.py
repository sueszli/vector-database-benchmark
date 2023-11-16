from AlgorithmImports import *

class VIXDualThrustAlpha(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.k1 = 0.63
        self.k2 = 0.63
        self.rangePeriod = 20
        self.consolidatorBars = 30
        self.SetStartDate(2018, 10, 1)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.UniverseSettings.Resolution = Resolution.Minute
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        resolutionInTimeSpan = Extensions.ToTimeSpan(self.UniverseSettings.Resolution)
        warmUpTimeSpan = Time.Multiply(resolutionInTimeSpan, self.consolidatorBars)
        self.SetWarmUp(warmUpTimeSpan)
        self.SetAlpha(DualThrustAlphaModel(self.k1, self.k2, self.rangePeriod, self.UniverseSettings.Resolution, self.consolidatorBars))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.03))

class DualThrustAlphaModel(AlphaModel):
    """Alpha model that uses dual-thrust strategy to create insights
    https://medium.com/@FMZ_Quant/dual-thrust-trading-strategy-2cc74101a626
    or here:
    https://www.quantconnect.com/tutorials/strategy-library/dual-thrust-trading-algorithm"""

    def __init__(self, k1, k2, rangePeriod, resolution=Resolution.Daily, barsToConsolidate=1):
        if False:
            print('Hello World!')
        'Initializes a new instance of the class\n        Args:\n            k1: Coefficient for upper band\n            k2: Coefficient for lower band\n            rangePeriod: Amount of last bars to calculate the range\n            resolution: The resolution of data sent into the EMA indicators\n            barsToConsolidate: If we want alpha to work on trade bars whose length is different\n                from the standard resolution - 1m 1h etc. - we need to pass this parameters along\n                with proper data resolution'
        self.k1 = k1
        self.k2 = k2
        self.rangePeriod = rangePeriod
        self.symbolDataBySymbol = dict()
        resolutionInTimeSpan = Extensions.ToTimeSpan(resolution)
        self.consolidatorTimeSpan = Time.Multiply(resolutionInTimeSpan, barsToConsolidate)
        self.period = timedelta(5)

    def Update(self, algorithm, data):
        if False:
            print('Hello World!')
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if not symbolData.IsReady:
                continue
            holding = algorithm.Portfolio[symbol]
            price = algorithm.Securities[symbol].Price
            if price > symbolData.UpperLine and (not holding.IsLong):
                insightCloseTimeUtc = algorithm.UtcTime + self.period
                insights.append(Insight.Price(symbol, insightCloseTimeUtc, InsightDirection.Up))
            if price < symbolData.LowerLine and (not holding.IsShort):
                insightCloseTimeUtc = algorithm.UtcTime + self.period
                insights.append(Insight.Price(symbol, insightCloseTimeUtc, InsightDirection.Down))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        for symbol in [x.Symbol for x in changes.AddedSecurities]:
            if symbol not in self.symbolDataBySymbol:
                symbolData = self.SymbolData(symbol, self.k1, self.k2, self.rangePeriod, self.consolidatorTimeSpan)
                self.symbolDataBySymbol[symbol] = symbolData
                algorithm.SubscriptionManager.AddConsolidator(symbol, symbolData.GetConsolidator())
        for symbol in [x.Symbol for x in changes.RemovedSecurities]:
            symbolData = self.symbolDataBySymbol.pop(symbol, None)
            if symbolData is None:
                algorithm.Error('Unable to remove data from collection: DualThrustAlphaModel')
            else:
                algorithm.SubscriptionManager.RemoveConsolidator(symbol, symbolData.GetConsolidator())

    class SymbolData:
        """Contains data specific to a symbol required by this model"""

        def __init__(self, symbol, k1, k2, rangePeriod, consolidatorResolution):
            if False:
                for i in range(10):
                    print('nop')
            self.Symbol = symbol
            self.rangeWindow = RollingWindow[TradeBar](rangePeriod)
            self.consolidator = TradeBarConsolidator(consolidatorResolution)

            def onDataConsolidated(sender, consolidated):
                if False:
                    while True:
                        i = 10
                self.rangeWindow.Add(consolidated)
                if self.rangeWindow.IsReady:
                    hh = max([x.High for x in self.rangeWindow])
                    hc = max([x.Close for x in self.rangeWindow])
                    lc = min([x.Close for x in self.rangeWindow])
                    ll = min([x.Low for x in self.rangeWindow])
                    range = max([hh - lc, hc - ll])
                    self.UpperLine = consolidated.Close + k1 * range
                    self.LowerLine = consolidated.Close - k2 * range
            self.consolidator.DataConsolidated += onDataConsolidated

        def GetConsolidator(self):
            if False:
                return 10
            return self.consolidator

        @property
        def IsReady(self):
            if False:
                while True:
                    i = 10
            return self.rangeWindow.IsReady