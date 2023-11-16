"""
    This Alpha Model uses Wells Fargo 30-year Fixed Rate Mortgage data from Quandl to 
    generate Insights about the movement of Real Estate ETFs. Mortgage rates can provide information 
    regarding the general price trend of real estate, and ETFs provide good continuous-time instruments 
    to measure the impact against. Volatility in mortgage rates tends to put downward pressure on real 
    estate prices, whereas stable mortgage rates, regardless of true rate, lead to stable or higher real
    estate prices. This Alpha model seeks to take advantage of this correlation by emitting insights
    based on volatility and rate deviation from its historic mean.

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
    sourced so the community and client funds can see an example of an alpha.
"""
from AlgorithmImports import *

class MortgageRateVolatilityAlpha(QCAlgorithmFramework):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2017, 1, 1)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        etfs = ['VNQ', 'REET', 'TAO', 'FREL', 'SRET', 'HIPS']
        symbols = [Symbol.Create(etf, SecurityType.Equity, Market.USA) for etf in etfs]
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(MortgageRateVolatilityAlphaModel(self))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class MortgageRateVolatilityAlphaModel(AlphaModel):

    def __init__(self, algorithm, indicatorPeriod=15, insightMagnitude=0.005, deviations=2):
        if False:
            for i in range(10):
                print('nop')
        self.mortgageRate = algorithm.AddData(QuandlMortgagePriceColumns, 'WFC/PR_GOV_30YFIXEDVA_APR').Symbol
        self.indicatorPeriod = indicatorPeriod
        self.insightDuration = TimeSpan.FromDays(indicatorPeriod)
        self.insightMagnitude = insightMagnitude
        self.deviations = deviations
        self.mortgageRateStd = algorithm.STD(self.mortgageRate.Value, indicatorPeriod)
        self.mortgageRateSma = algorithm.SMA(self.mortgageRate.Value, indicatorPeriod)
        self.WarmupIndicators(algorithm)

    def Update(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        insights = []
        if self.mortgageRate not in data.Keys:
            return []
        mortgageRate = data[self.mortgageRate].Value
        deviation = self.deviations * self.mortgageRateStd.Current.Value
        sma = self.mortgageRateSma.Current.Value
        if mortgageRate < sma - deviation or mortgageRate > sma + deviation:
            insights = [Insight(security, self.insightDuration, InsightType.Price, InsightDirection.Down, self.insightMagnitude, None) for security in algorithm.ActiveSecurities.Keys if security != self.mortgageRate]
        if mortgageRate < sma - deviation / 2 or mortgageRate > sma + deviation / 2:
            insights = [Insight(security, self.insightDuration, InsightType.Price, InsightDirection.Up, self.insightMagnitude, None) for security in algorithm.ActiveSecurities.Keys if security != self.mortgageRate]
        return insights

    def WarmupIndicators(self, algorithm):
        if False:
            for i in range(10):
                print('nop')
        history = algorithm.History(self.mortgageRate, self.indicatorPeriod, Resolution.Daily)
        for (index, row) in history.iterrows():
            self.mortgageRateStd.Update(index[1], row['value'])
            self.mortgageRateSma.Update(index[1], row['value'])

class QuandlMortgagePriceColumns(PythonQuandl):

    def __init__(self):
        if False:
            print('Hello World!')
        self.ValueColumnName = 'Value'