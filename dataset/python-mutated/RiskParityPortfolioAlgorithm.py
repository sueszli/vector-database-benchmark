from AlgorithmImports import *
from Portfolio.RiskParityPortfolioConstructionModel import *

class RiakParityPortfolioAlgorithm(QCAlgorithm):
    """Example algorithm of using RiskParityPortfolioConstructionModel"""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2021, 2, 21)
        self.SetEndDate(2021, 3, 30)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetMarketPrice(self.GetLastKnownPrice(security)))
        self.AddEquity('SPY', Resolution.Daily)
        self.AddEquity('AAPL', Resolution.Daily)
        self.AddAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(RiskParityPortfolioConstructionModel())