from AlgorithmImports import *
from Portfolio.MeanReversionPortfolioConstructionModel import *

class MeanReversionPortfolioAlgorithm(QCAlgorithm):
    """Example algorithm of using MeanReversionPortfolioConstructionModel"""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2020, 9, 1)
        self.SetEndDate(2021, 2, 28)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetMarketPrice(self.GetLastKnownPrice(security)))
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in ['SPY', 'AAPL']]
        self.AddAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(1)))
        self.SetPortfolioConstruction(MeanReversionPortfolioConstructionModel())