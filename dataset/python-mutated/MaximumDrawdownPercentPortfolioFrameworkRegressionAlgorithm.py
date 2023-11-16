from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Risk.CompositeRiskManagementModel import CompositeRiskManagementModel
from Risk.MaximumDrawdownPercentPortfolio import MaximumDrawdownPercentPortfolio

class MaximumDrawdownPercentPortfolioFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        super().Initialize()
        self.SetUniverseSelection(ManualUniverseSelectionModel(Symbol.Create('AAPL', SecurityType.Equity, Market.USA)))
        self.SetRiskManagement(CompositeRiskManagementModel(MaximumDrawdownPercentPortfolio(0.01), MaximumDrawdownPercentPortfolio(0.015, True)))