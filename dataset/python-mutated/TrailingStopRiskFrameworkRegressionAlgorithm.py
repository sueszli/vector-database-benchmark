from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Risk.TrailingStopRiskManagementModel import TrailingStopRiskManagementModel

class TrailingStopRiskFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):
    """Show example of how to use the TrailingStopRiskManagementModel"""

    def Initialize(self):
        if False:
            print('Hello World!')
        super().Initialize()
        self.SetUniverseSelection(ManualUniverseSelectionModel([Symbol.Create('AAPL', SecurityType.Equity, Market.USA)]))
        self.SetRiskManagement(TrailingStopRiskManagementModel(0.01))