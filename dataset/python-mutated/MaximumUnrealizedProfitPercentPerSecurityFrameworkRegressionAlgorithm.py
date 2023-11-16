from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Risk.MaximumUnrealizedProfitPercentPerSecurity import MaximumUnrealizedProfitPercentPerSecurity

class MaximumUnrealizedProfitPercentPerSecurityFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        super().Initialize()
        self.SetUniverseSelection(ManualUniverseSelectionModel(Symbol.Create('AAPL', SecurityType.Equity, Market.USA)))
        self.SetRiskManagement(MaximumUnrealizedProfitPercentPerSecurity(0.004))