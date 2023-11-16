from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Risk.MaximumDrawdownPercentPerSecurity import MaximumDrawdownPercentPerSecurity

class MaximumDrawdownPercentPerSecurityFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        super().Initialize()
        self.SetUniverseSelection(ManualUniverseSelectionModel(Symbol.Create('AAPL', SecurityType.Equity, Market.USA)))
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.004))