from AlgorithmImports import *
from SetHoldingsMultipleTargetsRegressionAlgorithm import SetHoldingsMultipleTargetsRegressionAlgorithm

class SetHoldingsLiquidateExistingHoldingsMultipleTargetsRegressionAlgorithm(SetHoldingsMultipleTargetsRegressionAlgorithm):

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            self.SetHoldings([PortfolioTarget(self._spy, 0.8), PortfolioTarget(self._ibm, 0.2)], liquidateExistingHoldings=True)