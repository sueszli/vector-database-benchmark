from AlgorithmImports import *
from FundamentalRegressionAlgorithm import FundamentalRegressionAlgorithm

class AsynchronousUniverseRegressionAlgorithm(FundamentalRegressionAlgorithm):

    def Initialize(self):
        if False:
            return 10
        super().Initialize()
        self.UniverseSettings.Asynchronous = True