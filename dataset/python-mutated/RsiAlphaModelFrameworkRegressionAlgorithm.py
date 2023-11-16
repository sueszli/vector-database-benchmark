from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Alphas.RsiAlphaModel import RsiAlphaModel

class RsiAlphaModelFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        super().Initialize()
        self.SetAlpha(RsiAlphaModel())

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        consolidator_count = sum([s.Consolidators.Count for s in self.SubscriptionManager.Subscriptions])
        if consolidator_count > 0:
            raise Exception(f'The number of consolidators should be zero. Actual: {consolidator_count}')