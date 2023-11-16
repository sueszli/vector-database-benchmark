from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Alphas.EmaCrossAlphaModel import EmaCrossAlphaModel

class EmaCrossAlphaModelFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        super().Initialize()
        self.SetAlpha(EmaCrossAlphaModel())

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        pass