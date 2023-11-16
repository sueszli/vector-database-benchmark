from AlgorithmImports import *
from QuantConnect.Algorithm.CSharp import *

class NullOptionAssignmentRegressionAlgorithm(OptionAssignmentRegressionAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        super().Initialize()

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        super().OnData(data)

    def CustomSecurityInitializer(self, security):
        if False:
            print('Hello World!')
        if Extensions.IsOption(security.Symbol.SecurityType):
            security.SetOptionAssignmentModel(NullOptionAssignmentModel())