from AlgorithmImports import *
from QuantConnect.Algorithm.CSharp import *

class CustomOptionAssignmentRegressionAlgorithm(OptionAssignmentRegressionAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        super().Initialize()

    def CustomSecurityInitializer(self, security):
        if False:
            i = 10
            return i + 15
        if Extensions.IsOption(security.Symbol.SecurityType):
            security.SetOptionAssignmentModel(PyCustomOptionAssignmentModel(0.1))

    def OnData(self, data):
        if False:
            print('Hello World!')
        super().OnData(data)

class PyCustomOptionAssignmentModel(DefaultOptionAssignmentModel):

    def __init__(self, requiredInTheMoneyPercent):
        if False:
            return 10
        super().__init__(requiredInTheMoneyPercent)

    def GetAssignment(self, parameters):
        if False:
            return 10
        result = super().GetAssignment(parameters)
        result.Tag = 'Custom Option Assignment'
        return result