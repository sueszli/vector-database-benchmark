from AlgorithmImports import *

class FinancialAdvisorDemoAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.symbol = self.AddEquity('SPY', Resolution.Second).Symbol
        self.DefaultOrderProperties = InteractiveBrokersOrderProperties()
        self.DefaultOrderProperties.FaGroup = 'TestGroupEQ'
        self.DefaultOrderProperties.FaMethod = 'EqualQuantity'

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)