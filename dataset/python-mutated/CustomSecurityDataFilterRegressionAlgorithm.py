from AlgorithmImports import *

class CustomSecurityDataFilterRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCash(2500000)
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)
        security = self.AddSecurity(SecurityType.Equity, 'SPY')
        security.SetDataFilter(CustomDataFilter())
        self.dataPoints = 0

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        self.dataPoints += 1
        self.SetHoldings('SPY', 0.2)
        if self.dataPoints > 5:
            raise Exception('There should not be more than 5 data points, but there were ' + str(self.dataPoints))

class CustomDataFilter(SecurityDataFilter):

    def Filter(self, vehicle: Security, data: BaseData) -> bool:
        if False:
            while True:
                i = 10
        '\n        Skip data after 9:35am\n        '
        if data.Time >= datetime(2013, 10, 7, 9, 35, 0):
            return False
        else:
            return True