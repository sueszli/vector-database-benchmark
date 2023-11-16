from AlgorithmImports import *

class CustomVolatilityModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2015, 7, 15)
        self.SetCash(100000)
        self.equity = self.AddEquity('SPY', Resolution.Daily)
        self.equity.SetVolatilityModel(CustomVolatilityModel(10))

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested and self.equity.VolatilityModel.Volatility > 0:
            self.SetHoldings('SPY', 1)

class CustomVolatilityModel:

    def __init__(self, periods):
        if False:
            print('Hello World!')
        self.lastUpdate = datetime.min
        self.lastPrice = 0
        self.needsUpdate = False
        self.periodSpan = timedelta(1)
        self.window = RollingWindow[float](periods)
        self.Volatility = 0

    def Update(self, security, data):
        if False:
            print('Hello World!')
        timeSinceLastUpdate = data.EndTime - self.lastUpdate
        if timeSinceLastUpdate >= self.periodSpan and data.Price > 0:
            if self.lastPrice > 0:
                self.window.Add(float(data.Price / self.lastPrice) - 1.0)
                self.needsUpdate = self.window.IsReady
            self.lastUpdate = data.EndTime
            self.lastPrice = data.Price
        if self.window.Count < 2:
            self.Volatility = 0
            return
        if self.needsUpdate:
            self.needsUpdate = False
            std = np.std([x for x in self.window])
            self.Volatility = std * np.sqrt(252.0)

    def GetHistoryRequirements(self, security, utcTime):
        if False:
            for i in range(10):
                print('nop')
        return None