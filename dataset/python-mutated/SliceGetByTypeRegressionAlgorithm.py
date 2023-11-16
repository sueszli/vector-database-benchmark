from AlgorithmImports import *
TradeFlag = False

class SliceGetByTypeRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.AddEquity('SPY', Resolution.Minute)
        self.SetAlpha(TestAlphaModel())

    def OnData(self, data):
        if False:
            print('Hello World!')
        if 'SPY' in data:
            tb = data.Get(TradeBar)['SPY']
            global TradeFlag
            if not self.Portfolio.Invested and TradeFlag:
                self.SetHoldings('SPY', 1)

class TestAlphaModel(AlphaModel):

    def Update(self, algorithm, data):
        if False:
            i = 10
            return i + 15
        insights = []
        if 'SPY' in data:
            tb = data.Get(TradeBar)['SPY']
            global TradeFlag
            TradeFlag = True
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            i = 10
            return i + 15
        return