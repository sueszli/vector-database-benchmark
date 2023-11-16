from AlgorithmImports import *

class IndicatorWithRenkoBarsRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 9)
        self.AddEquity('SPY')
        self.AddEquity('AIG')
        spyRenkoConsolidator = RenkoConsolidator(0.1)
        spyRenkoConsolidator.DataConsolidated += self.OnSPYDataConsolidated
        aigRenkoConsolidator = RenkoConsolidator(0.05)
        aigRenkoConsolidator.DataConsolidated += self.OnAIGDataConsolidated
        self.SubscriptionManager.AddConsolidator('SPY', spyRenkoConsolidator)
        self.SubscriptionManager.AddConsolidator('AIG', aigRenkoConsolidator)
        self.mi = MassIndex('MassIndex', 9, 25)
        self.wasi = WilderAccumulativeSwingIndex('WilderAccumulativeSwingIndex', 8)
        self.wsi = WilderSwingIndex('WilderSwingIndex', 8)
        self.b = Beta('Beta', 3, 'AIG', 'SPY')
        self.indicators = [self.mi, self.wasi, self.wsi, self.b]

    def OnSPYDataConsolidated(self, sender, renkoBar):
        if False:
            return 10
        self.mi.Update(renkoBar)
        self.wasi.Update(renkoBar)
        self.wsi.Update(renkoBar)
        self.b.Update(renkoBar)

    def OnAIGDataConsolidated(self, sender, renkoBar):
        if False:
            for i in range(10):
                print('nop')
        self.b.Update(renkoBar)

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        for indicator in self.indicators:
            if not indicator.IsReady:
                raise Exception(f'{indicator.Name} indicator should be ready')
            elif indicator.Current.Value == 0:
                raise Exception(f'The current value of the {indicator.Name} indicator should be different than zero')