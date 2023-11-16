from AlgorithmImports import *

class VolumeRenkoConsolidatorAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.sma = SimpleMovingAverage(10)
        self.tick_consolidated = False
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        self.tradebar_volume_consolidator = VolumeRenkoConsolidator(1000000)
        self.tradebar_volume_consolidator.DataConsolidated += self.OnSPYDataConsolidated
        self.ibm = self.AddEquity('IBM', Resolution.Tick).Symbol
        self.tick_volume_consolidator = VolumeRenkoConsolidator(1000000)
        self.tick_volume_consolidator.DataConsolidated += self.OnIBMDataConsolidated
        history = self.History[TradeBar](self.spy, 1000, Resolution.Minute)
        for bar in history:
            self.tradebar_volume_consolidator.Update(bar)

    def OnSPYDataConsolidated(self, sender, bar):
        if False:
            i = 10
            return i + 15
        self.sma.Update(bar.EndTime, bar.Value)
        self.Debug(f'SPY {bar.Time} to {bar.EndTime} :: O:{bar.Open} H:{bar.High} L:{bar.Low} C:{bar.Close} V:{bar.Volume}')
        if bar.Volume != 1000000:
            raise Exception('Volume of consolidated bar does not match set value!')

    def OnIBMDataConsolidated(self, sender, bar):
        if False:
            while True:
                i = 10
        self.Debug(f'IBM {bar.Time} to {bar.EndTime} :: O:{bar.Open} H:{bar.High} L:{bar.Low} C:{bar.Close} V:{bar.Volume}')
        if bar.Volume != 1000000:
            raise Exception('Volume of consolidated bar does not match set value!')
        self.tick_consolidated = True

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if slice.Bars.ContainsKey(self.spy):
            self.tradebar_volume_consolidator.Update(slice.Bars[self.spy])
        if slice.Ticks.ContainsKey(self.ibm):
            for tick in slice.Ticks[self.ibm]:
                self.tick_volume_consolidator.Update(tick)
        if self.sma.IsReady and self.sma.Current.Value < self.Securities[self.spy].Price:
            self.SetHoldings(self.spy, 1)
        else:
            self.SetHoldings(self.spy, 0)

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.tick_consolidated:
            raise Exception('Tick consolidator was never been called')