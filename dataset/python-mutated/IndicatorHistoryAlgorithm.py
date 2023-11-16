from AlgorithmImports import *

class IndicatorHistoryAlgorithm(QCAlgorithm):
    """Demonstration algorithm of indicators history window usage."""

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2014, 12, 31)
        self.SetCash(25000)
        self.symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.bollingerBands = self.BB(self.symbol, 20, 2.0, resolution=Resolution.Daily)
        self.bollingerBands.Window.Size = 20
        self.bollingerBands.MiddleBand.Window.Size = 20

    def OnData(self, slice: Slice):
        if False:
            for i in range(10):
                print('nop')
        if not self.bollingerBands.Window.IsReady:
            return
        self.Log(f'Current BB value: {self.bollingerBands[0].EndTime} - {self.bollingerBands[0].Value}')
        self.Log(f'Oldest BB value: {self.bollingerBands[self.bollingerBands.Window.Count - 1].EndTime} - {self.bollingerBands[self.bollingerBands.Window.Count - 1].Value}')
        for dataPoint in self.bollingerBands:
            self.Log(f'BB @{dataPoint.EndTime}: {dataPoint.Value}')
        middleBand = self.bollingerBands.MiddleBand
        self.Log(f'Current BB Middle Band value: {middleBand[0].EndTime} - {middleBand[0].Value}')
        self.Log(f'Oldest BB Middle Band value: {middleBand[middleBand.Window.Count - 1].EndTime} - {middleBand[middleBand.Window.Count - 1].Value}')
        for dataPoint in middleBand:
            self.Log(f'BB Middle Band @{dataPoint.EndTime}: {dataPoint.Value}')
        self.Quit()