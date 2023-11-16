from AlgorithmImports import *
import base64

class DropboxUniverseSelectionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2017, 7, 4)
        self.SetEndDate(2018, 7, 4)
        self.backtestSymbolsPerDay = {}
        self.current_universe = []
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse('my-dropbox-universe', self.selector)

    def selector(self, date):
        if False:
            print('Hello World!')
        if self.LiveMode:
            str = self.Download('https://www.dropbox.com/s/2l73mu97gcehmh7/daily-stock-picker-live.csv?dl=1')
            self.current_universe = str.split(',') if len(str) > 0 else self.current_universe
            return self.current_universe
        if len(self.backtestSymbolsPerDay) == 0:
            byteKey = base64.b64encode('UserName:Password'.encode('ASCII'))
            headers = {'Authorization': f"Basic ({byteKey.decode('ASCII')})"}
            str = self.Download('https://www.dropbox.com/s/ae1couew5ir3z9y/daily-stock-picker-backtest.csv?dl=1', headers)
            for line in str.splitlines():
                data = line.split(',')
                self.backtestSymbolsPerDay[data[0]] = data[1:]
        index = date.strftime('%Y%m%d')
        self.current_universe = self.backtestSymbolsPerDay.get(index, self.current_universe)
        return self.current_universe

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        if slice.Bars.Count == 0:
            return
        if self.changes is None:
            return
        self.Liquidate()
        percentage = 1 / slice.Bars.Count
        for tradeBar in slice.Bars.Values:
            self.SetHoldings(tradeBar.Symbol, percentage)
        self.changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        self.changes = changes