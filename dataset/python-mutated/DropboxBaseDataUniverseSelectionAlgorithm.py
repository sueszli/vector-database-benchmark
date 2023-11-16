from AlgorithmImports import *
from System.Collections.Generic import List

class DropboxBaseDataUniverseSelectionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2017, 7, 4)
        self.SetEndDate(2018, 7, 4)
        self.AddUniverse(StockDataSource, 'my-stock-data-source', self.stockDataSource)

    def stockDataSource(self, data):
        if False:
            print('Hello World!')
        list = []
        for item in data:
            for symbol in item['Symbols']:
                list.append(symbol)
        return list

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if slice.Bars.Count == 0:
            return
        if self._changes is None:
            return
        self.Liquidate()
        percentage = 1 / slice.Bars.Count
        for tradeBar in slice.Bars.Values:
            self.SetHoldings(tradeBar.Symbol, percentage)
        self._changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        self._changes = changes

class StockDataSource(PythonData):

    def GetSource(self, config, date, isLiveMode):
        if False:
            while True:
                i = 10
        url = 'https://www.dropbox.com/s/2l73mu97gcehmh7/daily-stock-picker-live.csv?dl=1' if isLiveMode else 'https://www.dropbox.com/s/ae1couew5ir3z9y/daily-stock-picker-backtest.csv?dl=1'
        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        if not (line.strip() and line[0].isdigit()):
            return None
        stocks = StockDataSource()
        stocks.Symbol = config.Symbol
        csv = line.split(',')
        if isLiveMode:
            stocks.Time = date
            stocks['Symbols'] = csv
        else:
            stocks.Time = datetime.strptime(csv[0], '%Y%m%d')
            stocks['Symbols'] = csv[1:]
        return stocks