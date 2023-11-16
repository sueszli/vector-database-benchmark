from datetime import datetime
from AlgorithmImports import *

class CustomDataUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 3, 31)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(CoarseFundamental, 'custom-data-universe', self.Selection)
        self._selectionTime = [datetime(2014, 3, 24), datetime(2014, 3, 25), datetime(2014, 3, 26), datetime(2014, 3, 27), datetime(2014, 3, 28), datetime(2014, 3, 29), datetime(2014, 3, 30), datetime(2014, 3, 31)]

    def Selection(self, coarse):
        if False:
            return 10
        self.Debug(f'Universe selection called: {self.Time} Count: {len(coarse)}')
        expectedTime = self._selectionTime.pop(0)
        if expectedTime != self.Time:
            raise ValueError(f'Unexpected selection time {self.Time} expected {expectedTime}')
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        underlyingSymbols = [x.Symbol for x in sortedByDollarVolume[:10]]
        customSymbols = []
        for symbol in underlyingSymbols:
            customSymbols.append(Symbol.CreateBase(MyPyCustomData, symbol))
        return underlyingSymbols + customSymbols

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            customData = data.Get(MyPyCustomData)
            symbols = [symbol for symbol in data.Keys if symbol.SecurityType is SecurityType.Equity]
            for symbol in symbols:
                self.SetHoldings(symbol, 1 / len(symbols))
                if len([x for x in customData.Keys if x.Underlying == symbol]) == 0:
                    raise ValueError(f'Custom data was not found for symbol {symbol}')

class MyPyCustomData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        if False:
            return 10
        source = f'{Globals.DataFolder}/equity/usa/daily/{LeanData.GenerateZipFileName(config.Symbol, date, config.Resolution, config.TickType)}'
        return SubscriptionDataSource(source, SubscriptionTransportMedium.LocalFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            return 10
        csv = line.split(',')
        _scaleFactor = 1 / 10000
        custom = MyPyCustomData()
        custom.Symbol = config.Symbol
        custom.Time = datetime.strptime(csv[0], '%Y%m%d %H:%M')
        custom.Open = float(csv[1]) * _scaleFactor
        custom.High = float(csv[2]) * _scaleFactor
        custom.Low = float(csv[3]) * _scaleFactor
        custom.Close = float(csv[4]) * _scaleFactor
        custom.Value = float(csv[4]) * _scaleFactor
        custom.Period = Time.OneDay
        custom.EndTime = custom.Time + custom.Period
        return custom