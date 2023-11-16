from AlgorithmImports import *
from QuantConnect.Data.Auxiliary import *
from QuantConnect.Lean.Engine.DataFeeds import DefaultDataProvider
_ticker = 'GOOGL'
_expectedRawPrices = [1157.93, 1158.72, 1131.97, 1114.28, 1120.15, 1114.51, 1134.89, 567.55, 571.5, 545.25, 540.63]

class RawDataRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2014, 3, 25)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(100000)
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self._googl = self.AddEquity(_ticker, Resolution.Daily).Symbol
        dataProvider = DefaultDataProvider()
        mapFileProvider = LocalDiskMapFileProvider()
        mapFileProvider.Initialize(dataProvider)
        factorFileProvider = LocalDiskFactorFileProvider()
        factorFileProvider.Initialize(mapFileProvider, dataProvider)
        self._factorFile = factorFileProvider.Get(self._googl)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested:
            self.SetHoldings(self._googl, 1)
        if data.Bars.ContainsKey(self._googl):
            googlData = data.Bars[self._googl]
            expectedRawPrice = _expectedRawPrices.pop(0)
            if expectedRawPrice != googlData.Close:
                dayFactor = self._factorFile.GetPriceScaleFactor(googlData.Time)
                probableRawPrice = googlData.Close / dayFactor
                raise Exception('Close price was incorrect; it appears to be the adjusted value' if expectedRawPrice == probableRawPrice else 'Close price was incorrect; Data may have changed.')