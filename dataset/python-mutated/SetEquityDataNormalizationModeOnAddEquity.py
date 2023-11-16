from AlgorithmImports import *

class SetEquityDataNormalizationModeOnAddEquity(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)
        spyNormalizationMode = DataNormalizationMode.Raw
        ibmNormalizationMode = DataNormalizationMode.Adjusted
        aigNormalizationMode = DataNormalizationMode.TotalReturn
        self._priceRanges = {}
        spyEquity = self.AddEquity('SPY', Resolution.Minute, dataNormalizationMode=spyNormalizationMode)
        self.CheckEquityDataNormalizationMode(spyEquity, spyNormalizationMode)
        self._priceRanges[spyEquity] = (167.28, 168.37)
        ibmEquity = self.AddEquity('IBM', Resolution.Minute, dataNormalizationMode=ibmNormalizationMode)
        self.CheckEquityDataNormalizationMode(ibmEquity, ibmNormalizationMode)
        self._priceRanges[ibmEquity] = (135.864131052, 136.819606508)
        aigEquity = self.AddEquity('AIG', Resolution.Minute, dataNormalizationMode=aigNormalizationMode)
        self.CheckEquityDataNormalizationMode(aigEquity, aigNormalizationMode)
        self._priceRanges[aigEquity] = (48.73, 49.1)

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        for (equity, (minExpectedPrice, maxExpectedPrice)) in self._priceRanges.items():
            if equity.HasData and (equity.Price < minExpectedPrice or equity.Price > maxExpectedPrice):
                raise Exception(f'{equity.Symbol}: Price {equity.Price} is out of expected range [{minExpectedPrice}, {maxExpectedPrice}]')

    def CheckEquityDataNormalizationMode(self, equity, expectedNormalizationMode):
        if False:
            return 10
        subscriptions = [x for x in self.SubscriptionManager.Subscriptions if x.Symbol == equity.Symbol]
        if any([x.DataNormalizationMode != expectedNormalizationMode for x in subscriptions]):
            raise Exception(f'Expected {equity.Symbol} to have data normalization mode {expectedNormalizationMode} but was {subscriptions[0].DataNormalizationMode}')