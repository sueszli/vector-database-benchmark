from AlgorithmImports import *

class BasicTemplateIntrinioEconomicData(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2013, 12, 31)
        self.SetCash(100000)
        IntrinioConfig.SetUserAndPassword('intrinio-username', 'intrinio-password')
        IntrinioConfig.SetTimeIntervalBetweenCalls(timedelta(minutes=1))
        self.uso = self.AddEquity('USO', Resolution.Daily).Symbol
        self.Securities[self.uso].SetLeverage(2)
        self.bno = self.AddEquity('BNO', Resolution.Daily).Symbol
        self.Securities[self.bno].SetLeverage(2)
        self.AddData(IntrinioEconomicData, '$DCOILWTICO', Resolution.Daily)
        self.AddData(IntrinioEconomicData, '$DCOILBRENTEU', Resolution.Daily)
        self.emaWti = self.EMA('$DCOILWTICO', 10)

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if slice.ContainsKey('$DCOILBRENTEU') or slice.ContainsKey('$DCOILWTICO'):
            spread = slice['$DCOILBRENTEU'].Value - slice['$DCOILWTICO'].Value
        else:
            return
        if spread > 0 and (not self.Portfolio[self.bno].IsLong) or (spread < 0 and (not self.Portfolio[self.uso].IsShort)):
            self.SetHoldings(self.bno, 0.25 * sign(spread))
            self.SetHoldings(self.uso, -0.25 * sign(spread))