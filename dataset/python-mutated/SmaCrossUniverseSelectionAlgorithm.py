from AlgorithmImports import *

class SmaCrossUniverseSelectionAlgorithm(QCAlgorithm):
    """Provides an example where WarmUpIndicator method is used to warm up indicators
    after their security is added and before (Universe Selection scenario)"""
    count = 10
    tolerance = 0.01
    targetPercent = 1 / count
    averages = dict()

    def Initialize(self):
        if False:
            return 10
        self.UniverseSettings.Leverage = 2
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2019, 1, 1)
        self.SetCash(1000000)
        self.EnableAutomaticIndicatorWarmUp = True
        ibm = self.AddEquity('IBM', Resolution.Tick).Symbol
        ibmSma = self.SMA(ibm, 40)
        self.Log(f'{ibmSma.Name}: {ibmSma.Current.Time} - {ibmSma}. IsReady? {ibmSma.IsReady}')
        spy = self.AddEquity('SPY', Resolution.Hour).Symbol
        spySma = self.SMA(spy, 10)
        spyAtr = self.ATR(spy, 10)
        spyVwap = self.VWAP(spy, 10)
        self.Log(f'SPY    - Is ready? SMA: {spySma.IsReady}, ATR: {spyAtr.IsReady}, VWAP: {spyVwap.IsReady}')
        eur = self.AddForex('EURUSD', Resolution.Hour).Symbol
        eurSma = self.SMA(eur, 20, Resolution.Daily)
        eurAtr = self.ATR(eur, 20, MovingAverageType.Simple, Resolution.Daily)
        self.Log(f'EURUSD - Is ready? SMA: {eurSma.IsReady}, ATR: {eurAtr.IsReady}')
        self.AddUniverse(self.CoarseSmaSelector)
        self.SetWarmUp(10)

    def CoarseSmaSelector(self, coarse):
        if False:
            i = 10
            return i + 15
        score = dict()
        for cf in coarse:
            if not cf.HasFundamentalData:
                continue
            symbol = cf.Symbol
            price = cf.AdjustedPrice
            avg = self.averages.setdefault(symbol, SimpleMovingAverage(100))
            self.WarmUpIndicator(symbol, avg, Resolution.Daily)
            if avg.Update(cf.EndTime, price):
                value = avg.Current.Value
                if value > price * self.tolerance:
                    score[symbol] = (value - price) / ((value + price) / 2)
        sortedScore = sorted(score.items(), key=lambda kvp: kvp[1], reverse=True)
        return [x[0] for x in sortedScore[:self.count]]

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, self.targetPercent)