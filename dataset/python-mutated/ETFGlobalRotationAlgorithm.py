from AlgorithmImports import *
from System.Collections.Generic import List

class ETFGlobalRotationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetCash(25000)
        self.SetStartDate(2007, 1, 1)
        self.LastRotationTime = datetime.min
        self.RotationInterval = timedelta(days=30)
        self.first = True
        GrowthSymbols = ['MDY', 'IEV', 'EEM', 'ILF', 'EPP']
        SafetySymbols = ['EDV', 'SHY']
        self.SymbolData = []
        for symbol in list(set(GrowthSymbols) | set(SafetySymbols)):
            self.AddSecurity(SecurityType.Equity, symbol, Resolution.Minute)
            self.oneMonthPerformance = self.MOM(symbol, 30, Resolution.Daily)
            self.threeMonthPerformance = self.MOM(symbol, 90, Resolution.Daily)
            self.SymbolData.append([symbol, self.oneMonthPerformance, self.threeMonthPerformance])

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if self.first:
            self.first = False
            self.LastRotationTime = self.Time
            return
        delta = self.Time - self.LastRotationTime
        if delta > self.RotationInterval:
            self.LastRotationTime = self.Time
            orderedObjScores = sorted(self.SymbolData, key=lambda x: Score(x[1].Current.Value, x[2].Current.Value).ObjectiveScore(), reverse=True)
            for x in orderedObjScores:
                self.Log('>>SCORE>>' + x[0] + '>>' + str(Score(x[1].Current.Value, x[2].Current.Value).ObjectiveScore()))
            bestGrowth = orderedObjScores[0]
            if Score(bestGrowth[1].Current.Value, bestGrowth[2].Current.Value).ObjectiveScore() > 0:
                if self.Portfolio[bestGrowth[0]].Quantity == 0:
                    self.Log('PREBUY>>LIQUIDATE>>')
                    self.Liquidate()
                self.Log('>>BUY>>' + str(bestGrowth[0]) + '@' + str(100 * bestGrowth[1].Current.Value))
                qty = self.Portfolio.MarginRemaining / self.Securities[bestGrowth[0]].Close
                self.MarketOrder(bestGrowth[0], int(qty))
            else:
                self.Log('>>LIQUIDATE>>CASH')
                self.Liquidate()

class Score(object):

    def __init__(self, oneMonthPerformanceValue, threeMonthPerformanceValue):
        if False:
            print('Hello World!')
        self.oneMonthPerformance = oneMonthPerformanceValue
        self.threeMonthPerformance = threeMonthPerformanceValue

    def ObjectiveScore(self):
        if False:
            while True:
                i = 10
        weight1 = 100
        weight2 = 75
        return (weight1 * self.oneMonthPerformance + weight2 * self.threeMonthPerformance) / (weight1 + weight2)