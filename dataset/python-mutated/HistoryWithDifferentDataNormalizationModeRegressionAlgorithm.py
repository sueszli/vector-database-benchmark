from AlgorithmImports import *

class HistoryWithDifferentDataMappingModeRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2014, 1, 1)
        self.aaplEquitySymbol = self.AddEquity('AAPL', Resolution.Daily).Symbol
        self.esFutureSymbol = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily).Symbol

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        equityDataNormalizationModes = [DataNormalizationMode.Raw, DataNormalizationMode.Adjusted, DataNormalizationMode.SplitAdjusted]
        self.CheckHistoryResultsForDataNormalizationModes(self.aaplEquitySymbol, self.StartDate, self.EndDate, Resolution.Daily, equityDataNormalizationModes)
        futureDataNormalizationModes = [DataNormalizationMode.Raw, DataNormalizationMode.BackwardsRatio, DataNormalizationMode.BackwardsPanamaCanal, DataNormalizationMode.ForwardPanamaCanal]
        self.CheckHistoryResultsForDataNormalizationModes(self.esFutureSymbol, self.StartDate, self.EndDate, Resolution.Daily, futureDataNormalizationModes)

    def CheckHistoryResultsForDataNormalizationModes(self, symbol, start, end, resolution, dataNormalizationModes):
        if False:
            while True:
                i = 10
        historyResults = [self.History([symbol], start, end, resolution, dataNormalizationMode=x) for x in dataNormalizationModes]
        historyResults = [x.droplevel(0, axis=0) for x in historyResults] if len(historyResults[0].index.levels) == 3 else historyResults
        historyResults = [x.loc[symbol].close for x in historyResults]
        if any((x.size == 0 or x.size != historyResults[0].size for x in historyResults)):
            raise Exception(f'History results for {symbol} have different number of bars')
        for j in range(historyResults[0].size):
            closePrices = set((historyResults[i][j] for i in range(len(historyResults))))
            if len(closePrices) != len(dataNormalizationModes):
                raise Exception(f'History results for {symbol} have different close prices at the same time')