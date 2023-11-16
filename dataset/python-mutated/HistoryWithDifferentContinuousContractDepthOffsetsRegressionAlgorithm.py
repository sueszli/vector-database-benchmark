from AlgorithmImports import *

class HistoryWithDifferentContinuousContractDepthOffsetsRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 6)
        self.SetEndDate(2014, 1, 1)
        self._continuousContractSymbol = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily).Symbol

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        contractDepthOffsets = range(3)
        historyResults = [self.History([self._continuousContractSymbol], self.StartDate, self.EndDate, Resolution.Daily, contractDepthOffset=contractDepthOffset).droplevel(0, axis=0).loc[self._continuousContractSymbol].close for contractDepthOffset in contractDepthOffsets]
        if any((x.size == 0 or x.size != historyResults[0].size for x in historyResults)):
            raise Exception('History results are empty or bar counts did not match')
        for j in range(historyResults[0].size):
            closePrices = set((historyResults[i][j] for i in range(len(historyResults))))
            if len(closePrices) != len(contractDepthOffsets):
                raise Exception('History results close prices should have been different for each data mapping mode at each time')