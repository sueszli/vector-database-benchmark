from AlgorithmImports import *
from System import *

class HistoryWithDifferentDataMappingModeRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 6)
        self.SetEndDate(2014, 1, 1)
        self._continuousContractSymbol = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily).Symbol

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        dataMappingModes = [DataMappingMode(x) for x in Enum.GetValues(DataMappingMode)]
        historyResults = [self.History([self._continuousContractSymbol], self.StartDate, self.EndDate, Resolution.Daily, dataMappingMode=dataMappingMode).droplevel(0, axis=0).loc[self._continuousContractSymbol].close for dataMappingMode in dataMappingModes]
        if any((x.size != historyResults[0].size for x in historyResults)):
            raise Exception('History results bar count did not match')
        for j in range(historyResults[0].size):
            closePrices = set((historyResults[i][j] for i in range(len(historyResults))))
            if len(closePrices) != len(dataMappingModes):
                raise Exception('History results close prices should have been different for each data mapping mode at each time')