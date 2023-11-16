from AlgorithmImports import *

class CoarseFineAsyncUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.UniverseSettings.Asynchronous = True
        threw_exception = False
        try:
            self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        except:
            threw_exception = True
            pass
        if not threw_exception:
            raise ValueError('Expected exception to be thrown for AddUniverse')
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction))

    def CoarseSelectionFunction(self, coarse):
        if False:
            for i in range(10):
                print('nop')
        return []

    def FineSelectionFunction(self, fine):
        if False:
            while True:
                i = 10
        return []