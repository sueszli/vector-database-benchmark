from AlgorithmImports import *

class ConstituentsQC500GeneratorAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2019, 1, 1)
        self.SetCash(100000)
        self.AddUniverse(self.Universe.QC500)