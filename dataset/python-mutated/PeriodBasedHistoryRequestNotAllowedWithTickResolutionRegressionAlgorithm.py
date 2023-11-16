from AlgorithmImports import *

class PeriodBasedHistoryRequestNotAllowedWithTickResolutionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 9)
        spy = self.AddEquity('SPY', Resolution.Tick).Symbol
        self.AssertThatHistoryThrowsForTickResolution(lambda : self.History[Tick](spy, 1), 'Tick history call with implicit tick resolution')
        self.AssertThatHistoryThrowsForTickResolution(lambda : self.History[Tick](spy, 1, Resolution.Tick), 'Tick history call with explicit tick resolution')
        self.AssertThatHistoryThrowsForTickResolution(lambda : self.History[Tick]([spy], 1), 'Tick history call with symbol array with implicit tick resolution')
        self.AssertThatHistoryThrowsForTickResolution(lambda : self.History[Tick]([spy], 1, Resolution.Tick), 'Tick history call with symbol array with explicit tick resolution')
        history = self.History[Tick](spy, TimeSpan.FromHours(12))
        if len(list(history)) == 0:
            raise Exception('On history call with implicit tick resolution: history returned no results')
        history = self.History[Tick](spy, TimeSpan.FromHours(12), Resolution.Tick)
        if len(list(history)) == 0:
            raise Exception('On history call with explicit tick resolution: history returned no results')

    def AssertThatHistoryThrowsForTickResolution(self, historyCall, historyCallDescription):
        if False:
            print('Hello World!')
        try:
            historyCall()
            raise Exception(f'{historyCallDescription}: expected an exception to be thrown')
        except InvalidOperationException:
            pass