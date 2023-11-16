from AlgorithmImports import *

class WarmupHistoryAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 5, 2)
        self.SetEndDate(2014, 5, 2)
        self.SetCash(100000)
        forex = self.AddForex('EURUSD', Resolution.Second)
        forex = self.AddForex('NZDUSD', Resolution.Second)
        fast_period = 60
        slow_period = 3600
        self.fast = self.EMA('EURUSD', fast_period)
        self.slow = self.EMA('EURUSD', slow_period)
        history = self.History(['EURUSD', 'NZDUSD'], slow_period + 1)
        self.Log(str(history.loc['EURUSD'].tail()))
        self.Log(str(history.loc['NZDUSD'].tail()))
        for (index, row) in history.loc['EURUSD'].iterrows():
            self.fast.Update(index, row['close'])
            self.slow.Update(index, row['close'])
        self.Log('FAST {0} READY. Samples: {1}'.format('IS' if self.fast.IsReady else 'IS NOT', self.fast.Samples))
        self.Log('SLOW {0} READY. Samples: {1}'.format('IS' if self.slow.IsReady else 'IS NOT', self.slow.Samples))

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        if self.fast.Current.Value > self.slow.Current.Value:
            self.SetHoldings('EURUSD', 1)
        else:
            self.SetHoldings('EURUSD', -1)