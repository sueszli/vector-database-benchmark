from AlgorithmImports import *

class ConstituentsUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self._appl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        self._spy = Symbol.Create('SPY', SecurityType.Equity, Market.USA)
        self._qqq = Symbol.Create('QQQ', SecurityType.Equity, Market.USA)
        self._fb = Symbol.Create('FB', SecurityType.Equity, Market.USA)
        self._step = 0
        self.UniverseSettings.Resolution = Resolution.Daily
        customUniverseSymbol = Symbol(SecurityIdentifier.GenerateConstituentIdentifier('constituents-universe-qctest', SecurityType.Equity, Market.USA), 'constituents-universe-qctest')
        self.AddUniverse(ConstituentsUniverse(customUniverseSymbol, self.UniverseSettings))

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        self._step = self._step + 1
        if self._step == 1:
            if not data.ContainsKey(self._qqq) or not data.ContainsKey(self._appl):
                raise ValueError('Unexpected symbols found, step: ' + str(self._step))
            if data.Count != 2:
                raise ValueError('Unexpected data count, step: ' + str(self._step))
            self.SetHoldings(self._appl, 0.5)
        elif self._step == 2:
            if not data.ContainsKey(self._appl):
                raise ValueError('Unexpected symbols found, step: ' + str(self._step))
            if data.Count != 1:
                raise ValueError('Unexpected data count, step: ' + str(self._step))
            self.Liquidate()
        elif self._step == 3:
            if not data.ContainsKey(self._fb) or not data.ContainsKey(self._spy) or (not data.ContainsKey(self._appl)):
                raise ValueError('Unexpected symbols found, step: ' + str(self._step))
            if data.Count != 3:
                raise ValueError('Unexpected data count, step: ' + str(self._step))
        elif self._step == 4:
            if not data.ContainsKey(self._fb) or not data.ContainsKey(self._spy):
                raise ValueError('Unexpected symbols found, step: ' + str(self._step))
            if data.Count != 2:
                raise ValueError('Unexpected data count, step: ' + str(self._step))
        elif self._step == 5:
            if not data.ContainsKey(self._fb) or not data.ContainsKey(self._spy):
                raise ValueError('Unexpected symbols found, step: ' + str(self._step))
            if data.Count != 2:
                raise ValueError('Unexpected data count, step: ' + str(self._step))

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if self._step != 5:
            raise ValueError('Unexpected step count: ' + str(self._step))

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        for added in changes.AddedSecurities:
            self.Log('AddedSecurities ' + str(added))
        for removed in changes.RemovedSecurities:
            self.Log('RemovedSecurities ' + str(removed) + str(self._step))
            if removed.Symbol == self._appl and self._step != 1 and (self._step != 2) or (removed.Symbol == self._qqq and self._step != 1):
                raise ValueError('Unexpected removal step count: ' + str(self._step))