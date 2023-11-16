from AlgorithmImports import *

class BasicTemplateContinuousFutureAlgorithm(QCAlgorithm):
    """Basic template algorithm simply initializes the date range and cash"""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2013, 7, 1)
        self.SetEndDate(2014, 1, 1)
        self._continuousContract = self.AddFuture(Futures.Indices.SP500EMini, dataNormalizationMode=DataNormalizationMode.BackwardsRatio, dataMappingMode=DataMappingMode.LastTradingDay, contractDepthOffset=0)
        self._fast = self.SMA(self._continuousContract.Symbol, 4, Resolution.Daily)
        self._slow = self.SMA(self._continuousContract.Symbol, 10, Resolution.Daily)
        self._currentContract = None

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        for changedEvent in data.SymbolChangedEvents.Values:
            if changedEvent.Symbol == self._continuousContract.Symbol:
                self.Log(f'SymbolChanged event: {changedEvent}')
        if not self.Portfolio.Invested:
            if self._fast.Current.Value > self._slow.Current.Value:
                self._currentContract = self.Securities[self._continuousContract.Mapped]
                self.Buy(self._currentContract.Symbol, 1)
        elif self._fast.Current.Value < self._slow.Current.Value:
            self.Liquidate()
        if self._currentContract is not None and self._currentContract.Symbol != self._continuousContract.Mapped and self._continuousContract.Exchange.ExchangeOpen:
            self.Log(f'{self.Time} - rolling position from {self._currentContract.Symbol} to {self._continuousContract.Mapped}')
            currentPositionSize = self._currentContract.Holdings.Quantity
            self.Liquidate(self._currentContract.Symbol)
            self.Buy(self._continuousContract.Mapped, currentPositionSize)
            self._currentContract = self.Securities[self._continuousContract.Mapped]

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Debug('Purchased Stock: {0}'.format(orderEvent.Symbol))

    def OnSecuritiesChanged(self, changes):
        if False:
            return 10
        self.Debug(f'{self.Time}-{changes}')