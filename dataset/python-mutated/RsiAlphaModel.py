from AlgorithmImports import *
from QuantConnect.Logging import *
from enum import Enum

class RsiAlphaModel(AlphaModel):
    """Uses Wilder's RSI to create insights.
    Using default settings, a cross over below 30 or above 70 will trigger a new insight."""

    def __init__(self, period=14, resolution=Resolution.Daily):
        if False:
            print('Hello World!')
        'Initializes a new instance of the RsiAlphaModel class\n        Args:\n            period: The RSI indicator period'
        self.period = period
        self.resolution = resolution
        self.insightPeriod = Time.Multiply(Extensions.ToTimeSpan(resolution), period)
        self.symbolDataBySymbol = {}
        resolutionString = Extensions.GetEnumString(resolution, Resolution)
        self.Name = '{}({},{})'.format(self.__class__.__name__, period, resolutionString)

    def Update(self, algorithm, data):
        if False:
            while True:
                i = 10
        'Updates this alpha model with the latest data from the algorithm.\n        This is called each time the algorithm receives data for subscribed securities\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            rsi = symbolData.RSI
            previous_state = symbolData.State
            state = self.GetState(rsi, previous_state)
            if state != previous_state and rsi.IsReady:
                if state == State.TrippedLow:
                    insights.append(Insight.Price(symbol, self.insightPeriod, InsightDirection.Up))
                if state == State.TrippedHigh:
                    insights.append(Insight.Price(symbol, self.insightPeriod, InsightDirection.Down))
            symbolData.State = state
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            i = 10
            return i + 15
        'Cleans out old security data and initializes the RSI for any newly added securities.\n        Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for security in changes.RemovedSecurities:
            symbol_data = self.symbolDataBySymbol.pop(security.Symbol, None)
            if symbol_data:
                symbol_data.dispose()
        added_symbols = []
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbolDataBySymbol:
                symbol_data = SymbolData(algorithm, symbol, self.period, self.resolution)
                self.symbolDataBySymbol[symbol] = symbol_data
                added_symbols.append(symbol)
        if added_symbols:
            history = algorithm.History[TradeBar](added_symbols, self.period, self.resolution)
            for trade_bars in history:
                for bar in trade_bars.Values:
                    self.symbolDataBySymbol[bar.Symbol].update(bar)

    def GetState(self, rsi, previous):
        if False:
            while True:
                i = 10
        ' Determines the new state. This is basically cross-over detection logic that\n        includes considerations for bouncing using the configured bounce tolerance.'
        if rsi.Current.Value > 70:
            return State.TrippedHigh
        if rsi.Current.Value < 30:
            return State.TrippedLow
        if previous == State.TrippedLow:
            if rsi.Current.Value > 35:
                return State.Middle
        if previous == State.TrippedHigh:
            if rsi.Current.Value < 65:
                return State.Middle
        return previous

class SymbolData:
    """Contains data specific to a symbol required by this model"""

    def __init__(self, algorithm, symbol, period, resolution):
        if False:
            i = 10
            return i + 15
        self.algorithm = algorithm
        self.Symbol = symbol
        self.State = State.Middle
        self.RSI = RelativeStrengthIndex(period, MovingAverageType.Wilders)
        self.consolidator = algorithm.ResolveConsolidator(symbol, resolution)
        algorithm.RegisterIndicator(symbol, self.RSI, self.consolidator)

    def update(self, bar):
        if False:
            for i in range(10):
                print('nop')
        self.consolidator.Update(bar)

    def dispose(self):
        if False:
            i = 10
            return i + 15
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.Symbol, self.consolidator)

class State(Enum):
    """Defines the state. This is used to prevent signal spamming and aid in bounce detection."""
    TrippedLow = 0
    Middle = 1
    TrippedHigh = 2