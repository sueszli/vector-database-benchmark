from AlgorithmImports import *
from Selection.ManualUniverseSelectionModel import ManualUniverseSelectionModel

class G10CurrencySelectionModelFrameworkAlgorithm(QCAlgorithm):
    """Framework algorithm that uses the G10CurrencySelectionModel,
    a Universe Selection Model that inherits from ManualUniverseSelectionMode"""

    def Initialize(self):
        if False:
            print('Hello World!')
        ' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.SetUniverseSelection(self.G10CurrencySelectionModel())
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.01))

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug('Purchased Stock: {0}'.format(orderEvent.Symbol))

    class G10CurrencySelectionModel(ManualUniverseSelectionModel):
        """Provides an implementation of IUniverseSelectionModel that simply subscribes to G10 currencies"""

        def __init__(self):
            if False:
                i = 10
                return i + 15
            "Initializes a new instance of the G10CurrencySelectionModel class\n            using the algorithm's security initializer and universe settings"
            super().__init__([Symbol.Create(x, SecurityType.Forex, Market.Oanda) for x in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK']])