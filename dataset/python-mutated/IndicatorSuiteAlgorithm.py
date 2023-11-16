from AlgorithmImports import *
from QuantConnect.Algorithm.CSharp import *

class IndicatorSuiteAlgorithm(QCAlgorithm):
    """Demonstration algorithm of popular indicators and plotting them."""

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.symbol = 'SPY'
        self.symbol2 = 'GOOG'
        self.customSymbol = 'IBM'
        self.price = None
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2014, 12, 31)
        self.SetCash(25000)
        self.AddEquity(self.symbol, Resolution.Daily)
        self.AddEquity(self.symbol2, Resolution.Daily)
        self.AddData(CustomData, self.customSymbol, Resolution.Daily)
        self.indicators = {'BB': self.BB(self.symbol, 20, 1, MovingAverageType.Simple, Resolution.Daily), 'RSI': self.RSI(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily), 'EMA': self.EMA(self.symbol, 14, Resolution.Daily), 'SMA': self.SMA(self.symbol, 14, Resolution.Daily), 'MACD': self.MACD(self.symbol, 12, 26, 9, MovingAverageType.Simple, Resolution.Daily), 'MOM': self.MOM(self.symbol, 20, Resolution.Daily), 'MOMP': self.MOMP(self.symbol, 20, Resolution.Daily), 'STD': self.STD(self.symbol, 20, Resolution.Daily), 'MIN': self.MIN(self.symbol, 14, Resolution.Daily), 'MAX': self.MAX(self.symbol, 14, Resolution.Daily), 'ATR': self.ATR(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily), 'AROON': self.AROON(self.symbol, 20, Resolution.Daily), 'B': self.B(self.symbol, self.symbol2, 14)}
        self.selectorIndicators = {'BB': self.BB(self.symbol, 20, 1, MovingAverageType.Simple, Resolution.Daily, Field.Low), 'RSI': self.RSI(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily, Field.Low), 'EMA': self.EMA(self.symbol, 14, Resolution.Daily, Field.Low), 'SMA': self.SMA(self.symbol, 14, Resolution.Daily, Field.Low), 'MACD': self.MACD(self.symbol, 12, 26, 9, MovingAverageType.Simple, Resolution.Daily, Field.Low), 'MOM': self.MOM(self.symbol, 20, Resolution.Daily, Field.Low), 'MOMP': self.MOMP(self.symbol, 20, Resolution.Daily, Field.Low), 'STD': self.STD(self.symbol, 20, Resolution.Daily, Field.Low), 'MIN': self.MIN(self.symbol, 14, Resolution.Daily, Field.High), 'MAX': self.MAX(self.symbol, 14, Resolution.Daily, Field.Low), 'ATR': self.ATR(self.symbol, 14, MovingAverageType.Simple, Resolution.Daily, Func[IBaseData, IBaseDataBar](self.selector_double_TradeBar)), 'AROON': self.AROON(self.symbol, 20, Resolution.Daily, Func[IBaseData, IBaseDataBar](self.selector_double_TradeBar))}
        self.rsiCustom = self.RSI(self.customSymbol, 14, MovingAverageType.Simple, Resolution.Daily)
        self.minCustom = self.MIN(self.customSymbol, 14, Resolution.Daily)
        self.maxCustom = self.MAX(self.customSymbol, 14, Resolution.Daily)
        spyClose = Identity(self.symbol)
        ibmClose = Identity(self.customSymbol)
        self.ratio = IndicatorExtensions.Over(ibmClose, spyClose)
        self.PlotIndicator('Ratio', self.ratio)
        Chart('BB')
        Chart('STD')
        Chart('ATR')
        Chart('AROON')
        Chart('MACD')
        Chart('Averages')
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose(self.symbol), self.update_plots)

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.indicators['BB'].IsReady or not self.indicators['RSI'].IsReady:
            return
        self.price = data[self.symbol].Close
        if not self.Portfolio.HoldStock:
            quantity = int(self.Portfolio.Cash / self.price)
            self.Order(self.symbol, quantity)
            self.Debug('Purchased SPY on ' + self.Time.strftime('%Y-%m-%d'))

    def update_plots(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.indicators['BB'].IsReady or not self.indicators['STD'].IsReady:
            return
        self.Plot('RSI', self.indicators['RSI'])
        self.Plot('RSI-FB', self.rsiCustom)
        self.Plot('STD', 'STD', self.indicators['STD'].Current.Value)
        self.Plot('BB', 'Price', self.price)
        self.Plot('BB', 'BollingerUpperBand', self.indicators['BB'].UpperBand.Current.Value)
        self.Plot('BB', 'BollingerMiddleBand', self.indicators['BB'].MiddleBand.Current.Value)
        self.Plot('BB', 'BollingerLowerBand', self.indicators['BB'].LowerBand.Current.Value)
        self.Plot('AROON', 'Aroon', self.indicators['AROON'].Current.Value)
        self.Plot('AROON', 'AroonUp', self.indicators['AROON'].AroonUp.Current.Value)
        self.Plot('AROON', 'AroonDown', self.indicators['AROON'].AroonDown.Current.Value)

    def selector_double_TradeBar(self, bar):
        if False:
            return 10
        trade_bar = TradeBar()
        trade_bar.Close = 2 * bar.Close
        trade_bar.DataType = bar.DataType
        trade_bar.High = 2 * bar.High
        trade_bar.Low = 2 * bar.Low
        trade_bar.Open = 2 * bar.Open
        trade_bar.Symbol = bar.Symbol
        trade_bar.Time = bar.Time
        trade_bar.Value = 2 * bar.Value
        trade_bar.Period = bar.Period
        return trade_bar