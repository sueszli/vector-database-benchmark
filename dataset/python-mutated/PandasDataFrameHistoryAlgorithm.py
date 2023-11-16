from AlgorithmImports import *

class PandasDataFrameHistoryAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2014, 6, 9)
        self.SetEndDate(2014, 6, 9)
        self.spy = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.eur = self.AddForex('EURUSD', Resolution.Daily).Symbol
        aapl = self.AddEquity('AAPL', Resolution.Minute).Symbol
        self.option = Symbol.CreateOption(aapl, Market.USA, OptionStyle.American, OptionRight.Call, 750, datetime(2014, 10, 18))
        self.AddOptionContract(self.option)
        sp1 = self.AddData(QuandlFuture, 'CHRIS/CME_SP1', Resolution.Daily)
        sp1.Exchange = EquityExchange()
        self.sp1 = sp1.Symbol
        self.AddUniverse(self.CoarseSelection)

    def CoarseSelection(self, coarse):
        if False:
            i = 10
            return i + 15
        if self.Portfolio.Invested:
            return Universe.Unchanged
        selected = [x.Symbol for x in coarse if x.Symbol.Value in ['AAA', 'AIG', 'BAC']]
        if len(selected) == 0:
            return Universe.Unchanged
        universeHistory = self.History(selected, 10, Resolution.Daily)
        for symbol in selected:
            self.AssertHistoryIndex(universeHistory, 'close', 10, '', symbol)
        return selected

    def OnData(self, data):
        if False:
            print('Hello World!')
        if self.Portfolio.Invested:
            return
        self.spyDailySma = SimpleMovingAverage(14)
        tradeBarHistory = self.History(['SPY'], timedelta(365))
        self.AssertHistoryIndex(tradeBarHistory, 'close', 251, 'SPY', self.spy)
        quoteBarHistory = self.History(['EURUSD'], timedelta(298))
        self.AssertHistoryIndex(quoteBarHistory, 'bidclose', 251, 'EURUSD', self.eur)
        optionHistory = self.History([self.option], timedelta(3))
        optionHistory.index = optionHistory.index.droplevel(level=[0, 1, 2])
        self.AssertHistoryIndex(optionHistory, 'bidclose', 390, '', self.option)
        quandlHistory = self.History(QuandlFuture, 'CHRIS/CME_SP1', timedelta(365))
        self.AssertHistoryIndex(quandlHistory, 'settle', 251, 'CHRIS/CME_SP1', self.sp1)
        self.spyDailySma.Reset()
        for (index, tradeBar) in tradeBarHistory.loc['SPY'].iterrows():
            self.spyDailySma.Update(index, tradeBar['close'])
        self.spyDailySma.Reset()
        for (index, quandl) in quandlHistory.loc['CHRIS/CME_SP1'].iterrows():
            self.spyDailySma.Update(index, quandl['settle'])
        self.SetHoldings(self.eur, 1)

    def AssertHistoryIndex(self, df, column, expected, ticker, symbol):
        if False:
            while True:
                i = 10
        if df.empty:
            raise Exception(f'Empty history data frame for {symbol}')
        if column not in df:
            raise Exception(f"Could not unstack df. Columns: {', '.join(df.columns)} | {column}")
        value = df.iat[0, 0]
        df2 = df.xs(df.index.get_level_values('time')[0], level='time')
        df3 = df[column].unstack(level=0)
        try:
            self.AssertHistoryCount(f'df.iloc[0]', df.iloc[0], len(df.columns))
            self.AssertHistoryCount(f'df.loc[str({symbol.ID})]', df.loc[str(symbol.ID)], expected)
            self.AssertHistoryCount(f'df.xs(str({symbol.ID}))', df.xs(str(symbol.ID)), expected)
            self.AssertHistoryCount(f"df.at[(str({symbol.ID}),), '{column}']", list(df.at[(str(symbol.ID),), column]), expected)
            self.AssertHistoryCount(f'df2.loc[str({symbol.ID})]', df2.loc[str(symbol.ID)], len(df2.columns))
            self.AssertHistoryCount(f'df3[str({symbol.ID})]', df3[str(symbol.ID)], expected)
            self.AssertHistoryCount(f'df3.get(str({symbol.ID}))', df3.get(str(symbol.ID)), expected)
            self.AssertHistoryCount(f'df.loc[str({symbol})]', df.loc[str(symbol)], expected)
            self.AssertHistoryCount(f'df.xs(str({symbol}))', df.xs(str(symbol)), expected)
            self.AssertHistoryCount(f"df.at[(str({symbol}),), '{column}']", list(df.at[(str(symbol),), column]), expected)
            self.AssertHistoryCount(f'df2.loc[str({symbol})]', df2.loc[str(symbol)], len(df2.columns))
            self.AssertHistoryCount(f'df3[str({symbol})]', df3[str(symbol)], expected)
            self.AssertHistoryCount(f'df3.get(str({symbol}))', df3.get(str(symbol)), expected)
            if len(ticker) == 0:
                return
            self.AssertHistoryCount(f'df.loc[{ticker}]', df.loc[ticker], expected)
            self.AssertHistoryCount(f'df.xs({ticker})', df.xs(ticker), expected)
            self.AssertHistoryCount(f"df.at[(ticker,), '{column}']", list(df.at[(ticker,), column]), expected)
            self.AssertHistoryCount(f'df2.loc[{ticker}]', df2.loc[ticker], len(df2.columns))
            self.AssertHistoryCount(f'df3[{ticker}]', df3[ticker], expected)
            self.AssertHistoryCount(f'df3.get({ticker})', df3.get(ticker), expected)
        except Exception as e:
            symbols = set(df.index.get_level_values(level='symbol'))
            raise Exception(f'{symbols}, {symbol.ID}, {symbol}, {ticker}. {e}')

    def AssertHistoryCount(self, methodCall, tradeBarHistory, expected):
        if False:
            i = 10
            return i + 15
        if isinstance(tradeBarHistory, list):
            count = len(tradeBarHistory)
        else:
            count = len(tradeBarHistory.index)
        if count != expected:
            raise Exception(f'{methodCall} expected {expected}, but received {count}')

class QuandlFuture(PythonQuandl):
    """Custom quandl data type for setting customized value column name. Value column is used for the primary trading calculations and charting."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.ValueColumnName = 'Settle'