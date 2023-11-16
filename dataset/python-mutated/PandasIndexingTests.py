from AlgorithmImports import *
from QuantConnect.Tests import *
from QuantConnect.Tests.Python import *

class PandasIndexingTests:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.qb = QuantBook()
        self.qb.SetStartDate(2020, 1, 1)
        self.qb.SetEndDate(2020, 1, 4)
        self.symbol = self.qb.AddEquity('SPY', Resolution.Daily).Symbol

    def test_indexing_dataframe_with_list(self):
        if False:
            print('Hello World!')
        symbols = [self.symbol]
        self.history = self.qb.History(symbols, 30)
        self.history = self.history['close'].unstack(level=0).dropna()
        test = self.history[[self.symbol]]
        return True

class PandasDataFrameTests:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.spy = Symbols.SPY
        self.aapl = Symbols.AAPL
        SymbolCache.Set('SPY', self.spy)
        SymbolCache.Set('AAPL', self.aapl)
        pdConverter = PandasConverter()
        self.spydf = pdConverter.GetDataFrame(PythonTestingUtils.GetSlices(self.spy))

    def test_contains_user_mapped_ticker(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'spy': [2, 5, 8, 10]})
        return 'spy' in df

    def test_expected_exception(self):
        if False:
            return 10
        try:
            self.spydf['aapl']
        except KeyError as e:
            return str(e)

    def test_contains_user_defined_columns_with_spaces(self, column_name):
        if False:
            print('Hello World!')
        df = self.spydf.copy()
        df[column_name] = 1
        try:
            x = df[column_name]
            return True
        except:
            return False