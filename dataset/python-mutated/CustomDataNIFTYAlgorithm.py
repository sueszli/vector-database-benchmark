from AlgorithmImports import *

class CustomDataNIFTYAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2008, 1, 8)
        self.SetEndDate(2014, 7, 25)
        self.SetCash(100000)
        rupee = self.AddData(DollarRupee, 'USDINR', Resolution.Daily).Symbol
        nifty = self.AddData(Nifty, 'NIFTY', Resolution.Daily).Symbol
        self.EnableAutomaticIndicatorWarmUp = True
        rupeeSma = self.SMA(rupee, 20)
        niftySma = self.SMA(rupee, 20)
        self.Log(f'SMA - Is ready? USDINR: {rupeeSma.IsReady} NIFTY: {niftySma.IsReady}')
        self.minimumCorrelationHistory = 50
        self.today = CorrelationPair()
        self.prices = []

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data.ContainsKey('USDINR'):
            self.today = CorrelationPair(self.Time)
            self.today.CurrencyPrice = data['USDINR'].Close
        if not data.ContainsKey('NIFTY'):
            return
        self.today.NiftyPrice = data['NIFTY'].Close
        if self.today.date() == data['NIFTY'].Time.date():
            self.prices.append(self.today)
            if len(self.prices) > self.minimumCorrelationHistory:
                self.prices.pop(0)
        if self.Time.weekday() != 2:
            return
        cur_qnty = self.Portfolio['NIFTY'].Quantity
        quantity = int(self.Portfolio.MarginRemaining * 0.9 / data['NIFTY'].Close)
        hi_nifty = max((price.NiftyPrice for price in self.prices))
        lo_nifty = min((price.NiftyPrice for price in self.prices))
        if data['NIFTY'].Open >= hi_nifty:
            code = self.Order('NIFTY', quantity - cur_qnty)
            self.Debug('LONG  {0} Time: {1} Quantity: {2} Portfolio: {3} Nifty: {4} Buying Power: {5}'.format(code, self.Time, quantity, self.Portfolio['NIFTY'].Quantity, data['NIFTY'].Close, self.Portfolio.TotalPortfolioValue))
        elif data['NIFTY'].Open <= lo_nifty:
            code = self.Order('NIFTY', -quantity - cur_qnty)
            self.Debug('SHORT {0} Time: {1} Quantity: {2} Portfolio: {3} Nifty: {4} Buying Power: {5}'.format(code, self.Time, quantity, self.Portfolio['NIFTY'].Quantity, data['NIFTY'].Close, self.Portfolio.TotalPortfolioValue))

class Nifty(PythonData):
    """NIFTY Custom Data Class"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            while True:
                i = 10
        return SubscriptionDataSource('https://www.dropbox.com/s/rsmg44jr6wexn2h/CNXNIFTY.csv?dl=1', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            print('Hello World!')
        if not (line.strip() and line[0].isdigit()):
            return None
        index = Nifty()
        index.Symbol = config.Symbol
        try:
            data = line.split(',')
            index.Time = datetime.strptime(data[0], '%Y-%m-%d')
            index.EndTime = index.Time + timedelta(days=1)
            index.Value = data[4]
            index['Open'] = float(data[1])
            index['High'] = float(data[2])
            index['Low'] = float(data[3])
            index['Close'] = float(data[4])
        except ValueError:
            return None
        return index

class DollarRupee(PythonData):
    """Dollar Rupe is a custom data type we create for this algorithm"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            i = 10
            return i + 15
        return SubscriptionDataSource('https://www.dropbox.com/s/m6ecmkg9aijwzy2/USDINR.csv?dl=1', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            return 10
        if not (line.strip() and line[0].isdigit()):
            return None
        currency = DollarRupee()
        currency.Symbol = config.Symbol
        try:
            data = line.split(',')
            currency.Time = datetime.strptime(data[0], '%Y-%m-%d')
            currency.EndTime = currency.Time + timedelta(days=1)
            currency.Value = data[1]
            currency['Close'] = float(data[1])
        except ValueError:
            return None
        return currency

class CorrelationPair:
    """Correlation Pair is a helper class to combine two data points which we'll use to perform the correlation."""

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        self.NiftyPrice = 0
        self.CurrencyPrice = 0
        self._date = datetime.min
        if len(args) > 0:
            self._date = args[0]

    def date(self):
        if False:
            i = 10
            return i + 15
        return self._date.date()