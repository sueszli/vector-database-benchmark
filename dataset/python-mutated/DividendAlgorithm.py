from AlgorithmImports import *

class DividendAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(1998, 1, 1)
        self.SetEndDate(2006, 1, 21)
        self.SetCash(100000)
        equity = self.AddEquity('MSFT', Resolution.Daily)
        equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
        self.SetBrokerageModel(BrokerageName.TradierBrokerage)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'
        bar = data['MSFT']
        if self.Transactions.OrdersCount == 0:
            self.SetHoldings('MSFT', 0.5)
            quantity = self.CalculateOrderQuantity('MSFT', 0.25)
            self.Debug(f'Purchased Stock: {bar.Price}')
            self.StopMarketOrder('MSFT', -quantity, bar.Low / 2)
            self.LimitOrder('MSFT', -quantity, bar.High * 2)
        if data.Dividends.ContainsKey('MSFT'):
            dividend = data.Dividends['MSFT']
            self.Log(f"{self.Time} >> DIVIDEND >> {dividend.Symbol} - {dividend.Distribution} - {self.Portfolio.Cash} - {self.Portfolio['MSFT'].Price}")
        if data.Splits.ContainsKey('MSFT'):
            split = data.Splits['MSFT']
            self.Log(f"{self.Time} >> SPLIT >> {split.Symbol} - {split.SplitFactor} - {self.Portfolio.Cash} - {self.Portfolio['MSFT'].Price}")

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        self.Log(f'{self.Time} >> ORDER >> {order}')