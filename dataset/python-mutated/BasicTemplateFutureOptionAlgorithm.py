from AlgorithmImports import *

class BasicTemplateFutureOptionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 2, 1)
        self.SetCash(100000)
        gold_futures = self.AddFuture(Futures.Metals.Gold, Resolution.Minute)
        gold_futures.SetFilter(0, 180)
        self.symbol = gold_futures.Symbol
        self.AddFutureOption(self.symbol, lambda universe: universe.Strikes(-5, +5).CallsOnly().BackMonth().OnlyApplyFilterAtMarketOpen())
        history = self.History(self.symbol, 60, Resolution.Daily)
        self.Log(f'Received {len(history)} bars from {self.symbol} FutureOption historical data call.')

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n        Arguments:\n            slice: Slice object keyed by symbol containing the stock data\n        '
        for kvp in data.OptionChains:
            underlying_future_contract = kvp.Key.Underlying
            chain = kvp.Value
            if not chain:
                continue
            for contract in chain:
                self.Log(f'Canonical Symbol: {kvp.Key}; \n                    Contract: {contract}; \n                    Right: {contract.Right}; \n                    Expiry: {contract.Expiry}; \n                    Bid price: {contract.BidPrice}; \n                    Ask price: {contract.AskPrice}; \n                    Implied Volatility: {contract.ImpliedVolatility}')
            if not self.Portfolio.Invested:
                atm_strike = sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike))[0].Strike
                selected_contract = sorted([contract for contract in chain if contract.Strike == atm_strike], key=lambda x: x.Expiry, reverse=True)[0]
                self.MarketOrder(selected_contract.Symbol, 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Debug('{} {}'.format(self.Time, orderEvent.ToString()))