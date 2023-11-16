from AlgorithmImports import *

class OpenInterestFuturesRegressionAlgorithm(QCAlgorithm):
    expected_expiry_dates = {datetime(2013, 12, 27), datetime(2014, 2, 26)}

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Tick
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(10000000)
        universe = OpenInterestFutureUniverseSelectionModel(self, lambda date_time: [Symbol.Create(Futures.Metals.Gold, SecurityType.Future, Market.COMEX)], None, len(self.expected_expiry_dates))
        self.SetUniverseSelection(universe)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.Transactions.OrdersCount == 0 and data.HasData:
            matched = list(filter(lambda s: not s.ID.Date in self.expected_expiry_dates and (not s.IsCanonical()), data.Keys))
            if len(matched) != 0:
                raise Exception(f'{len(matched)}/{len(slice.Keys)} were unexpected expiry date(s): ' + ', '.join(list(map(lambda x: x.ID.Date, matched))))
            for symbol in data.Keys:
                self.MarketOrder(symbol, 1)
        elif any((p.Value.Invested for p in self.Portfolio)):
            self.Liquidate()