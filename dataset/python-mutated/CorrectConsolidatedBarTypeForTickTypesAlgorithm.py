from AlgorithmImports import *

class CorrectConsolidatedBarTypeForTickTypesAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 7)
        symbol = self.AddEquity('SPY', Resolution.Tick).Symbol
        self.Consolidate(symbol, timedelta(minutes=1), TickType.Quote, self.quote_tick_consolidation_handler)
        self.Consolidate(symbol, timedelta(minutes=1), TickType.Trade, self.trade_tick_consolidation_handler)
        self.quote_tick_consolidation_handler_called = False
        self.trade_tick_consolidation_handler_called = False

    def OnData(self, slice: Slice) -> None:
        if False:
            return 10
        if self.Time.hour > 9:
            self.Quit('Early quit to save time')

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not self.quote_tick_consolidation_handler_called:
            raise Exception('quote_tick_consolidation_handler was not called')
        if not self.trade_tick_consolidation_handler_called:
            raise Exception('trade_tick_consolidation_handler was not called')

    def quote_tick_consolidation_handler(self, consolidated_bar: QuoteBar) -> None:
        if False:
            print('Hello World!')
        if type(consolidated_bar) != QuoteBar:
            raise Exception(f'Expected the consolidated bar to be of type {QuoteBar} but was {type(consolidated_bar)}')
        self.quote_tick_consolidation_handler_called = True

    def trade_tick_consolidation_handler(self, consolidated_bar: TradeBar) -> None:
        if False:
            for i in range(10):
                print('nop')
        if type(consolidated_bar) != TradeBar:
            raise Exception(f'Expected the consolidated bar to be of type {TradeBar} but was {type(consolidated_bar)}')
        self.trade_tick_consolidation_handler_called = True