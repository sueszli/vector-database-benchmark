from cmath import isclose
from AlgorithmImports import *

class BybitCryptoFuturesRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2022, 12, 13)
        self.SetEndDate(2022, 12, 13)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.Bybit, AccountType.Margin)
        self.AddCrypto('BTCUSDT', Resolution.Minute)
        self.btcUsdt = self.AddCryptoFuture('BTCUSDT', Resolution.Minute)
        self.btcUsd = self.AddCryptoFuture('BTCUSD', Resolution.Minute)
        self.fast = self.EMA(self.btcUsdt.Symbol, 30, Resolution.Minute)
        self.slow = self.EMA(self.btcUsdt.Symbol, 60, Resolution.Minute)
        self.interestPerSymbol = {}
        self.interestPerSymbol[self.btcUsd.Symbol] = 0
        self.interestPerSymbol[self.btcUsdt.Symbol] = 0
        self.btcUsdt.QuoteCurrency.SetAmount(200)
        self.btcUsd.BaseCurrency.SetAmount(0.005)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        interestRates = data.Get[MarginInterestRate]()
        for interestRate in interestRates:
            self.interestPerSymbol[interestRate.Key] += 1
            cachedInterestRate = self.Securities[interestRate.Key].Cache.GetData[MarginInterestRate]()
            if cachedInterestRate != interestRate.Value:
                raise Exception(f'Unexpected cached margin interest rate for {interestRate.Key}!')
        if not self.slow.IsReady:
            return
        if self.fast > self.slow:
            if not self.Portfolio.Invested and self.Transactions.OrdersCount == 0:
                ticket = self.Buy(self.btcUsd.Symbol, 1000)
                if ticket.Status != OrderStatus.Invalid:
                    raise Exception(f'Unexpected valid order {ticket}, should fail due to margin not sufficient')
                self.Buy(self.btcUsd.Symbol, 100)
                marginUsed = self.Portfolio.TotalMarginUsed
                btcUsdHoldings = self.btcUsd.Holdings
                holdingsValueBtcUsd = 100
                if abs(btcUsdHoldings.TotalSaleVolume - holdingsValueBtcUsd) > 1:
                    raise Exception(f'Unexpected TotalSaleVolume {btcUsdHoldings.TotalSaleVolume}')
                if abs(btcUsdHoldings.AbsoluteHoldingsCost - holdingsValueBtcUsd) > 1:
                    raise Exception(f'Unexpected holdings cost {btcUsdHoldings.HoldingsCost}')
                if abs(btcUsdHoldings.AbsoluteHoldingsCost * 0.05 - marginUsed) > 1 or not isclose(self.btcUsd.BuyingPowerModel.GetMaintenanceMargin(MaintenanceMarginParameters.ForCurrentHoldings(self.btcUsd)).Value, marginUsed):
                    raise Exception(f'Unexpected margin used {marginUsed}')
                self.Buy(self.btcUsdt.Symbol, 0.01)
                marginUsed = self.Portfolio.TotalMarginUsed - marginUsed
                btcUsdtHoldings = self.btcUsdt.Holdings
                holdingsValueUsdt = self.btcUsdt.Price * self.btcUsdt.SymbolProperties.ContractMultiplier * 0.01
                if abs(btcUsdtHoldings.TotalSaleVolume - holdingsValueUsdt) > 1:
                    raise Exception(f'Unexpected TotalSaleVolume {btcUsdtHoldings.TotalSaleVolume}')
                if abs(btcUsdtHoldings.AbsoluteHoldingsCost - holdingsValueUsdt) > 1:
                    raise Exception(f'Unexpected holdings cost {btcUsdtHoldings.HoldingsCost}')
                if abs(btcUsdtHoldings.AbsoluteHoldingsCost * 0.05 - marginUsed) > 1 or not isclose(self.btcUsdt.BuyingPowerModel.GetMaintenanceMargin(MaintenanceMarginParameters.ForCurrentHoldings(self.btcUsdt)).Value, marginUsed):
                    raise Exception(f'Unexpected margin used {marginUsed}')
                unrealizedProfit = self.Portfolio.TotalUnrealizedProfit
                if 5 - abs(unrealizedProfit) < 0:
                    raise Exception(f'Unexpected TotalUnrealizedProfit {self.Portfolio.TotalUnrealizedProfit}')
                if self.Portfolio.TotalProfit != 0:
                    raise Exception(f'Unexpected TotalProfit {self.Portfolio.TotalProfit}')
        elif self.Transactions.OrdersCount == 3:
            self.Sell(self.btcUsd.Symbol, 300)
            btcUsdHoldings = self.btcUsd.Holdings
            if abs(btcUsdHoldings.AbsoluteHoldingsCost - 100 * 2) > 1:
                raise Exception(f'Unexpected holdings cost {btcUsdHoldings.HoldingsCost}')
            self.Sell(self.btcUsdt.Symbol, 0.03)
            holdingsValueUsdt = self.btcUsdt.Price * self.btcUsdt.SymbolProperties.ContractMultiplier * 0.02
            if abs(self.btcUsdt.Holdings.AbsoluteHoldingsCost - holdingsValueUsdt) > 1:
                raise Exception(f'Unexpected holdings cost {self.btcUsdt.Holdings.HoldingsCost}')
            profit = self.Portfolio.TotalUnrealizedProfit
            if 5 - abs(profit) < 0:
                raise Exception(f'Unexpected TotalUnrealizedProfit {self.Portfolio.TotalUnrealizedProfit}')
            if 5 - abs(self.Portfolio.TotalProfit) < 0:
                raise Exception(f'Unexpected TotalProfit {self.Portfolio.TotalProfit}')

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Debug('{} {}'.format(self.Time, orderEvent.ToString()))

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        self.Log(f'{self.Time} - TotalPortfolioValue: {self.Portfolio.TotalPortfolioValue}')
        self.Log(f'{self.Time} - CashBook: {self.Portfolio.CashBook}')
        if any((x == 0 for x in self.interestPerSymbol.values())):
            raise Exception('Expected interest rate data for all symbols')