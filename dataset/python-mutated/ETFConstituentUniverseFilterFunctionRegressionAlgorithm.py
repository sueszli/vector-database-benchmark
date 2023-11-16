from AlgorithmImports import *

class ETFConstituentUniverseFilterFunctionRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2021, 1, 31)
        self.SetCash(100000)
        self.filtered = False
        self.securitiesChanged = False
        self.receivedData = False
        self.etfConstituentData = {}
        self.etfRebalanced = False
        self.rebalanceCount = 0
        self.rebalanceAssetCount = 0
        self.UniverseSettings.Resolution = Resolution.Hour
        self.spy = self.AddEquity('SPY', Resolution.Hour).Symbol
        self.aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        self.AddUniverse(self.Universe.ETF(self.spy, self.UniverseSettings, self.FilterETFs))

    def FilterETFs(self, constituents):
        if False:
            print('Hello World!')
        constituentsData = list(constituents)
        constituentsSymbols = [i.Symbol for i in constituentsData]
        self.etfConstituentData = {i.Symbol: i for i in constituentsData}
        if len(constituentsData) == 0:
            raise Exception(f"Constituents collection is empty on {self.UtcTime.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        if self.aapl not in constituentsSymbols:
            raise Exception('AAPL is not int he constituents data provided to the algorithm')
        aaplData = [i for i in constituentsData if i.Symbol == self.aapl][0]
        if aaplData.Weight == 0.0:
            raise Exception('AAPL weight is expected to be a non-zero value')
        self.filtered = True
        self.etfRebalanced = True
        return constituentsSymbols

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not self.filtered and len(data.Bars) != 0 and (self.aapl in data.Bars):
            raise Exception('AAPL TradeBar data added to algorithm before constituent universe selection took place')
        if len(data.Bars) == 1 and self.spy in data.Bars:
            return
        if len(data.Bars) != 0 and self.aapl not in data.Bars:
            raise Exception(f"Expected AAPL TradeBar data on {self.UtcTime.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        self.receivedData = True
        if not self.etfRebalanced:
            return
        for bar in data.Bars.Values:
            constituentData = self.etfConstituentData.get(bar.Symbol)
            if constituentData is not None and constituentData.Weight is not None and (constituentData.Weight >= 0.0001):
                boundedWeight = max(0.01, min(constituentData.Weight, 0.05))
                self.SetHoldings(bar.Symbol, boundedWeight)
                if self.etfRebalanced:
                    self.rebalanceCount += 1
                self.etfRebalanced = False
                self.rebalanceAssetCount += 1

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        if self.filtered and (not self.securitiesChanged) and (len(changes.AddedSecurities) < 500):
            raise Exception(f'Added SPY S&P 500 ETF to algorithm, but less than 500 equities were loaded (added {len(changes.AddedSecurities)} securities)')
        self.securitiesChanged = True

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.rebalanceCount != 1:
            raise Exception(f'Expected 1 rebalance, instead rebalanced: {self.rebalanceCount}')
        if self.rebalanceAssetCount != 4:
            raise Exception(f'Invested in {self.rebalanceAssetCount} assets (expected 4)')
        if not self.filtered:
            raise Exception('Universe selection was never triggered')
        if not self.securitiesChanged:
            raise Exception('Security changes never propagated to the algorithm')
        if not self.receivedData:
            raise Exception('Data was never loaded for the S&P 500 constituent AAPL')