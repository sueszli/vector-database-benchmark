from AlgorithmImports import *
from clr import GetClrType as typeof
from Selection.UniverseSelectionModel import UniverseSelectionModel
from itertools import groupby

class ManualUniverseSelectionModel(UniverseSelectionModel):
    """Provides an implementation of IUniverseSelectionModel that simply subscribes to the specified set of symbols"""

    def __init__(self, symbols=list(), universeSettings=None):
        if False:
            while True:
                i = 10
        self.MarketHours = MarketHoursDatabase.FromDataFolder()
        self.symbols = symbols
        self.universeSettings = universeSettings
        for symbol in symbols:
            SymbolCache.Set(symbol.Value, symbol)

    def CreateUniverses(self, algorithm):
        if False:
            return 10
        'Creates the universes for this algorithm. Called once after IAlgorithm.Initialize\n        Args:\n            algorithm: The algorithm instance to create universes for</param>\n        Returns:\n            The universes to be used by the algorithm'
        universeSettings = self.universeSettings if self.universeSettings is not None else algorithm.UniverseSettings
        resolution = universeSettings.Resolution
        type = typeof(Tick) if resolution == Resolution.Tick else typeof(TradeBar)
        universes = list()
        self.symbols = sorted(self.symbols, key=lambda s: (s.ID.Market, s.SecurityType))
        for (key, grp) in groupby(self.symbols, lambda s: (s.ID.Market, s.SecurityType)):
            market = key[0]
            securityType = key[1]
            securityTypeString = Extensions.GetEnumString(securityType, SecurityType)
            universeSymbol = Symbol.Create(f'manual-universe-selection-model-{securityTypeString}-{market}', securityType, market)
            if securityType == SecurityType.Base:
                symbolString = MarketHoursDatabase.GetDatabaseSymbolKey(universeSymbol)
                alwaysOpen = SecurityExchangeHours.AlwaysOpen(TimeZones.NewYork)
                entry = self.MarketHours.SetEntry(market, symbolString, securityType, alwaysOpen, TimeZones.NewYork)
            else:
                entry = self.MarketHours.GetEntry(market, None, securityType)
            config = SubscriptionDataConfig(type, universeSymbol, resolution, entry.DataTimeZone, entry.ExchangeHours.TimeZone, False, False, True)
            universes.append(ManualUniverse(config, universeSettings, list(grp)))
        return universes