from AlgorithmImports import *

class CustomDataUsingMapFileRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 6, 27)
        self.SetEndDate(2013, 7, 2)
        self.initialMapping = False
        self.executionMapping = False
        self.foxa = Symbol.Create('FOXA', SecurityType.Equity, Market.USA)
        self.symbol = self.AddData(CustomDataUsingMapping, self.foxa).Symbol
        for config in self.SubscriptionManager.SubscriptionDataConfigService.GetSubscriptionDataConfigs(self.symbol):
            if config.Resolution != Resolution.Minute:
                raise ValueError('Expected resolution to be set to Minute')

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        date = self.Time.date()
        if slice.SymbolChangedEvents.ContainsKey(self.symbol):
            mappingEvent = slice.SymbolChangedEvents[self.symbol]
            self.Log('{0} - Ticker changed from: {1} to {2}'.format(str(self.Time), mappingEvent.OldSymbol, mappingEvent.NewSymbol))
            if date == datetime(2013, 6, 27).date():
                if mappingEvent.NewSymbol != 'NWSA' or mappingEvent.OldSymbol != 'FOXA':
                    raise Exception('Unexpected mapping event mappingEvent')
                self.initialMapping = True
            if date == datetime(2013, 6, 29).date():
                if mappingEvent.NewSymbol != 'FOXA' or mappingEvent.OldSymbol != 'NWSA':
                    raise Exception('Unexpected mapping event mappingEvent')
                self.SetHoldings(self.symbol, 1)
                self.executionMapping = True

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.initialMapping:
            raise Exception('The ticker generated the initial rename event')
        if not self.executionMapping:
            raise Exception('The ticker did not rename throughout the course of its life even though it should have')

class CustomDataUsingMapping(PythonData):
    """Test example custom data showing how to enable the use of mapping.
    Implemented as a wrapper of existing NWSA->FOXA equity"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            print('Hello World!')
        return TradeBar().GetSource(SubscriptionDataConfig(config, CustomDataUsingMapping, Symbol.Create(config.MappedSymbol, SecurityType.Equity, config.Market)), date, isLiveMode)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            while True:
                i = 10
        return TradeBar.ParseEquity(config, line, date)

    def RequiresMapping(self):
        if False:
            i = 10
            return i + 15
        'True indicates mapping should be done'
        return True

    def IsSparseData(self):
        if False:
            i = 10
            return i + 15
        'Indicates that the data set is expected to be sparse'
        return True

    def DefaultResolution(self):
        if False:
            while True:
                i = 10
        'Gets the default resolution for this data and security type'
        return Resolution.Minute

    def SupportedResolutions(self):
        if False:
            while True:
                i = 10
        'Gets the supported resolution for this data and security type'
        return [Resolution.Minute]