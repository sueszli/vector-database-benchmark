from AlgorithmImports import *

class CustomConsolidatorRegressionAlgorithm(QCAlgorithm):
    """Custom Consolidator Regression Algorithm shows some examples of how to build custom 
    consolidators in Python."""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 4)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Minute)
        fiveDayConsolidator = QuoteBarConsolidator(timedelta(days=5))
        fiveDayConsolidator.DataConsolidated += self.OnQuoteBarDataConsolidated
        self.SubscriptionManager.AddConsolidator('SPY', fiveDayConsolidator)
        timedConsolidator = DailyTimeQuoteBarConsolidator(time(hour=15, minute=10))
        timedConsolidator.DataConsolidated += self.OnQuoteBarDataConsolidated
        self.SubscriptionManager.AddConsolidator('SPY', timedConsolidator)
        self.customConsolidator = CustomQuoteBarConsolidator(timedelta(days=2))
        self.customConsolidator.DataConsolidated += self.OnQuoteBarDataConsolidated
        self.SubscriptionManager.AddConsolidator('SPY', self.customConsolidator)
        self.movingAverage = SimpleMovingAverage(5)
        self.customConsolidator2 = CustomQuoteBarConsolidator(timedelta(hours=1))
        self.RegisterIndicator('SPY', self.movingAverage, self.customConsolidator2)

    def OnQuoteBarDataConsolidated(self, sender, bar):
        if False:
            for i in range(10):
                print('nop')
        'Function assigned to be triggered by consolidators.\n        Designed to post debug messages to show how the examples work, including\n        which consolidator is posting, as well as its values.\n\n        If using an inherited class and not overwriting OnDataConsolidated\n        we expect to see the super C# class as the sender type.\n\n        Using sender.Period only works when all consolidators have a Period value.\n        '
        consolidatorInfo = str(type(sender)) + str(sender.Period)
        self.Debug('Bar Type: ' + consolidatorInfo)
        self.Debug('Bar Range: ' + bar.Time.ctime() + ' - ' + bar.EndTime.ctime())
        self.Debug('Bar value: ' + str(bar.Close))

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        test = slice.get_Values()
        if self.customConsolidator.Consolidated and slice.ContainsKey('SPY'):
            data = slice['SPY']
            if self.movingAverage.IsReady:
                if data.Value > self.movingAverage.Current.Price:
                    self.SetHoldings('SPY', 0.5)
                else:
                    self.SetHoldings('SPY', 0)

class DailyTimeQuoteBarConsolidator(QuoteBarConsolidator):
    """A custom QuoteBar consolidator that inherits from C# class QuoteBarConsolidator. 

    This class shows an example of building on top of an existing consolidator class, it is important
    to note that this class can leverage the functions of QuoteBarConsolidator but its private fields
    (_period, _workingbar, etc.) are separate from this Python. For that reason if we want Scan() to work
    we must overwrite the function with our desired Scan function and trigger OnDataConsolidated().
    
    For this particular example we implemented the scan method to trigger a consolidated bar
    at closeTime everyday"""

    def __init__(self, closeTime):
        if False:
            return 10
        self.closeTime = closeTime
        self.workingBar = None

    def Update(self, data):
        if False:
            while True:
                i = 10
        'Updates this consolidator with the specified data'
        if self.workingBar is None:
            self.workingBar = QuoteBar(data.Time, data.Symbol, data.Bid, data.LastBidSize, data.Ask, data.LastAskSize)
        self.AggregateBar(self.workingBar, data)

    def Scan(self, time):
        if False:
            i = 10
            return i + 15
        'Scans this consolidator to see if it should emit a bar due yet'
        if time.hour == self.closeTime.hour and time.minute == self.closeTime.minute:
            self.workingBar.EndTime = time
            self.OnDataConsolidated(self.workingBar)
            self.workingBar = None

class CustomQuoteBarConsolidator(PythonConsolidator):
    """A custom quote bar consolidator that inherits from PythonConsolidator and implements 
    the IDataConsolidator interface, it must implement all of IDataConsolidator. Reference 
    PythonConsolidator.cs and DataConsolidatorPythonWrapper.py for more information.

    This class shows how to implement a consolidator from scratch in Python, this gives us more
    freedom to determine the behavior of the consolidator but can't leverage any of the built in
    functions of an inherited class.
    
    For this example we implemented a Quotebar from scratch"""

    def __init__(self, period):
        if False:
            return 10
        self.Consolidated = None
        self.WorkingData = None
        self.InputType = QuoteBar
        self.OutputType = QuoteBar
        self.Period = period

    def Update(self, data):
        if False:
            i = 10
            return i + 15
        'Updates this consolidator with the specified data'
        if self.WorkingData is None:
            self.WorkingData = QuoteBar(data.Time, data.Symbol, data.Bid, data.LastBidSize, data.Ask, data.LastAskSize, self.Period)
        self.WorkingData.Update(data.Value, data.Bid.Close, data.Ask.Close, 0, data.LastBidSize, data.LastAskSize)

    def Scan(self, time):
        if False:
            return 10
        'Scans this consolidator to see if it should emit a bar due to time passing'
        if self.Period is not None and self.WorkingData is not None:
            if time - self.WorkingData.Time >= self.Period:
                self.OnDataConsolidated(self, self.WorkingData)
                self.Consolidated = self.WorkingData
                self.WorkingData = None