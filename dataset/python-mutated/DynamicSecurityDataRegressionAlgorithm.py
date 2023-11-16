from AlgorithmImports import *
from System.Collections.Generic import List
from QuantConnect.Data.Custom.IconicTypes import *

class DynamicSecurityDataRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2015, 10, 22)
        self.SetEndDate(2015, 10, 30)
        self.Ticker = 'GOOGL'
        self.Equity = self.AddEquity(self.Ticker, Resolution.Daily)
        customLinkedEquity = self.AddData(LinkedData, self.Ticker, Resolution.Daily)
        firstLinkedData = LinkedData()
        firstLinkedData.Count = 100
        firstLinkedData.Symbol = customLinkedEquity.Symbol
        firstLinkedData.EndTime = self.StartDate
        secondLinkedData = LinkedData()
        secondLinkedData.Count = 100
        secondLinkedData.Symbol = customLinkedEquity.Symbol
        secondLinkedData.EndTime = self.StartDate
        customLinkedEquityType = list(customLinkedEquity.Subscriptions)[0].Type
        customLinkedData = List[LinkedData]()
        customLinkedData.Add(firstLinkedData)
        customLinkedData.Add(secondLinkedData)
        self.Equity.Cache.AddDataList(customLinkedData, customLinkedEquityType, False)

    def OnData(self, data):
        if False:
            return 10
        customLinkedData = self.Equity.Data.Get(LinkedData)
        self.Log('{}: LinkedData: {}'.format(self.Time, str(customLinkedData)))
        customLinkedDataList = self.Equity.Data.GetAll(LinkedData)
        self.Log('{}: LinkedData: {}'.format(self.Time, len(customLinkedDataList)))
        if not self.Portfolio.Invested:
            self.Buy(self.Equity.Symbol, 10)