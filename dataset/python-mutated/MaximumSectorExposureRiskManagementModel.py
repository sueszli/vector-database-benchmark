from AlgorithmImports import *
from itertools import groupby

class MaximumSectorExposureRiskManagementModel(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that that limits the sector exposure to the specified percentage"""

    def __init__(self, maximumSectorExposure=0.2):
        if False:
            i = 10
            return i + 15
        'Initializes a new instance of the MaximumSectorExposureRiskManagementModel class\n        Args:\n            maximumDrawdownPercent: The maximum exposure for any sector, defaults to 20% sector exposure.'
        if maximumSectorExposure <= 0:
            raise ValueError('MaximumSectorExposureRiskManagementModel: the maximum sector exposure cannot be a non-positive value.')
        self.maximumSectorExposure = maximumSectorExposure
        self.targetsCollection = PortfolioTargetCollection()

    def ManageRisk(self, algorithm, targets):
        if False:
            while True:
                i = 10
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance"
        maximumSectorExposureValue = float(algorithm.Portfolio.TotalPortfolioValue) * self.maximumSectorExposure
        self.targetsCollection.AddRange(targets)
        risk_targets = list()
        filtered = list(filter(lambda x: x.Value.Fundamentals is not None and x.Value.Fundamentals.HasFundamentalData, algorithm.UniverseManager.ActiveSecurities))
        filtered.sort(key=lambda x: x.Value.Fundamentals.CompanyReference.IndustryTemplateCode)
        groupBySector = groupby(filtered, lambda x: x.Value.Fundamentals.CompanyReference.IndustryTemplateCode)
        for (code, securities) in groupBySector:
            quantities = {}
            sectorAbsoluteHoldingsValue = 0
            for security in securities:
                symbol = security.Value.Symbol
                quantities[symbol] = security.Value.Holdings.Quantity
                absoluteHoldingsValue = security.Value.Holdings.AbsoluteHoldingsValue
                if self.targetsCollection.ContainsKey(symbol):
                    quantities[symbol] = self.targetsCollection[symbol].Quantity
                    absoluteHoldingsValue = security.Value.Price * abs(quantities[symbol]) * security.Value.SymbolProperties.ContractMultiplier * security.Value.QuoteCurrency.ConversionRate
                sectorAbsoluteHoldingsValue += absoluteHoldingsValue
            ratio = float(sectorAbsoluteHoldingsValue) / maximumSectorExposureValue
            if ratio > 1:
                for (symbol, quantity) in quantities.items():
                    if quantity != 0:
                        risk_targets.append(PortfolioTarget(symbol, float(quantity) / ratio))
        return risk_targets

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            return 10
        'Event fired each time the we add/remove securities from the data feed\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        anyFundamentalData = any([kvp.Value.Fundamentals is not None and kvp.Value.Fundamentals.HasFundamentalData for kvp in algorithm.ActiveSecurities])
        if not anyFundamentalData:
            raise Exception('MaximumSectorExposureRiskManagementModel.OnSecuritiesChanged: Please select a portfolio selection model that selects securities with fundamental data.')