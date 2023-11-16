from AlgorithmImports import *
import scipy.stats as sp
from Risk.NullRiskManagementModel import NullRiskManagementModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
from Execution.ImmediateExecutionModel import ImmediateExecutionModel

class ContingentClaimsAnalysisDefaultPredictionAlpha(QCAlgorithm):
    """ Contingent Claim Analysis is put forth by Robert Merton, recepient of the Noble Prize in Economics in 1997 for his work in contributing to
    Black-Scholes option pricing theory, which says that the equity market value of stockholders’ equity is given by the Black-Scholes solution
    for a European call option. This equation takes into account Debt, which in CCA is the equivalent to a strike price in the BS solution. The probability
    of default on corporate debt can be calculated as the N(-d2) term, where d2 is a function of the interest rate on debt(µ), face value of the debt (B), value of the firm's assets (V),
    standard deviation of the change in a firm's asset value (σ), the dividend and interest payouts due (D), and the time to maturity of the firm's debt(τ). N(*) is the cumulative
    distribution function of a standard normal distribution, and calculating N(-d2) gives us the probability of the firm's assets being worth less
    than the debt of the company at the time that the debt reaches maturity -- that is, the firm doesn't have enough in assets to pay off its debt and defaults.

    We use a Fine/Coarse Universe Selection model to select small cap stocks, who we postulate are more likely to default
    on debt in general than blue-chip companies, and extract Fundamental data to plug into the CCA formula.
    This Alpha emits insights based on whether or not a company is likely to default given its probability of default vs a default probability threshold that we set arbitrarily.

    Prob. default (on principal B at maturity T) = Prob(VT < B) = 1 - N(d2) = N(-d2) where -d2(µ) = -{ln(V/B) + [(µ - D) - ½σ2]τ}/ σ √τ.
        N(d) = (univariate) cumulative standard normal distribution function (from -inf to d)
        B = face value (principal) of the debt
        D = dividend + interest payout
        V = value of firm’s assets
        σ (sigma) = standard deviation of firm value changes (returns in V)
        τ (tau)  = time to debt’s maturity
        µ (mu) = interest rate

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
    sourced so the community and client funds can see an example of an alpha."""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.UniverseSettings.Resolution = Resolution.Daily
        self.month = -1
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetBenchmark('IJR')
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction))
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetAlpha(ContingentClaimsAnalysisAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        if self.Time.month == self.month:
            return Universe.Unchanged
        self.month = self.Time.month
        sortedByDollarVolume = sorted([x for x in coarse if x.HasFundamentalData], key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:750]]

    def FineSelectionFunction(self, fine):
        if False:
            i = 10
            return i + 15

        def IsValid(x):
            if False:
                return 10
            statement = x.FinancialStatements
            sheet = statement.BalanceSheet
            total_assets = sheet.TotalAssets
            ratios = x.OperationRatios
            return total_assets.OneMonth > 0 and total_assets.ThreeMonths > 0 and (total_assets.SixMonths > 0) and (total_assets.TwelveMonths > 0) and (sheet.CurrentLiabilities.TwelveMonths > 0) and (sheet.InterestPayable.TwelveMonths > 0) and (ratios.TotalAssetsGrowth.OneYear > 0) and (statement.IncomeStatement.GrossDividendPayment.TwelveMonths > 0) and (ratios.ROA.OneYear > 0)
        return [x.Symbol for x in sorted(fine, key=lambda x: IsValid(x))]

class ContingentClaimsAnalysisAlphaModel:

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.ProbabilityOfDefaultBySymbol = {}
        self.default_threshold = kwargs['default_threshold'] if 'default_threshold' in kwargs else 0.25

    def Update(self, algorithm, data):
        if False:
            return 10
        'Updates this alpha model with the latest data from the algorithm.\n        This is called each time the algorithm receives data for subscribed securities\n        Args:\n            algorithm: The algorithm instance\n            data: The new data available\n        Returns:\n            The new insights generated'
        insights = []
        for (symbol, pod) in self.ProbabilityOfDefaultBySymbol.items():
            if pod >= self.default_threshold and pod != 1.0:
                insights.append(Insight.Price(symbol, timedelta(30), InsightDirection.Down, pod, None))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        for removed in changes.RemovedSecurities:
            self.ProbabilityOfDefaultBySymbol.pop(removed.Symbol, None)
        symbols = [x.Symbol for x in changes.AddedSecurities]
        for symbol in symbols:
            if symbol not in self.ProbabilityOfDefaultBySymbol:
                pod = self.GetProbabilityOfDefault(algorithm, symbol)
                if pod is not None:
                    self.ProbabilityOfDefaultBySymbol[symbol] = pod

    def GetProbabilityOfDefault(self, algorithm, symbol):
        if False:
            return 10
        'This model applies options pricing theory, Black-Scholes specifically,\n        to fundamental data to give the probability of a default'
        security = algorithm.Securities[symbol]
        if security.Fundamentals is None or security.Fundamentals.FinancialStatements is None or security.Fundamentals.OperationRatios is None:
            return None
        statement = security.Fundamentals.FinancialStatements
        sheet = statement.BalanceSheet
        total_assets = sheet.TotalAssets
        tau = 360
        mu = security.Fundamentals.OperationRatios.ROA.OneYear
        V = total_assets.TwelveMonths
        B = sheet.CurrentLiabilities.TwelveMonths
        D = statement.IncomeStatement.GrossDividendPayment.TwelveMonths + sheet.InterestPayable.TwelveMonths
        series = pd.Series([total_assets.OneMonth, total_assets.ThreeMonths, total_assets.SixMonths, V])
        sigma = series.iloc[series.nonzero()[0]]
        sigma = np.std(sigma.pct_change()[1:len(sigma)])
        d2 = (np.log(V) - np.log(B) + (mu - D - 0.5 * sigma ** 2.0) * tau) / (sigma * np.sqrt(tau))
        return sp.norm.cdf(-d2)