from AlgorithmImports import *
from scipy.optimize import minimize

class MinimumVariancePortfolioOptimizer:
    """Provides an implementation of a portfolio optimizer that calculate the optimal weights 
    with the weight range from -1 to 1 and minimize the portfolio variance with a target return of 2%"""

    def __init__(self, minimum_weight=-1, maximum_weight=1, target_return=0.02):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the MinimumVariancePortfolioOptimizer\n        Args:\n            minimum_weight(float): The lower bounds on portfolio weights\n            maximum_weight(float): The upper bounds on portfolio weights\n            target_return(float): The target portfolio return'
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.target_return = target_return

    def Optimize(self, historicalReturns, expectedReturns=None, covariance=None):
        if False:
            i = 10
            return i + 15
        '\n        Perform portfolio optimization for a provided matrix of historical returns and an array of expected returns\n        args:\n            historicalReturns: Matrix of annualized historical returns where each column represents a security and each row returns for the given date/time (size: K x N).\n            expectedReturns: Array of double with the portfolio annualized expected returns (size: K x 1).\n            covariance: Multi-dimensional array of double with the portfolio covariance of annualized returns (size: K x K).\n        Returns:\n            Array of double with the portfolio weights (size: K x 1)\n        '
        if covariance is None:
            covariance = historicalReturns.cov()
        if expectedReturns is None:
            expectedReturns = historicalReturns.mean()
        size = historicalReturns.columns.size
        x0 = np.array(size * [1.0 / size])
        constraints = [{'type': 'eq', 'fun': lambda weights: self.get_budget_constraint(weights)}, {'type': 'eq', 'fun': lambda weights: self.get_target_constraint(weights, expectedReturns)}]
        opt = minimize(lambda weights: self.portfolio_variance(weights, covariance), x0, bounds=self.get_boundary_conditions(size), constraints=constraints, method='SLSQP')
        if not opt['success']:
            return x0
        sum_of_absolute_weights = np.sum(np.abs(opt['x']))
        return opt['x'] / sum_of_absolute_weights

    def portfolio_variance(self, weights, covariance):
        if False:
            i = 10
            return i + 15
        'Computes the portfolio variance\n        Args:\n            weighs: Portfolio weights\n            covariance: Covariance matrix of historical returns'
        variance = np.dot(weights.T, np.dot(covariance, weights))
        if variance == 0 and np.any(weights):
            raise ValueError(f'MinimumVariancePortfolioOptimizer.portfolio_variance: Volatility cannot be zero. Weights: {weights}')
        return variance

    def get_boundary_conditions(self, size):
        if False:
            return 10
        'Creates the boundary condition for the portfolio weights'
        return tuple(((self.minimum_weight, self.maximum_weight) for x in range(size)))

    def get_budget_constraint(self, weights):
        if False:
            for i in range(10):
                print('nop')
        'Defines a budget constraint: the sum of the weights equals unity'
        return np.sum(weights) - 1

    def get_target_constraint(self, weights, expectedReturns):
        if False:
            while True:
                i = 10
        'Ensure that the portfolio return target a given return'
        return np.dot(np.matrix(expectedReturns), np.matrix(weights).T).item() - self.target_return