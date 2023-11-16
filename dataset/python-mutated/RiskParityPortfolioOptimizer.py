from AlgorithmImports import *
from scipy.optimize import *

class RiskParityPortfolioOptimizer:

    def __init__(self, minimum_weight=1e-05, maximum_weight=sys.float_info.max):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the RiskParityPortfolioOptimizer\n        Args:\n            minimum_weight(float): The lower bounds on portfolio weights\n            maximum_weight(float): The upper bounds on portfolio weights'
        self.minimum_weight = minimum_weight if minimum_weight >= 1e-05 else 1e-05
        self.maximum_weight = maximum_weight if maximum_weight >= minimum_weight else minimum_weight

    def Optimize(self, historicalReturns, budget=None, covariance=None):
        if False:
            while True:
                i = 10
        '\n        Perform portfolio optimization for a provided matrix of historical returns and an array of expected returns\n        args:\n            historicalReturns: Matrix of annualized historical returns where each column represents a security and each row returns for the given date/time (size: K x N).\n            budget: Risk budget vector (size: K x 1).\n            covariance: Multi-dimensional array of double with the portfolio covariance of annualized returns (size: K x K).\n        Returns:\n            Array of double with the portfolio weights (size: K x 1)\n        '
        if covariance is None:
            covariance = np.cov(historicalReturns.T)
        size = historicalReturns.columns.size
        x0 = np.array(size * [1.0 / size])
        budget = budget if budget is not None else x0
        objective = lambda weights: 0.5 * weights.T @ covariance @ weights - budget.T @ np.log(weights)
        gradient = lambda weights: covariance @ weights - budget / weights
        hessian = lambda weights: covariance + np.diag((budget / weights ** 2).flatten())
        solver = minimize(objective, jac=gradient, hess=hessian, x0=x0, method='Newton-CG')
        if not solver['success']:
            return x0
        return np.clip(solver['x'] / np.sum(solver['x']), self.minimum_weight, self.maximum_weight)