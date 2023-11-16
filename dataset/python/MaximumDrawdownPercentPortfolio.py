# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *

class MaximumDrawdownPercentPortfolio(RiskManagementModel):
    '''Provides an implementation of IRiskManagementModel that limits the drawdown of the portfolio to the specified percentage.'''

    def __init__(self, maximumDrawdownPercent = 0.05, isTrailing = False):
        '''Initializes a new instance of the MaximumDrawdownPercentPortfolio class
        Args:
            maximumDrawdownPercent: The maximum percentage drawdown allowed for algorithm portfolio compared with starting value, defaults to 5% drawdown</param>
            isTrailing: If "false", the drawdown will be relative to the starting value of the portfolio.
                        If "true", the drawdown will be relative the last maximum portfolio value'''
        self.maximumDrawdownPercent = -abs(maximumDrawdownPercent)
        self.isTrailing = isTrailing
        self.initialised = False
        self.portfolioHigh = 0

    def ManageRisk(self, algorithm, targets):
        '''Manages the algorithm's risk at each time step
        Args:
            algorithm: The algorithm instance
            targets: The current portfolio targets to be assessed for risk'''
        currentValue = algorithm.Portfolio.TotalPortfolioValue

        if not self.initialised:
            self.portfolioHigh = currentValue   # Set initial portfolio value
            self.initialised = True

        # Update trailing high value if in trailing mode
        if self.isTrailing and self.portfolioHigh < currentValue:
            self.portfolioHigh = currentValue
            return []   # return if new high reached

        pnl = self.GetTotalDrawdownPercent(currentValue)
        if pnl < self.maximumDrawdownPercent and len(targets) != 0:
            self.initialised = False # reset the trailing high value for restart investing on next rebalcing period

            risk_adjusted_targets = []
            for target in targets:
                symbol = target.Symbol

                # Cancel insights
                algorithm.Insights.Cancel([symbol])

                # liquidate
                risk_adjusted_targets.append(PortfolioTarget(symbol, 0))
            return risk_adjusted_targets

        return []

    def GetTotalDrawdownPercent(self, currentValue):
        return (float(currentValue) / float(self.portfolioHigh)) - 1.0
