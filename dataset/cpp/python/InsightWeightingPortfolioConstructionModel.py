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
from EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel, PortfolioBias

class InsightWeightingPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    '''Provides an implementation of IPortfolioConstructionModel that generates percent targets based on the
    Insight.Weight. The target percent holdings of each Symbol is given by the Insight.Weight from the last
    active Insight for that symbol.
    For insights of direction InsightDirection.Up, long targets are returned and for insights of direction
    InsightDirection.Down, short targets are returned.
    If the sum of all the last active Insight per symbol is bigger than 1, it will factor down each target
    percent holdings proportionally so the sum is 1.
    It will ignore Insight that have no Insight.Weight value.'''

    def __init__(self, rebalance = Resolution.Daily, portfolioBias = PortfolioBias.LongShort):
        '''Initialize a new instance of InsightWeightingPortfolioConstructionModel
        Args:
            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.
                              If None will be ignored.
                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.
                              The function returns null if unknown, in which case the function will be called again in the
                              next loop. Returning current time will trigger rebalance.
            portfolioBias: Specifies the bias of the portfolio (Short, Long/Short, Long)'''
        super().__init__(rebalance, portfolioBias)

    def ShouldCreateTargetForInsight(self, insight):
        '''Method that will determine if the portfolio construction model should create a
        target for this insight
        Args:
            insight: The insight to create a target for'''
        # Ignore insights that don't have Weight value
        return insight.Weight is not None

    def DetermineTargetPercent(self, activeInsights):
        '''Will determine the target percent for each insight
        Args:
            activeInsights: The active insights to generate a target for'''
        result = {}

        # We will adjust weights proportionally in case the sum is > 1 so it sums to 1.
        weightSums = sum(self.GetValue(insight) for insight in activeInsights if self.RespectPortfolioBias(insight))
        weightFactor = 1.0
        if weightSums > 1:
            weightFactor = 1 / weightSums
        for insight in activeInsights:
            result[insight] = (insight.Direction if self.RespectPortfolioBias(insight) else InsightDirection.Flat) * self.GetValue(insight) * weightFactor
        return result

    def GetValue(self, insight):
        '''Method that will determine which member will be used to compute the weights and gets its value
        Args:
            insight: The insight to create a target for
        Returns:
            The value of the selected insight member'''
        return abs(insight.Weight)
