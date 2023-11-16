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
from SetHoldingsMultipleTargetsRegressionAlgorithm import SetHoldingsMultipleTargetsRegressionAlgorithm

### <summary>
### Regression algorithm testing GH feature 3790, using SetHoldings with a collection of targets
### which will be ordered by margin impact before being executed, with the objective of avoiding any
### margin errors
### Asserts that liquidateExistingHoldings equal false does not close positions inadvertedly (GH 7008) 
### </summary>
class SetHoldingsLiquidateExistingHoldingsMultipleTargetsRegressionAlgorithm(SetHoldingsMultipleTargetsRegressionAlgorithm):
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings([PortfolioTarget(self._spy, 0.8), PortfolioTarget(self._ibm, 0.2)],
                             liquidateExistingHoldings=True)
