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
# limitations under the License

from AlgorithmImports import *

### <summary>
### This regression algorithm tests In The Money (ITM) future option expiry for short puts.
### We expect 3 orders from the algorithm, which are:
###
###   * Initial entry, sell ES Put Option (expiring ITM)
###   * Option assignment, buy 1 contract of the underlying (ES)
###   * Future contract expiry, liquidation (sell 1 ES future)
###
### Additionally, we test delistings for future options and assert that our
### portfolio holdings reflect the orders the algorithm has submitted.
### </summary>
class FutureOptionShortPutITMExpiryRegressionAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 5)
        self.SetEndDate(2020, 6, 30)

        self.es19m20 = self.AddFutureContract(
            Symbol.CreateFuture(
                Futures.Indices.SP500EMini,
                Market.CME,
                datetime(2020, 6, 19)),
            Resolution.Minute).Symbol

        # Select a future option expiring ITM, and adds it to the algorithm.
        self.esOption = self.AddFutureOptionContract(
            list(
                sorted(
                    [x for x in self.OptionChainProvider.GetOptionContractList(self.es19m20, self.Time) if x.ID.StrikePrice <= 3400.0 and x.ID.OptionRight == OptionRight.Put],
                    key=lambda x: x.ID.StrikePrice,
                    reverse=True
                )
            )[0], Resolution.Minute).Symbol

        self.expectedContract = Symbol.CreateOption(self.es19m20, Market.CME, OptionStyle.American, OptionRight.Put, 3400.0, datetime(2020, 6, 19))
        if self.esOption != self.expectedContract:
            raise AssertionError(f"Contract {self.expectedContract} was not found in the chain")

        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.es19m20, 1), self.ScheduledMarketOrder)

    def ScheduledMarketOrder(self):
        self.MarketOrder(self.esOption, -1)

    def OnData(self, data: Slice):
        # Assert delistings, so that we can make sure that we receive the delisting warnings at
        # the expected time. These assertions detect bug #4872
        for delisting in data.Delistings.Values:
            if delisting.Type == DelistingType.Warning:
                if delisting.Time != datetime(2020, 6, 19):
                    raise AssertionError(f"Delisting warning issued at unexpected date: {delisting.Time}")

            if delisting.Type == DelistingType.Delisted:
                if delisting.Time != datetime(2020, 6, 20):
                    raise AssertionError(f"Delisting happened at unexpected date: {delisting.Time}")


    def OnOrderEvent(self, orderEvent: OrderEvent):
        if orderEvent.Status != OrderStatus.Filled:
            # There's lots of noise with OnOrderEvent, but we're only interested in fills.
            return

        if not self.Securities.ContainsKey(orderEvent.Symbol):
            raise AssertionError(f"Order event Symbol not found in Securities collection: {orderEvent.Symbol}")

        security = self.Securities[orderEvent.Symbol]
        if security.Symbol == self.es19m20:
            self.AssertFutureOptionOrderExercise(orderEvent, security, self.Securities[self.expectedContract])

        elif security.Symbol == self.expectedContract:
            self.AssertFutureOptionContractOrder(orderEvent, security)

        else:
            raise AssertionError(f"Received order event for unknown Symbol: {orderEvent.Symbol}")

        self.Log(f"{orderEvent}")

    def AssertFutureOptionOrderExercise(self, orderEvent: OrderEvent, future: Security, optionContract: Security):
        if "Assignment" in orderEvent.Message:
            if orderEvent.FillPrice != 3400.0:
                raise AssertionError("Option was not assigned at expected strike price (3400)")

            if orderEvent.Direction != OrderDirection.Buy or future.Holdings.Quantity != 1:
                raise AssertionError(f"Expected Qty: 1 futures holdings for assigned future {future.Symbol}, found {future.Holdings.Quantity}")

            return

        if orderEvent.Direction == OrderDirection.Sell and future.Holdings.Quantity != 0:
            # We buy back the underlying at expiration, so we expect a neutral position then
            raise AssertionError(f"Expected no holdings when liquidating future contract {future.Symbol}")

    def AssertFutureOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != -1:
            raise AssertionError(f"No holdings were created for option contract {option.Symbol}")

        if orderEvent.IsAssignment and option.Holdings.Quantity != 0:
            raise AssertionError(f"Holdings were found after option contract was assigned: {option.Symbol}")

    def OnEndOfAlgorithm(self):
        if self.Portfolio.Invested:
            raise AssertionError(f"Expected no holdings at end of algorithm, but are invested in: {', '.join([str(i.ID) for i in self.Portfolio.Keys])}")
