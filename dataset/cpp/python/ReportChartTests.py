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

# You can run this test by first running `nPython.exe` (with mono or otherwise):
# $ ./nPython.exe ReportChartTests.py

import numpy as np
import pandas as pd
from datetime import datetime
from ReportCharts import ReportCharts

charts = ReportCharts()

## Test GetReturnsPerTrade
backtest = list(np.random.normal(0, 1, 1000))
live = list(np.random.normal(0.5, 1, 400))
result = charts.GetReturnsPerTrade([], [])
result = charts.GetReturnsPerTrade(backtest, [])
result = charts.GetReturnsPerTrade(backtest, live)

## Test GetCumulativeReturnsPlot
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01T00:00:00', periods=365)]
strategy = np.linspace(1, 25, 365)
benchmark = np.linspace(2, 26, 365)
backtest = [time, strategy, time, benchmark]

time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2013-10-01T00:00:00', periods=50)]
strategy = np.linspace(25, 29, 50)
benchmark = np.linspace(26, 30, 50)
live = [time, strategy, time, benchmark]

result = charts.GetCumulativeReturns()
result = charts.GetCumulativeReturns(backtest)
result = charts.GetCumulativeReturns(backtest, live)

## Test GetDailyReturnsPlot
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01T00:00:00', periods=365)]
data = list(np.random.normal(0, 1, 365))
backtest = [time, data]

time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2013-10-01T00:00:00', periods=120)]
data = list(np.random.normal(0.5, 1.5, 120))
live = [time, data]

empty = [[], []]
result = charts.GetDailyReturns(empty, empty)
result = charts.GetDailyReturns(backtest, empty)
result = charts.GetDailyReturns(backtest, live)

## Test GetMonthlyReturnsPlot
backtest = {'2016': [0.5, 0.7, 0.2, 0.23, 1.3, 1.45, 1.67, -2.3, -0.5, 1.23, 1.23, -3.5],
            '2017': [0.5, 0.7, 0.2, 0.23, 1.3, 1.45, 1.67, -2.3, -0.5, 1.23, 1.23, -3.5][::-1]}

live = {'2018': [0.5, 0.7, 0.2, 0.23, 1.3, 1.45, 1.67, -2.3, -0.5, 1.23, 1.23, -3.5],
        '2019': [1.5, 2.7, -3.2, -0.23, 4.3, -2.45, -1.67, 2.3, np.nan, np.nan, np.nan, np.nan]}

result = charts.GetMonthlyReturns({}, {})
result = charts.GetMonthlyReturns(backtest, pd.DataFrame())
result = charts.GetMonthlyReturns(backtest, live)

## Test GetAnnualReturnsPlot
time = ['2012', '2013', '2014', '2015', '2016']
strategy = list(np.random.normal(0, 1, 5))
backtest = [time, strategy]

time = ['2017', '2018']
strategy = list(np.random.normal(0.5, 1.5, 2))
live = [time, strategy]

empty = [[], []]
result = charts.GetAnnualReturns()
result = charts.GetAnnualReturns(backtest)
result = charts.GetAnnualReturns(backtest, live)

## Test GetDrawdownPlot
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01', periods=365)]
data = list(np.random.uniform(-5, 0, 365))
backtest = [time, data]
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2013-10-01', periods=100)]
data = list(np.random.uniform(-5, 0, 100))
live = [time, data]
worst = [{'Begin': datetime(2012, 10, 1), 'End': datetime(2012, 10, 11)},
         {'Begin': datetime(2012, 12, 1), 'End': datetime(2012, 12, 11)},
         {'Begin': datetime(2013, 3, 1), 'End': datetime(2013, 3, 11)},
         {'Begin': datetime(2013, 4, 1), 'End': datetime(2013, 4, 1)},
         {'Begin': datetime(2013, 6, 1), 'End': datetime(2013, 6, 11)}]
empty = [[], []]
result = charts.GetDrawdown(empty, empty, {})
result = charts.GetDrawdown(backtest, empty, worst)
result = charts.GetDrawdown(backtest, live, worst)

## Test GetCrisisPlots  (backtest only)
equity = list(np.linspace(1, 25, 365))
benchmark = list(np.linspace(2, 26, 365))
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01 00:00:00', periods=365)]
backtest = [time, equity, benchmark]

empty = [[], [], []]
result = charts.GetCrisisEventsPlots(empty, 'empty_crisis')
result = charts.GetCrisisEventsPlots(backtest, 'dummy_crisis')

## Test GetRollingBetaPlot
empty = [[], [], [], []]
twelve = [np.nan for x in range(180)] + list(np.random.uniform(-1, 1, 185))
six = list(np.random.uniform(-1, 1, 365))
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01 00:00:00', periods=365)]
backtest = [time, six, twelve]

result = charts.GetRollingBeta([time, six, time, twelve], empty)
result = charts.GetRollingBeta([time, six, [], []], empty)
result = charts.GetRollingBeta(empty, empty)

twelve = [np.nan for x in range(180)] + list(np.random.uniform(-1, 1, 185))
six = list(np.random.uniform(-1, 1, 365))
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2013-10-01 00:00:00', periods=365)]
live = [time, six, time, twelve]

result = charts.GetRollingBeta(live)

## Test GetRollingSharpeRatioPlot
six = list(np.random.uniform(1, 3, 365 * 2))
twelve = [np.nan for x in range(365)] + list(np.random.uniform(1, 3, 365))
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2012-10-01 00:00:00', periods=365 * 2)]
six_live = list(np.random.uniform(1, 3, 365 + 180))
twelve_live = [np.nan for x in range(180)] + list(np.random.uniform(1, 3, 365))
time_live = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2014-10-01 00:00:00', periods=365 + 180)]

empty = [[], [], [], []]
result = charts.GetRollingSharpeRatio([time, six, time, twelve], empty)
result = charts.GetRollingSharpeRatio([time, six, [], []], empty)
result = charts.GetRollingSharpeRatio([time, six, time, twelve], [time_live, six_live, time_live, twelve_live])
result = charts.GetRollingSharpeRatio([time, six, [], []], [time_live, six_live, time_live, twelve_live])
result = charts.GetRollingSharpeRatio([time, six, time, twelve], [time_live, six_live, [], []])
result = charts.GetRollingSharpeRatio([time, six, [], []], [time_live, six_live, [], []])

## Test GetAssetAllocationPlot
backtest = [['SPY', 'IBM', 'NFLX', 'AAPL'], [0.50, 0.25, 0.125, 0.125]]
live = [['SPY', 'IBM', 'AAPL'], [0.4, 0.4, 0.2]]
empty = [[], []]
result = charts.GetAssetAllocation(empty, empty)
result = charts.GetAssetAllocation(backtest, empty)
result = charts.GetAssetAllocation(backtest, live)

## Test GetLeveragePlot
backtest = [[pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2014-10-01', periods=365)],
            list(np.random.uniform(0.5, 1.5, 365))]
live = [[pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2015-10-01', periods=100)],
        list(np.random.uniform(0.5, 2, 100))]
empty = [[], []]
result = charts.GetLeverage(empty, empty)
result = charts.GetLeverage(backtest, empty)
result = charts.GetLeverage(backtest, live)

## Test GetExposurePlot
time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2014-10-01', periods=365)]
long_securities = list(ReportCharts.color_map.keys())
short_securities = long_securities
long = [np.random.uniform(0, 0.5, 365) for x in long_securities]
short = [np.random.uniform(-0.5, 0, 365) for x in short_securities]

live_time = [pd.Timestamp(x).to_pydatetime() for x in pd.date_range('2015-10-01', periods=100)]
live_long_securities = long_securities
live_short_securities = long_securities
live_long = [np.random.uniform(0, 0.5, 100) for x in live_long_securities]
live_short = [np.random.uniform(-0.5, -0, 100) for x in live_short_securities]

result = charts.GetExposure()
result = charts.GetExposure(time, long_securities = long_securities, long_data=long, short_securities=[], short_data=[list(np.zeros(len(long[0])))])
result = charts.GetExposure(time, long_securities=[], long_data=[list(np.zeros(len(short[0])))], short_securities = short_securities, short_data=short)
result = charts.GetExposure(time, long_securities, short_securities, long, short)
result = charts.GetExposure(time, long_securities, short_securities, long, short,
                                live_time, live_long_securities, live_short_securities,
                                live_long, live_short)
