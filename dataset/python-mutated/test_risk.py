import datetime
import pandas as pd
import numpy as np
from zipline.utils import factory
from zipline.finance.trading import SimulationParameters
import zipline.testing.fixtures as zf
from zipline.finance.metrics import _ClassicRiskMetrics as ClassicRiskMetrics
RETURNS_BASE = 0.01
RETURNS = [RETURNS_BASE] * 251
BENCHMARK_BASE = 0.005
BENCHMARK = [BENCHMARK_BASE] * 251
DECIMAL_PLACES = 8
PERIODS = ['one_month', 'three_month', 'six_month', 'twelve_month']

class TestRisk(zf.WithBenchmarkReturns, zf.ZiplineTestCase):

    def init_instance_fixtures(self):
        if False:
            print('Hello World!')
        super(TestRisk, self).init_instance_fixtures()
        self.start_session = pd.Timestamp('2006-01-01', tz='UTC')
        self.end_session = self.trading_calendar.minute_to_session_label(pd.Timestamp('2006-12-31', tz='UTC'), direction='previous')
        self.sim_params = SimulationParameters(start_session=self.start_session, end_session=self.end_session, trading_calendar=self.trading_calendar)
        self.algo_returns = factory.create_returns_from_list(RETURNS, self.sim_params)
        self.benchmark_returns = factory.create_returns_from_list(BENCHMARK, self.sim_params)
        self.metrics = ClassicRiskMetrics.risk_report(algorithm_returns=self.algo_returns, benchmark_returns=self.benchmark_returns, algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index))

    def test_factory(self):
        if False:
            while True:
                i = 10
        returns = [0.1] * 100
        r_objects = factory.create_returns_from_list(returns, self.sim_params)
        self.assertLessEqual(r_objects.index[-1], pd.Timestamp('2006-12-31', tz='UTC'))

    def test_drawdown(self):
        if False:
            print('Hello World!')
        for period in PERIODS:
            self.assertTrue(all((x['max_drawdown'] == 0 for x in self.metrics[period])))

    def test_benchmark_returns_06(self):
        if False:
            i = 10
            return i + 15
        for (period, period_len) in zip(PERIODS, [1, 3, 6, 12]):
            np.testing.assert_almost_equal([x['benchmark_period_return'] for x in self.metrics[period]], [(1 + BENCHMARK_BASE) ** x['trading_days'] - 1 for x in self.metrics[period]], DECIMAL_PLACES)

    def test_trading_days(self):
        if False:
            return 10
        self.assertEqual([x['trading_days'] for x in self.metrics['twelve_month']], [251])
        self.assertEqual([x['trading_days'] for x in self.metrics['one_month']], [20, 19, 23, 19, 22, 22, 20, 23, 20, 22, 21, 20])

    def test_benchmark_volatility(self):
        if False:
            while True:
                i = 10
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['benchmark_volatility'], float) for x in self.metrics[period])))

    def test_algorithm_returns(self):
        if False:
            print('Hello World!')
        for period in PERIODS:
            np.testing.assert_almost_equal([x['algorithm_period_return'] for x in self.metrics[period]], [(1 + RETURNS_BASE) ** x['trading_days'] - 1 for x in self.metrics[period]], DECIMAL_PLACES)

    def test_algorithm_volatility(self):
        if False:
            i = 10
            return i + 15
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['algo_volatility'], float) for x in self.metrics[period])))

    def test_algorithm_sharpe(self):
        if False:
            while True:
                i = 10
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['sharpe'], float) for x in self.metrics[period])))

    def test_algorithm_sortino(self):
        if False:
            print('Hello World!')
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['sortino'], float) or x['sortino'] is None for x in self.metrics[period])))

    def test_algorithm_beta(self):
        if False:
            i = 10
            return i + 15
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['beta'], float) or x['beta'] is None for x in self.metrics[period])))

    def test_algorithm_alpha(self):
        if False:
            print('Hello World!')
        for period in PERIODS:
            self.assertTrue(all((isinstance(x['alpha'], float) or x['alpha'] is None for x in self.metrics[period])))

    def test_treasury_returns(self):
        if False:
            for i in range(10):
                print('nop')
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = ClassicRiskMetrics.risk_report(algorithm_returns=returns, benchmark_returns=self.benchmark_returns, algorithm_leverages=pd.Series(0.0, index=returns.index))
        for period in PERIODS:
            self.assertEqual([x['treasury_period_return'] for x in metrics[period]], [0.0] * len(metrics[period]))

    def test_benchmarkrange(self):
        if False:
            i = 10
            return i + 15
        start_session = self.trading_calendar.minute_to_session_label(pd.Timestamp('2008-01-01', tz='UTC'))
        end_session = self.trading_calendar.minute_to_session_label(pd.Timestamp('2010-01-01', tz='UTC'), direction='previous')
        sim_params = SimulationParameters(start_session=start_session, end_session=end_session, trading_calendar=self.trading_calendar)
        returns = factory.create_returns_from_range(sim_params)
        metrics = ClassicRiskMetrics.risk_report(algorithm_returns=returns, benchmark_returns=self.BENCHMARK_RETURNS, algorithm_leverages=pd.Series(0.0, index=returns.index))
        self.check_metrics(metrics, 24, start_session)

    def test_partial_month(self):
        if False:
            for i in range(10):
                print('nop')
        start_session = self.trading_calendar.minute_to_session_label(pd.Timestamp('1993-02-01', tz='UTC'))
        total_days = 365 * 5 + 2
        end_session = start_session + datetime.timedelta(days=total_days)
        sim_params90s = SimulationParameters(start_session=start_session, end_session=end_session, trading_calendar=self.trading_calendar)
        returns = factory.create_returns_from_range(sim_params90s)
        returns = returns[:-10]
        metrics = ClassicRiskMetrics.risk_report(algorithm_returns=returns, benchmark_returns=self.BENCHMARK_RETURNS, algorithm_leverages=pd.Series(0.0, index=returns.index))
        total_months = 60
        self.check_metrics(metrics, total_months, start_session)

    def check_metrics(self, metrics, total_months, start_date):
        if False:
            for i in range(10):
                print('nop')
        '\n        confirm that the right number of riskmetrics were calculated for each\n        window length.\n        '
        for (period, length) in zip(PERIODS, [1, 3, 6, 12]):
            self.assert_range_length(metrics[period], total_months, length, start_date)

    def assert_month(self, start_month, actual_end_month):
        if False:
            print('Hello World!')
        if start_month == 1:
            expected_end_month = 12
        else:
            expected_end_month = start_month - 1
        self.assertEqual(expected_end_month, actual_end_month)

    def assert_range_length(self, col, total_months, period_length, start_date):
        if False:
            i = 10
            return i + 15
        if period_length > total_months:
            self.assertFalse(col)
        else:
            period_end = pd.Timestamp(col[-1]['period_label'], tz='utc')
            self.assertEqual(len(col), total_months - (period_length - 1), 'mismatch for total months - expected:{total_months}/actual:{actual}, period:{period_length}, start:{start_date}, calculated end:{end}'.format(total_months=total_months, period_length=period_length, start_date=start_date, end=period_end, actual=len(col)))
            self.assert_month(start_date.month, period_end.month)

    def test_algorithm_leverages(self):
        if False:
            while True:
                i = 10
        for (period, expected_len) in zip(PERIODS, [12, 10, 7, 1]):
            self.assertEqual([x['max_leverage'] for x in self.metrics[period]], [0.0] * expected_len)
        test_period = ClassicRiskMetrics.risk_metric_period(start_session=self.start_session, end_session=self.end_session, algorithm_returns=self.algo_returns, benchmark_returns=self.benchmark_returns, algorithm_leverages=pd.Series([0.01, 0.02, 0.03]))
        self.assertEqual(test_period['max_leverage'], 0.03)

    def test_sharpe_value_when_null(self):
        if False:
            for i in range(10):
                print('nop')
        null_returns = factory.create_returns_from_list([0.0] * 251, self.sim_params)
        test_period = ClassicRiskMetrics.risk_metric_period(start_session=self.start_session, end_session=self.end_session, algorithm_returns=null_returns, benchmark_returns=self.benchmark_returns, algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index))
        self.assertEqual(test_period['sharpe'], 0.0)

    def test_sharpe_value_when_benchmark_null(self):
        if False:
            return 10
        null_returns = factory.create_returns_from_list([0.0] * 251, self.sim_params)
        test_period = ClassicRiskMetrics.risk_metric_period(start_session=self.start_session, end_session=self.end_session, algorithm_returns=null_returns, benchmark_returns=null_returns, algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index))
        self.assertEqual(test_period['sharpe'], 0.0)

    def test_representation(self):
        if False:
            print('Hello World!')
        test_period = ClassicRiskMetrics.risk_metric_period(start_session=self.start_session, end_session=self.end_session, algorithm_returns=self.algo_returns, benchmark_returns=self.benchmark_returns, algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index))
        metrics = {'algorithm_period_return', 'benchmark_period_return', 'treasury_period_return', 'period_label', 'excess_return', 'trading_days', 'benchmark_volatility', 'algo_volatility', 'sharpe', 'sortino', 'beta', 'alpha', 'max_drawdown', 'max_leverage'}
        self.assertEqual(set(test_period), metrics)