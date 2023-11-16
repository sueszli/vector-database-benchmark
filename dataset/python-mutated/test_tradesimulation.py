import pandas as pd
from zipline.gens.sim_engine import BEFORE_TRADING_START_BAR
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.finance import metrics
from zipline.finance.trading import SimulationParameters
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.testing.core import parameter_space
import zipline.testing.fixtures as zf

class TestBeforeTradingStartTiming(zf.WithMakeAlgo, zf.WithTradingSessions, zf.ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1,)
    BENCHMARK_SID = 1
    START_DATE = pd.Timestamp('2016-03-10', tz='UTC')
    END_DATE = pd.Timestamp('2016-03-15', tz='UTC')

    @parameter_space(num_sessions=[1, 2, 3], data_frequency=['daily', 'minute'], emission_rate=['daily', 'minute'], __fail_fast=True)
    def test_before_trading_start_runs_at_8_45(self, num_sessions, data_frequency, emission_rate):
        if False:
            while True:
                i = 10
        bts_times = []

        def initialize(algo, data):
            if False:
                return 10
            pass

        def before_trading_start(algo, data):
            if False:
                i = 10
                return i + 15
            bts_times.append(algo.get_datetime())
        sim_params = SimulationParameters(start_session=self.nyse_sessions[1], end_session=self.nyse_sessions[num_sessions], data_frequency=data_frequency, emission_rate=emission_rate, trading_calendar=self.trading_calendar)
        self.run_algorithm(before_trading_start=before_trading_start, sim_params=sim_params)
        self.assertEqual(len(bts_times), num_sessions)
        expected_times = [pd.Timestamp('2016-03-11 8:45', tz='US/Eastern').tz_convert('UTC'), pd.Timestamp('2016-03-14 8:45', tz='US/Eastern').tz_convert('UTC'), pd.Timestamp('2016-03-15 8:45', tz='US/Eastern').tz_convert('UTC')]
        self.assertEqual(bts_times, expected_times[:num_sessions])

class BeforeTradingStartsOnlyClock(object):

    def __init__(self, bts_minute):
        if False:
            for i in range(10):
                print('nop')
        self.bts_minute = bts_minute

    def __iter__(self):
        if False:
            return 10
        yield (self.bts_minute, BEFORE_TRADING_START_BAR)

class TestBeforeTradingStartSimulationDt(zf.WithMakeAlgo, zf.ZiplineTestCase):
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_bts_simulation_dt(self):
        if False:
            while True:
                i = 10
        code = '\ndef initialize(context):\n    pass\n'
        algo = self.make_algo(script=code, metrics=metrics.load('none'))
        algo.metrics_tracker = algo._create_metrics_tracker()
        benchmark_source = algo._create_benchmark_source()
        algo.metrics_tracker.handle_start_of_simulation(benchmark_source)
        dt = pd.Timestamp('2016-08-04 9:13:14', tz='US/Eastern')
        algo_simulator = AlgorithmSimulator(algo, self.sim_params, self.data_portal, BeforeTradingStartsOnlyClock(dt), benchmark_source, NoRestrictions(), None)
        list(algo_simulator.transform())
        self.assertEqual(dt, algo_simulator.simulation_dt)