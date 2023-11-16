__author__ = 'saeedamen'
import numpy as np
import pandas as pd
from finmarketpy.util.marketconstants import MarketConstants
from findatapy.util import SwimPool
from findatapy.util import LoggerManager
import pickle
try:
    import blosc
except:
    pass
import pickle
market_constants = MarketConstants()

class Backtest(object):
    """Conducts backtest for strategies trading assets. Assumes we have an input of total returns. Reports historical return statistics
    and returns time series.

    """

    def __init__(self):
        if False:
            return 10
        self._pnl = None
        self._portfolio = None
        return

    def calculate_diagnostic_trading_PnL(self, asset_a_df, signal_df, further_df=[], further_df_labels=[]):
        if False:
            i = 10
            return i + 15
        'Calculates P&L table which can be used for debugging purposes.\n\n        The table is populated with asset, signal and further dataframes provided by the user, can be used to check signalling methodology.\n        It does not apply parameters such as transaction costs, vol adjusment and so on.\n\n        Parameters\n        ----------\n        asset_a_df : DataFrame\n            Asset prices\n\n        signal_df : DataFrame\n            Trade signals (typically +1, -1, 0 etc)\n\n        further_df : DataFrame\n            Further dataframes user wishes to output in the diagnostic output (typically inputs for the signals)\n\n        further_df_labels\n            Labels to append to the further dataframes\n\n        Returns\n        -------\n        DataFrame with asset, trading signals and returns of the trading strategy for diagnostic purposes\n\n        '
        calculations = Calculations()
        asset_rets_df = calculations.calculate_returns(asset_a_df)
        strategy_rets = calculations.calculate_signal_returns(signal_df, asset_rets_df)
        reset_points = (signal_df - signal_df.shift(1)).abs()
        asset_a_df_entry = asset_a_df.copy(deep=True)
        asset_a_df_entry[reset_points == 0] = np.nan
        asset_a_df_entry = asset_a_df_entry.ffill()
        asset_a_df_entry.columns = [x + '_entry' for x in asset_a_df_entry.columns]
        asset_rets_df.columns = [x + '_asset_rets' for x in asset_rets_df.columns]
        strategy_rets.columns = [x + '_strat_rets' for x in strategy_rets.columns]
        signal_df.columns = [x + '_final_signal' for x in signal_df.columns]
        for i in range(0, len(further_df)):
            further_df[i].columns = [x + '_' + further_df_labels[i] for x in further_df[i].columns]
        flatten_df = [asset_a_df, asset_a_df_entry, asset_rets_df, strategy_rets, signal_df]
        for f in further_df:
            flatten_df.append(f)
        return calculations.join(flatten_df, how='outer')

    def calculate_trading_PnL(self, br, asset_a_df, signal_df, contract_value_df, run_in_parallel):
        if False:
            i = 10
            return i + 15
        'Calculates P&L of a trading strategy and statistics to be retrieved later\n\n        Calculates the P&L for each asset/signal combination and also for the finally strategy applying appropriate\n        weighting in the portfolio, depending on predefined parameters, for example:\n            static weighting for each asset\n            static weighting for each asset + vol weighting for each asset\n            static weighting for each asset + vol weighting for each asset + vol weighting for the portfolio\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for the backtest specifying start date, finish data, transaction costs etc.\n\n        asset_a_df : pd.DataFrame\n            Asset prices to be traded\n\n        signal_df : pd.DataFrame\n            Signals for the trading strategy\n\n        contract_value_df : pd.DataFrame\n            Daily size of contracts\n        '
        calculations = Calculations()
        risk_engine = RiskEngine()
        logger = LoggerManager().getLogger(__name__)
        logger.info('Calculating trading P&L...')
        signal_df = signal_df.shift(br.signal_delay)
        (asset_df, signal_df) = calculations.join_left_fill_right(asset_a_df, signal_df)
        if contract_value_df is not None:
            (asset_df, contract_value_df) = asset_df.align(contract_value_df, join='left', axis='index')
            contract_value_df = contract_value_df.fillna(method='ffill')
        non_trading_days = np.isnan(asset_df.values)
        signal_df = signal_df.mask(non_trading_days)
        signal_df = signal_df.fillna(method='ffill')
        tc = br.spot_tc_bp
        rc = br.spot_rc_bp
        signal_cols = signal_df.columns.values
        asset_df_cols = asset_df.columns.values
        pnl_cols = []
        for i in range(0, len(asset_df_cols)):
            pnl_cols.append(asset_df_cols[i] + ' / ' + signal_cols[i])
        asset_df = asset_df.fillna(method='ffill')
        returns_df = calculations.calculate_returns(asset_df)
        if br.take_profit is not None and br.stop_loss is not None:
            returns_df = calculations.calculate_returns(asset_df)
            temp_strategy_rets_df = calculations.calculate_signal_returns_as_matrix(signal_df, returns_df)
            trade_rets_df = calculations.calculate_cum_rets_trades(signal_df, temp_strategy_rets_df)
            signal_df = calculations.calculate_risk_stop_signals(signal_df, trade_rets_df, br.stop_loss, br.take_profit)
            signal_df = signal_df.mask(non_trading_days)
            signal_df = signal_df.fillna(method='ffill')
        if br.portfolio_weight_construction is None:
            pwc = PortfolioWeightConstruction(br=br)
        else:
            pwc = br.portfolio_weight_construction
        (portfolio_signal_before_weighting, portfolio_signal, portfolio_leverage_df, portfolio, individual_leverage_df, pnl) = pwc.optimize_portfolio_weights(returns_df, signal_df, pnl_cols)
        (portfolio_total_longs, portfolio_total_shorts, portfolio_net_exposure, portfolio_total_exposure) = self.calculate_exposures(portfolio_signal)
        position_clip_adjustment = risk_engine.calculate_position_clip_adjustment(portfolio_net_exposure, portfolio_total_exposure, br)
        if position_clip_adjustment is not None:
            length_cols = len(signal_df.columns)
            position_clip_adjustment_matrix = np.transpose(np.repeat(position_clip_adjustment.values.flatten()[np.newaxis, :], length_cols, 0))
            portfolio_signal_before_weighting = pd.DataFrame(data=portfolio_signal_before_weighting.values * position_clip_adjustment_matrix, index=portfolio_signal_before_weighting.index, columns=portfolio_signal_before_weighting.columns)
            portfolio_signal = pd.DataFrame(data=portfolio_signal.values * position_clip_adjustment_matrix, index=portfolio_signal.index, columns=portfolio_signal.columns)
            portfolio_leverage_df = pd.DataFrame(data=portfolio_leverage_df.values * position_clip_adjustment.values, index=portfolio_leverage_df.index, columns=portfolio_leverage_df.columns)
            (portfolio_total_longs, portfolio_total_shorts, portfolio_net_exposure, portfolio_total_exposure) = self.calculate_exposures(portfolio_signal)
        portfolio = calculations.calculate_signal_returns_with_tc_matrix(portfolio_leverage_df, portfolio, tc=tc, rc=rc)
        self._signal = self._filter_by_plot_start_finish_date(signal_df, br)
        self._portfolio_signal = self._filter_by_plot_start_finish_date(portfolio_signal, br)
        self._portfolio_leverage = self._filter_by_plot_start_finish_date(portfolio_leverage_df, br)
        self._portfolio = self._filter_by_plot_start_finish_date(portfolio, br)
        self._portfolio_trade = self._portfolio_signal - self._portfolio_signal.shift(1)
        self._portfolio_signal_notional = None
        self._portfolio_signal_trade_notional = None
        self._portfolio_signal_contracts = None
        self._portfolio_signal_trade_contracts = None
        self._portfolio_total_longs = self._filter_by_plot_start_finish_date(portfolio_total_longs, br)
        self._portfolio_total_shorts = self._filter_by_plot_start_finish_date(portfolio_total_shorts, br)
        self._portfolio_net_exposure = self._filter_by_plot_start_finish_date(portfolio_net_exposure, br)
        self._portfolio_total_exposure = self._filter_by_plot_start_finish_date(portfolio_total_exposure, br)
        self._portfolio_total_longs_notional = None
        self._portfolio_total_shorts_notional = None
        self._portfolio_net_exposure_notional = None
        self._portfolio_total_exposure_notional = None
        self._portfolio_signal_trade_notional_sizes = None
        self._pnl = pnl
        self._individual_leverage = self._filter_by_plot_start_finish_date(individual_leverage_df, br)
        self._components_pnl = self._filter_by_plot_start_finish_date(calculations.calculate_signal_returns_with_tc_matrix(portfolio_signal_before_weighting, returns_df, tc=tc, rc=rc), br)
        self._components_pnl.columns = pnl_cols
        self._pnl_trades = None
        self._components_pnl_trades = None
        self._trade_no = None
        self._portfolio_trade_no = None
        self._portfolio.columns = ['Port']
        self._pnl_ret_stats = RetStats(self._pnl, br.ann_factor, br.resample_ann_factor)
        self._components_pnl_ret_stats = RetStats(self._components_pnl, br.ann_factor, br.resample_ann_factor)
        self._portfolio_ret_stats = RetStats(self._portfolio, br.ann_factor, br.resample_ann_factor)
        if br.portfolio_notional_size is not None:
            self._portfolio_signal_notional = self._portfolio_signal * br.portfolio_notional_size
            self._portfolio_signal_trade_notional = self._portfolio_signal_notional - self._portfolio_signal_notional.shift(1)
            df_trades_sizes = pd.DataFrame()
            rounded_portfolio_signal_trade_notional = self._portfolio_signal_trade_notional.round(2)
            for k in rounded_portfolio_signal_trade_notional.columns:
                df_trades_sizes[k] = pd.value_counts(rounded_portfolio_signal_trade_notional[k], sort=True)
            df_trades_sizes = df_trades_sizes[df_trades_sizes.index != 0]
            self._portfolio_signal_trade_notional_sizes = df_trades_sizes
            self._portfolio_total_longs_notional = portfolio_total_longs * br.portfolio_notional_size
            self._portfolio_total_shorts_notional = portfolio_total_shorts * br.portfolio_notional_size
            self._portfolio_net_exposure_notional = portfolio_net_exposure * br.portfolio_notional_size
            self._portfolio_total_exposure_notional = portfolio_total_exposure * br.portfolio_notional_size
            notional_copy = self._portfolio_signal_notional.copy(deep=True)
            notional_copy_cols = [x.split('.')[0] for x in notional_copy.columns]
            notional_copy_cols = [x + '.contract-value' for x in notional_copy_cols]
            notional_copy.columns = notional_copy_cols
            if contract_value_df is not None:
                contract_value_df = contract_value_df[notional_copy_cols]
                (notional_df, contract_value_df) = notional_copy.align(contract_value_df, join='left', axis='index')
                self._portfolio_signal_contracts = notional_df / contract_value_df
                self._portfolio_signal_contracts.columns = self._portfolio_signal_notional.columns
                self._portfolio_signal_trade_contracts = self._portfolio_signal_contracts - self._portfolio_signal_contracts.shift(1)
        logger.info('Cumulative index calculations')
        if False:
            swim_pool = SwimPool(multiprocessing_library=market_constants.multiprocessing_library)
            pool = swim_pool.create_pool(thread_technique=market_constants.backtest_thread_technique, thread_no=market_constants.backtest_thread_no[market_constants.generic_plat])
            r1 = pool.apply_async(self._pnl_ret_stats.calculate_ret_stats)
            r2 = pool.apply_async(self._components_pnl_ret_stats.calculate_ret_stats)
            r3 = pool.apply_async(self._portfolio_ret_stats.calculate_ret_stats)
            resultsA = pool.apply_async(calculations.create_mult_index, args=(self._pnl,))
            resultsB = pool.apply_async(calculations.create_mult_index, args=(self._components_pnl,))
            resultsC = pool.apply_async(calculations.create_mult_index, args=(self._portfolio,))
            swim_pool.close_pool(pool)
            self._pnl_ret_stats = r1.get()
            self._components_pnl_ret_stats = r2.get()
            self._portfolio_ret_stats = r3.get()
            self._pnl_cum = resultsA.get()
            self._components_pnl_cum = resultsB.get()
            self._portfolio_cum = resultsC.get()
        elif br.cum_index == 'mult':
            self._pnl_cum = calculations.create_mult_index(self._pnl)
            self._components_pnl_cum = calculations.create_mult_index(self._components_pnl)
            self._portfolio_cum = calculations.create_mult_index(self._portfolio)
        elif br.cum_index == 'add':
            self._pnl_cum = calculations.create_add_index(self._pnl)
            self._components_pnl_cum = calculations.create_add_index(self._components_pnl)
            self._portfolio_cum = calculations.create_add_index(self._portfolio)
        logger.info('Completed cumulative index calculations')
        self._pnl_cum.columns = pnl_cols
        self._components_pnl_cum.columns = pnl_cols
        self._portfolio_cum.columns = ['Port']

    def _filter_by_plot_start_finish_date(self, df, br):
        if False:
            for i in range(10):
                print('nop')
        if br.plot_start is None and br.plot_finish is None:
            return df
        elif df is None:
            return None
        else:
            filter = Filter()
            plot_start = br.start_date
            plot_finish = br.finish_date
            if br.plot_start is not None:
                plot_start = br.plot_start
            if br.plot_finish is not None:
                plot_finish = br.plot_finish
            return filter.filter_time_series_by_date(plot_start, plot_finish, df)

    def calculate_exposures(self, portfolio_signal):
        if False:
            while True:
                i = 10
        'Calculates time series for the total longs, short, net and absolute exposure on an aggregated portfolio basis.\n\n        Parameters\n        ----------\n        portfolio_signal : DataFrame\n            Signals for each asset in the portfolio after all weighting, portfolio & signal level volatility adjustments\n\n        Returns\n        -------\n        DataFrame (list)\n\n        '
        portfolio_total_longs = pd.DataFrame(portfolio_signal[portfolio_signal > 0].sum(axis=1))
        portfolio_total_shorts = pd.DataFrame(portfolio_signal[portfolio_signal < 0].sum(axis=1))
        portfolio_total_longs.columns = ['Total Longs']
        portfolio_total_shorts.columns = ['Total Shorts']
        portfolio_net_exposure = pd.DataFrame(index=portfolio_signal.index, columns=['Net Exposure'], data=portfolio_total_longs.values + portfolio_total_shorts.values)
        portfolio_total_exposure = pd.DataFrame(index=portfolio_signal.index, columns=['Total Exposure'], data=portfolio_total_longs.values - portfolio_total_shorts.values)
        return (portfolio_total_longs, portfolio_total_shorts, portfolio_net_exposure, portfolio_total_exposure)

    def backtest_output(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def pnl(self):
        if False:
            return 10
        'Gets P&L returns of all the individual sub_components of the model (before any portfolio level leverage is applied)\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        return self._pnl

    def trade_no(self):
        if False:
            while True:
                i = 10
        'Gets number of trades for each signal in the backtest (before\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        if self._trade_no is None:
            calculations = Calculations()
            self._trade_no = calculations.calculate_trade_no(self._signal)
        return self._trade_no

    def pnl_trades(self):
        if False:
            i = 10
            return i + 15
        'Gets P&L of each individual trade per signal\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        if self._pnl_trades is None:
            calculations = Calculations()
            self._pnl_trades = calculations.calculate_individual_trade_gains(self._signal, self._pnl)
        return self._pnl_trades

    def pnl_desc(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets P&L return statistics in a string format\n\n        Returns\n        -------\n        str\n        '
        return self._ret_stats_signals.summary()

    def pnl_ret_stats(self):
        if False:
            print('Hello World!')
        'Gets P&L return statistics of individual strategies as class to be queried\n\n        Returns\n        -------\n        TimeSeriesDesc\n        '
        return self._pnl_ret_stats

    def pnl_cum(self):
        if False:
            while True:
                i = 10
        'Gets P&L as a cumulative time series of individual assets\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._pnl_cum

    def components_pnl(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets P&L returns of all the individual subcomponents of the model (after portfolio level leverage is applied)\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        return self._components_pnl

    def components_pnl_trades(self):
        if False:
            print('Hello World!')
        'Gets P&L of each individual trade per signal\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        if self._components_pnl_trades is None:
            calculations = Calculations()
            self._components_pnl_trades = calculations.calculate_individual_trade_gains(self._signal, self._components_pnl)
        return self._components_pnl_trades

    def components_pnl_desc(self):
        if False:
            i = 10
            return i + 15
        'Gets P&L of individual components as return statistics in a string format\n\n        Returns\n        -------\n        str\n        '

    def components_pnl_ret_stats(self):
        if False:
            print('Hello World!')
        'Gets P&L return statistics of individual strategies as class to be queried\n\n        Returns\n        -------\n        TimeSeriesDesc\n        '
        return self._components_pnl_ret_stats

    def components_pnl_cum(self):
        if False:
            print('Hello World!')
        'Gets P&L as a cumulative time series of individual assets (after portfolio level leverage adjustments)\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._components_pnl_cum

    def portfolio_cum(self):
        if False:
            print('Hello World!')
        'Gets P&L as a cumulative time series of portfolio\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._portfolio_cum

    def portfolio_pnl(self):
        if False:
            i = 10
            return i + 15
        'Gets portfolio returns in raw form (ie. not indexed into cumulative form)\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._portfolio

    def portfolio_pnl_desc(self):
        if False:
            return 10
        'Gets P&L return statistics of portfolio as string\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._portfolio_ret_stats.summary()

    def portfolio_pnl_ret_stats(self):
        if False:
            i = 10
            return i + 15
        'Gets P&L return statistics of portfolio as class to be queried\n\n        Returns\n        -------\n        RetStats\n        '
        return self._portfolio_ret_stats

    def individual_leverage(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets leverage for each asset historically\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._individual_leverage

    def portfolio_leverage(self):
        if False:
            while True:
                i = 10
        'Gets the leverage for the portfolio\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._portfolio_leverage

    def portfolio_trade_no(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets number of trades for each signal in the backtest (after both signal and portfolio level vol adjustment)\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        if self._portfolio_trade_no is None:
            calculations = Calculations()
            self._portfolio_trade_no = calculations.calculate_trade_no(self._portfolio_signal)
        return self._portfolio_trade_no

    def portfolio_signal(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the signals (with individual leverage & portfolio leverage) for each asset, which\n        equates to what we would trade in practice\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_signal

    def portfolio_total_longs(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the total long exposure in the portfolio\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_longs

    def portfolio_total_shorts(self):
        if False:
            i = 10
            return i + 15
        'Gets the total short exposure in the portfolio\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_shorts

    def portfolio_net_exposure(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the total net exposure of the portfolio\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_net_exposure

    def portfolio_total_exposure(self):
        if False:
            print('Hello World!')
        'Gets the total absolute exposure of the portfolio\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_exposure

    def portfolio_total_longs_notional(self):
        if False:
            return 10
        'Gets the total long exposure in the portfolio scaled by notional\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_longs_notional

    def portfolio_total_shorts_notional(self):
        if False:
            print('Hello World!')
        'Gets the total short exposure in the portfolio scaled by notional\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_shorts_notional

    def portfolio_net_exposure_notional(self):
        if False:
            print('Hello World!')
        'Gets the total net exposure of the portfolio scaled by notional\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_net_exposure_notional

    def portfolio_total_exposure_notional(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the total absolute exposure of the portfolio scaled by notional\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_total_exposure_notional

    def portfolio_trade(self):
        if False:
            return 10
        "Gets the trades (with individual leverage & portfolio leverage) for each asset, which\n        we'd need to execute\n\n        Returns\n        -------\n        DataFrame\n        "
        return self._portfolio_trade

    def portfolio_signal_notional(self):
        if False:
            while True:
                i = 10
        'Gets the signals (with individual leverage & portfolio leverage) for each asset, which\n        equates to what we would have a positions in practice, scaled by a notional amount we have already specified\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_signal_notional

    def portfolio_trade_notional(self):
        if False:
            return 10
        "Gets the trades (with individual leverage & portfolio leverage) for each asset, which\n        we'd need to execute, scaled by a notional amount we have already specified\n\n        Returns\n        -------\n        DataFrame\n        "
        return self._portfolio_signal_trade_notional

    def portfolio_trade_notional_sizes(self):
        if False:
            i = 10
            return i + 15
        "Gets the number of trades (with individual leverage & portfolio leverage) for each asset, which\n        we'd need to execute, scaled by a notional amount we have already specified\n\n        Returns\n        -------\n        DataFrame\n        "
        return self._portfolio_signal_trade_notional_sizes

    def portfolio_signal_contracts(self):
        if False:
            return 10
        'Gets the signals (with individual leverage & portfolio leverage) for each asset, which\n        equates to what we would have a positions in practice, scaled by a notional amount and into contract sizes (eg. for futures)\n        which we need to specify in another dataframe\n\n        Returns\n        -------\n        DataFrame\n        '
        return self._portfolio_signal_contracts

    def portfolio_trade_contracts(self):
        if False:
            return 10
        "Gets the trades (with individual leverage & portfolio leverage) for each asset, which\n        we'd need to execute, scaled by a notional amount we have already specified and into contract sizes (eg. for futures)\n        which we need to specify in another dataframe\n\n        Returns\n        -------\n        DataFrame\n        "
        return self._portfolio_signal_trade_contracts

    def signal(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets signal for each asset (with individual leverage, but excluding portfolio leverage constraints) for each asset\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        return self._signal
import abc
import datetime
import copy
from chartpy import Chart, Style, ChartConstants
from finmarketpy.economics import TechParams
from findatapy.timeseries import Calculations, RetStats, Filter

class TradingModel(object):
    """Abstract class which wraps around Backtest, providing convenient functions for analaysis. Implement your own
    subclasses of this for your own strategy. See tradingmodelfxtrend_example.py for a simple implementation of a
    FX trend following strategy.
    """
    SAVE_FIGURES = True
    SHOW_CHARTS = True
    SHOW_TITLES = True
    DEFAULT_PLOT_ENGINE = ChartConstants().chartfactory_default_engine
    SCALE_FACTOR = ChartConstants().chartfactory_scale_factor
    WIDTH = ChartConstants().chartfactory_width
    HEIGHT = ChartConstants().chartfactory_height
    CHART_SOURCE = ChartConstants().chartfactory_source
    CHART_STYLE = Style()
    DUMP_CSV = ''
    DUMP_PATH = datetime.date.today().strftime('%Y%m%d') + ' '

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def load_parameters(self, br=None):
        if False:
            while True:
                i = 10
        'Fills parameters for the backtest, such as start-end dates, transaction costs etc. To\n        be implemented by subclass. Can overwrite it with our own BacktestRequest.\n        '
        return

    @abc.abstractmethod
    def load_assets(self, br=None):
        if False:
            print('Hello World!')
        'Loads time series for the assets to be traded and also for data for generating signals. Can overwrite it\n        with our own BacktestRequest.\n        '
        return

    @abc.abstractmethod
    def construct_signal(self, spot_df, spot_df2, tech_params, br, run_in_parallel=False):
        if False:
            return 10
        'Constructs signal from pre-loaded time series\n\n        Parameters\n        ----------\n        spot_df : pd.DataFrame\n            Market time series for generating signals\n\n        spot_df2 : pd.DataFrame\n            Market time series for generated signals (can be of different frequency)\n\n        tech_params : TechParams\n            Parameters for generating signals\n\n        run_in_parallel : bool\n            Allow signal calculation in parallel\n        '
        return

    def save_model(self, path):
        if False:
            while True:
                i = 10
        '\n        Save the model instance as as pickle.\n\n        :param path: path to pickle.\n        :return:\n        '
        pickle.dump(self, path)

    @staticmethod
    def load_model(path):
        if False:
            while True:
                i = 10
        '\n        Load the pickle of the saved model.\n        :param path: path to pickle.\n        :return: TradingModel instance.\n        '
        pkl = pickle.load(path)
        return pkl

    def construct_strategy(self, br=None, run_in_parallel=False):
        if False:
            while True:
                i = 10
        'Constructs the returns for all the strategies which have been specified.\n\n        It gets backtesting parameters from fill_backtest_request (although these can be overwritten\n        and then market data from fill_assets\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters which define the backtest (for example start date, end date, transaction costs etc.\n        '
        logger = LoggerManager().getLogger(__name__)
        if br is not None:
            pass
        elif hasattr(self, 'br'):
            br = self.br
        elif br is None:
            br = self.load_parameters()
        try:
            market_data = self.load_assets(br=br)
        except:
            market_data = self.load_assets()
        asset_df = market_data[0]
        spot_df = market_data[1]
        spot_df2 = market_data[2]
        basket_dict = market_data[3]
        contract_value_df = None
        if len(market_data) == 5:
            contract_value_df = market_data[4]
        if hasattr(br, 'tech_params'):
            tech_params = br.tech_params
        else:
            tech_params = TechParams()
        cum_results = pd.DataFrame(index=asset_df.index)
        port_leverage = pd.DataFrame(index=asset_df.index)
        from collections import OrderedDict
        ret_stats_results = OrderedDict()
        bask_results = {}
        bask_keys = basket_dict.keys()
        if market_constants.backtest_thread_no[market_constants.generic_plat] > 1 and run_in_parallel:
            swim_pool = SwimPool(multiprocessing_library=market_constants.multiprocessing_library)
            pool = swim_pool.create_pool(thread_technique=market_constants.backtest_thread_technique, thread_no=market_constants.backtest_thread_no[market_constants.generic_plat])
            mult_results = []
            for key in bask_keys:
                if key != self.FINAL_STRATEGY:
                    logger.info('Calculating (parallel) ' + key)
                    asset_cut_df = asset_df[[x + '.' + br.trading_field for x in basket_dict[key]]]
                    spot_cut_df = spot_df[[x + '.' + br.trading_field for x in basket_dict[key]]]
                    mult_results.append(pool.apply_async(self.construct_individual_strategy, args=(br, spot_cut_df, spot_df2, asset_cut_df, tech_params, key, contract_value_df, False, True)))
            logger.info('Calculating final strategy ' + self.FINAL_STRATEGY)
            asset_cut_df = asset_df[[x + '.' + br.trading_field for x in basket_dict[self.FINAL_STRATEGY]]]
            spot_cut_df = spot_df[[x + '.' + br.trading_field for x in basket_dict[self.FINAL_STRATEGY]]]
            (desc, results, leverage, stats, key, backtest) = self.construct_individual_strategy(br, spot_cut_df, spot_df2, asset_cut_df, tech_params, self.FINAL_STRATEGY, contract_value_df, True, False)
            results.columns = desc
            cum_results[results.columns[0]] = results
            port_leverage[results.columns[0]] = leverage
            ret_stats_results[key] = stats
            self._assign_final_strategy_results(results, backtest)
            for p in mult_results:
                (desc, results, leverage, stats, key, backtest) = p.get()
                results.columns = desc
                cum_results[results.columns[0]] = results
                port_leverage[results.columns[0]] = leverage
                ret_stats_results[key] = stats
                if key == self.FINAL_STRATEGY:
                    self._assign_final_strategy_results(results, backtest)
            try:
                swim_pool.close_pool(pool)
            except:
                pass
        else:
            for key in bask_keys:
                logger.info('Calculating (single thread) ' + key)
                asset_cut_df = asset_df[[x + '.' + br.trading_field for x in basket_dict[key]]]
                spot_cut_df = spot_df[[x + '.' + br.trading_field for x in basket_dict[key]]]
                (desc, results, leverage, ret_stats, key, backtest) = self.construct_individual_strategy(br, spot_cut_df, spot_df2, asset_cut_df, tech_params, key, contract_value_df, False, False)
                results.columns = desc
                cum_results[results.columns[0]] = results
                port_leverage[results.columns[0]] = leverage
                ret_stats_results[key] = ret_stats
                if key == self.FINAL_STRATEGY:
                    self._assign_final_strategy_results(results, backtest)
        benchmark = self.construct_strategy_benchmark()
        cum_results_benchmark = self.compare_strategy_vs_benchmark(br, cum_results, benchmark)
        self._strategy_group_benchmark_pnl_ret_stats = ret_stats_results
        try:
            ret_stats_list = ret_stats_results
            ret_stats_list['Benchmark'] = self._strategy_benchmark_pnl_ret_stats
            self._strategy_group_benchmark_pnl_ret_stats = ret_stats_list
        except:
            pass
        self._strategy_group_pnl = cum_results
        self._strategy_group_pnl_ret_stats = ret_stats_results
        self._strategy_group_leverage = port_leverage
        self._strategy_group_benchmark_pnl = cum_results_benchmark

    def _assign_final_strategy_results(self, results, backtest):
        if False:
            for i in range(10):
                print('nop')
        self._strategy_pnl = results
        self._strategy_components_pnl = backtest.components_pnl_cum()
        self._strategy_components_pnl_ret_stats = backtest.components_pnl_ret_stats().split_into_dict()
        self._individual_leverage = backtest.individual_leverage()
        self._strategy_pnl_ret_stats = backtest.portfolio_pnl_ret_stats()
        self._strategy_leverage = backtest.portfolio_leverage()
        self._strategy_signal = backtest.portfolio_signal()
        self._strategy_trade_no = backtest.portfolio_trade_no()
        self._strategy_trade = backtest.portfolio_trade()
        self._strategy_signal_notional = backtest.portfolio_signal_notional()
        self._strategy_trade_notional = backtest.portfolio_trade_notional()
        self._strategy_trade_notional_sizes = backtest.portfolio_trade_notional_sizes()
        self._strategy_signal_contracts = backtest.portfolio_signal_contracts()
        self._strategy_trade_contracts = backtest.portfolio_trade_contracts()
        self._strategy_group_pnl_trades = backtest.pnl_trades()
        self._strategy_pnl_trades_components = backtest.components_pnl_trades()
        self._strategy_total_longs = backtest.portfolio_total_longs()
        self._strategy_total_shorts = backtest.portfolio_total_shorts()
        self._strategy_net_exposure = backtest.portfolio_net_exposure()
        self._strategy_total_exposure = backtest.portfolio_total_exposure()
        self._strategy_total_longs_notional = backtest.portfolio_total_longs_notional()
        self._strategy_total_shorts_notional = backtest.portfolio_total_shorts_notional()
        self._strategy_net_exposure_notional = backtest.portfolio_net_exposure_notional()
        self._strategy_total_exposure_notional = backtest.portfolio_total_exposure_notional()

    def construct_individual_strategy(self, br, spot_df, spot_df2, asset_df, tech_params, key, contract_value_df, run_in_parallel, compress_output):
        if False:
            print('Hello World!')
        'Combines the signal with asset returns to find the returns of an individual strategy\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for backtest such as start and finish dates\n\n        spot_df : pd.DataFrame\n            Market time series for generating signals\n\n        spot_df2 : pd.DataFrame\n            Secondary Market time series for generated signals (can be of different frequency)\n\n        tech_params : TechParams\n            Parameters for generating signals\n\n        contract_value_df : pd.DataFrame\n            Dataframe with the contract sizes for each asset\n\n        Returns\n        -------\n        portfolio_cum : pd.DataFrame\n        backtest : Backtest\n        '
        backtest = Backtest()
        logger = LoggerManager().getLogger(__name__)
        logger.info('Calculating trading signals for ' + key + '...')
        signal = self.construct_signal(spot_df, spot_df2, tech_params, br, run_in_parallel=run_in_parallel)
        logger.info('Calculated trading signals for ' + key)
        backtest.calculate_trading_PnL(br, asset_df, signal, contract_value_df, run_in_parallel)
        if br.write_csv:
            backtest.pnl_cum().to_csv(self.DUMP_CSV + key + '.csv')
        if br.calc_stats:
            desc = [key + ' ' + str(backtest.portfolio_pnl_desc()[0])]
        else:
            desc = [key]
        if key == self.FINAL_STRATEGY and compress_output:
            logger.debug('Compressing ' + key)
            backtest = blosc.compress(pickle.dumps(backtest))
            logger.debug('Compressed ' + key)
        logger.info('Calculated for ' + key)
        if key != self.FINAL_STRATEGY:
            return (desc, backtest.portfolio_cum(), backtest.portfolio_leverage(), backtest.portfolio_pnl_ret_stats(), key, None)
        return (desc, backtest.portfolio_cum(), backtest.portfolio_leverage(), backtest.portfolio_pnl_ret_stats(), key, backtest)

    def compare_strategy_vs_benchmark(self, br, strategy_df, benchmark_df):
        if False:
            for i in range(10):
                print('nop')
        'Compares the trading strategy we are backtesting against a benchmark\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for backtest such as start and finish dates\n        strategy_df : pd.DataFrame\n            Strategy time series\n        benchmark_df : pd.DataFrame\n            Benchmark time series\n        '
        if br.include_benchmark:
            ret_stats = RetStats(br.resample_ann_factor)
            risk_engine = RiskEngine()
            filter = Filter()
            calculations = Calculations()
            benchmark_df.columns = [x + ' be' for x in benchmark_df.columns]
            (strategy_df, benchmark_df) = strategy_df.align(benchmark_df, join='left', axis=0)
            if br.portfolio_vol_adjust is True:
                benchmark_df = risk_engine.calculate_vol_adjusted_index_from_prices(benchmark_df, br=br)
            benchmark_df = benchmark_df.fillna(method='ffill')
            benchmark_df = self._filter_by_plot_start_finish_date(benchmark_df, br)
            ret_stats.calculate_ret_stats_from_prices(benchmark_df, br.ann_factor)
            if br.calc_stats:
                benchmark_df.columns = ret_stats.summary()
            strategy_benchmark_df = strategy_df.join(benchmark_df, how='inner')
            strategy_benchmark_df = strategy_benchmark_df.fillna(method='ffill')
            strategy_benchmark_df = self._filter_by_plot_start_finish_date(strategy_benchmark_df, br)
            if br.cum_index == 'mult':
                strategy_benchmark_df = calculations.create_mult_index_from_prices(strategy_benchmark_df)
            elif br.cum_index == 'add':
                strategy_benchmark_df = calculations.create_add_index_from_prices(strategy_benchmark_df)
            self._strategy_benchmark_pnl = benchmark_df
            self._strategy_benchmark_pnl_ret_stats = ret_stats
            return strategy_benchmark_df
        return strategy_df

    def _filter_by_plot_start_finish_date(self, df, br):
        if False:
            while True:
                i = 10
        if br.plot_start is None and br.plot_finish is None:
            return df
        else:
            filter = Filter()
            plot_start = br.start_date
            plot_finish = br.finish_date
            if br.plot_start is not None:
                plot_start = br.plot_start
            if br.plot_finish is not None:
                plot_finish = br.plot_finish
            return filter.filter_time_series_by_date(plot_start, plot_finish, df)

    def _flatten_list(self, list_of_lists):
        if False:
            while True:
                i = 10
        'Flattens list, particularly useful for combining baskets\n\n        Parameters\n        ----------\n        list_of_lists : str (list)\n            List to be flattened\n\n        Returns\n        -------\n\n        '
        result = []
        for i in list_of_lists:
            if isinstance(i, str):
                result.append(i)
            else:
                result.extend(self._flatten_list(i))
        return result

    def strategy_name(self):
        if False:
            while True:
                i = 10
        return self.FINAL_STRATEGY

    def individual_leverage(self):
        if False:
            while True:
                i = 10
        return self._individual_leverage

    def strategy_group_pnl_trades(self):
        if False:
            return 10
        return self._strategy_group_pnl_trades

    def strategy_components_pnl(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy_components_pnl

    def strategy_components_pnl_ret_stats(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy_components_pnl_ret_stats

    def strategy_pnl(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy_pnl

    def strategy_pnl_ret_stats(self):
        if False:
            print('Hello World!')
        return self._strategy_pnl_ret_stats

    def strategy_leverage(self):
        if False:
            print('Hello World!')
        return self._strategy_leverage

    def strategy_benchmark_pnl(self):
        if False:
            while True:
                i = 10
        return self._strategy_benchmark_pnl

    def strategy_benchmark_pnl_ret_stats(self):
        if False:
            while True:
                i = 10
        return self._strategy_benchmark_pnl_ret_stats

    def strategy_group_pnl(self):
        if False:
            i = 10
            return i + 15
        return self._strategy_group_pnl

    def strategy_group_pnl_ret_stats(self):
        if False:
            print('Hello World!')
        return self._strategy_group_pnl_ret_stats

    def strategy_group_benchmark_pnl(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy_group_benchmark_pnl

    def strategy_group_benchmark_pnl_ret_stats(self):
        if False:
            return 10
        return self._strategy_group_benchmark_pnl_ret_stats

    def strategy_group_leverage(self):
        if False:
            print('Hello World!')
        return self._strategy_group_leverage

    def strategy_signal(self):
        if False:
            while True:
                i = 10
        return self._strategy_signal

    def strategy_trade(self):
        if False:
            return 10
        return self._strategy_trade

    def strategy_signal_notional(self):
        if False:
            while True:
                i = 10
        return self._strategy_signal_notional

    def strategy_trade_notional(self):
        if False:
            i = 10
            return i + 15
        return self._strategy_trade_notional

    def strategy_trade_notional_sizes(self):
        if False:
            print('Hello World!')
        return self._strategy_trade_notional_sizes

    def strategy_signal_contracts(self):
        if False:
            while True:
                i = 10
        return self._strategy_signal_contracts

    def strategy_trade_contracts(self):
        if False:
            return 10
        return self._strategy_trade_contracts

    def strategy_total_longs(self):
        if False:
            print('Hello World!')
        return self._strategy_total_longs

    def strategy_total_shorts(self):
        if False:
            i = 10
            return i + 15
        return self._strategy_total_shorts

    def strategy_net_exposure(self):
        if False:
            while True:
                i = 10
        return self._strategy_net_exposure

    def strategy_total_exposure(self):
        if False:
            return 10
        return self._strategy_total_exposure

    def strategy_total_longs_notional(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy_total_longs_notional

    def strategy_total_shorts_notional(self):
        if False:
            print('Hello World!')
        return self._strategy_total_shorts_notional

    def strategy_net_exposure_notional(self):
        if False:
            print('Hello World!')
        return self._strategy_net_exposure_notional

    def strategy_total_exposure_notional(self):
        if False:
            i = 10
            return i + 15
        return self._strategy_total_exposure_notional

    def _reduce_plot(self, data_frame, reduce_plot=True, resample='B'):
        if False:
            print('Hello World!')
        'Reduces the frequency of a time series to every business day so it can be plotted more easily\n\n        Parameters\n        ----------\n        data_frame: pd.DataFrame\n            Strategy time series\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        try:
            if reduce_plot and resample is not None:
                data_frame = data_frame.resample(resample).last()
                data_frame = data_frame.fillna(method='pad')
            return data_frame
        except:
            return data_frame

    def _chart_return_with_df(self, df, style, silent_plot, chart_type='line', ret_with_df=False, split_on_char=None):
        if False:
            print('Hello World!')
        if split_on_char is not None:
            d_split = []
            for d in df.columns:
                try:
                    d_split.append(d.split('.')[0])
                except:
                    d_split.append(d)
            df.columns = d_split
        chart = Chart(df, engine=self.DEFAULT_PLOT_ENGINE, chart_type=chart_type, style=style)
        if not silent_plot:
            chart.plot()
        if ret_with_df:
            return (chart, df)
        return chart

    def plot_individual_leverage(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            return 10
        style = self._create_style('Individual Leverage', 'Individual Leverage', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._individual_leverage, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_group_pnl_trades(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            return 10
        style = self._create_style('(bp)', 'Individual Trade PnL', reduce_plot=reduce_plot)
        try:
            strategy_pnl_trades = self._strategy_group_pnl_trades.fillna(0) * 100 * 100
            df = self._strip_dataframe(self._reduce_plot(strategy_pnl_trades, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_pnl(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            while True:
                i = 10
        style = self._create_style('', 'Strategy PnL', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._strategy_pnl, reduce_plot=reduce_plot, resample=resample), strip)
            if hasattr(self, 'br'):
                if self.br.write_csv_pnl:
                    df.to_csv(self.DUMP_PATH + self.FINAL_STRATEGY + '_pnl.csv')
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except Exception as e:
            print(str(e))

    def plot_strategy_trade_no(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        df_trades = self._strategy_trade_no
        if strip is not None:
            df_trades.index = [k.replace(strip, '') for k in df_trades.index]
        style = self._create_style('', '', reduce_plot=reduce_plot)
        try:
            style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy trade no).png'
            style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy trade no).html'
            df = self._strip_dataframe(self._reduce_plot(df_trades, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='bar', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_signal_proportion(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            return 10
        signal = self._strategy_signal
        long = signal[signal > 0].count()
        short = signal[signal < 0].count()
        flat = signal[signal == 0].count()
        df = pd.DataFrame(index=long.index, columns=['Long', 'Short', 'Flat'])
        df['Long'] = long
        df['Short'] = short
        df['Flat'] = flat
        if strip is not None:
            df.index = [k.replace(strip, '') for k in df.index]
        style = self._create_style('', '')
        try:
            style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy signal proportion).png'
            style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy signal proportion).html'
            df = self._strip_dataframe(self._reduce_plot(df), strip, reduce_plot=reduce_plot, resample=resample)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='bar', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_leverage(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            while True:
                i = 10
        style = self._create_style('Portfolio Leverage', 'Strategy Leverage', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._strategy_leverage, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_components_pnl(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style('Ind Components', 'Strategy PnL Components', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._strategy_components_pnl, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_components_pnl_ir(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            print('Hello World!')
        return self._plot_ret_stats_helper(self._strategy_components_pnl_ret_stats, 'IR', 'Ind Component', 'Ind Component IR', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_components_pnl_returns(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            i = 10
            return i + 15
        return self._plot_ret_stats_helper(self._strategy_components_pnl_ret_stats, 'Returns', 'Ind Component (%)', 'Ind Component Returns', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_components_pnl_vol(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        return self._plot_ret_stats_helper(self._strategy_components_pnl_ret_stats, 'Vol', 'Ind Component (%)', 'Ind Component Vol', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_components_pnl_drawdowns(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        return self._plot_ret_stats_helper(self._strategy_components_pnl_ret_stats, 'Drawdowns', 'Ind Component (%)', 'Ind Component Drawdowns', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_components_pnl_yoy(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            while True:
                i = 10
        return self.plot_yoy_helper(self._strategy_components_pnl_ret_stats, 'Ind Component YoY', 'Ind Component (%)', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        logger = LoggerManager().getLogger(__name__)
        style = self._create_style('', 'Group Benchmark PnL - cumulative')
        strat_list = self._strategy_group_benchmark_pnl.columns
        for line in strat_list:
            logger.info(line)
        df = self._strip_dataframe(self._reduce_plot(self._strategy_group_benchmark_pnl, reduce_plot=reduce_plot, resample=resample), strip)
        return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl_ir(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            print('Hello World!')
        return self._plot_ret_stats_helper(self._strategy_group_benchmark_pnl_ret_stats, 'IR', '', 'Group Benchmark IR', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl_returns(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        return self._plot_ret_stats_helper(self._strategy_group_benchmark_pnl_ret_stats, 'Returns', '(%)', 'Group Benchmark Returns', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl_vol(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            i = 10
            return i + 15
        return self._plot_ret_stats_helper(self._strategy_group_benchmark_pnl_ret_stats, 'Vol', '(%)', 'Group Benchmark Vol', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl_drawdowns(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            return 10
        return self._plot_ret_stats_helper(self._strategy_group_benchmark_pnl_ret_stats, 'Drawdowns', '(%)', 'Group Benchmark Drawdowns', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def _plot_ret_stats_helper(self, ret_stats, metric, title, file_tag, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style(title, file_tag)
        keys = ret_stats.keys()
        ret_metric = []
        for key in keys:
            if metric == 'IR':
                ret_metric.append(ret_stats[key].inforatio()[0])
            elif metric == 'Returns':
                ret_metric.append(ret_stats[key].ann_returns()[0] * 100)
            elif metric == 'Vol':
                ret_metric.append(ret_stats[key].ann_vol()[0] * 100)
            elif metric == 'Drawdowns':
                ret_metric.append(ret_stats[key].drawdowns()[0] * 100)
        if strip is not None:
            keys = [k.replace(strip, '') for k in keys]
        ret_stats = pd.DataFrame(index=keys, data=ret_metric, columns=[metric])
        style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_tag + ') ' + str(style.scale_factor) + '.png'
        style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_tag + ') ' + str(style.scale_factor) + '.html'
        style.display_brand_label = False
        return self._chart_return_with_df(ret_stats, style, silent_plot, chart_type='bar', ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_benchmark_pnl_yoy(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            print('Hello World!')
        return self.plot_yoy_helper(self._strategy_group_benchmark_pnl_ret_stats, '', 'Group Benchmark PnL YoY', strip=strip, silent_plot=silent_plot, ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_yoy_helper(self, ret_stats, title, file_tag, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style(title, title)
        yoy = []
        for key in ret_stats.keys():
            col = ret_stats[key].yoy_rets()
            col.columns = [key]
            yoy.append(col)
        calculations = Calculations()
        ret_stats = calculations.join(yoy, how='outer')
        ret_stats.index = ret_stats.index.year
        ret_stats = self._strip_dataframe(ret_stats, strip)
        style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_tag + ') ' + str(style.scale_factor) + '.png'
        style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_tag + ') ' + str(style.scale_factor) + '.html'
        style.display_brand_label = False
        style.date_formatter = '%Y'
        ret_stats = ret_stats * 100
        return self._chart_return_with_df(ret_stats, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_group_leverage(self, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style('Leverage', 'Group Leverage', reduce_plot=reduce_plot)
        df = self._reduce_plot(self._strategy_group_leverage, reduce_plot=reduce_plot, resample=resample)
        return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)

    def plot_strategy_all_signals(self, signal_show=None, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None, multiplier=100):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style('positions (% portfolio notional)', 'Positions', reduce_plot=reduce_plot)
        df = self._strategy_signal.copy() * multiplier
        if signal_show is not None:
            if signal_show != []:
                not_found = []
                if split_on_char is not None:
                    for d in df.columns:
                        d_split = d.split(split_on_char)[0]
                        if d_split not in signal_show:
                            not_found.append(d)
                    df = df.drop(not_found, axis=1)
        try:
            df = self._strip_dataframe(self._reduce_plot(df, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df)
        except:
            pass

    def plot_strategy_signals(self, date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=100):
        if False:
            for i in range(10):
                print('nop')
        return self._plot_signal(self._strategy_signal, label='positions (% portfolio notional)', caption='Positions', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, ret_with_df=ret_with_df, split_on_char=split_on_char, multiplier=multiplier)

    def plot_strategy_trades(self, date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=100):
        if False:
            i = 10
            return i + 15
        return self._plot_signal(self._strategy_trade, label='trades (% portfolio notional)', caption='Trades', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, ret_with_df=ret_with_df, split_on_char=split_on_char, multiplier=multiplier)

    def plot_strategy_signals_notional(self, date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=1):
        if False:
            for i in range(10):
                print('nop')
        return self._plot_signal(self._strategy_signal_notional, label='positions (scaled by notional)', caption='Positions', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, ret_with_df=ret_with_df, split_on_char=split_on_char, multiplier=multiplier)

    def plot_strategy_trades_notional(self, date=None, strip=None, silent_plot=False, strip_times=False, split_on_char=None, multiplier=1):
        if False:
            while True:
                i = 10
        return self._plot_signal(self._strategy_trade_notional, label='trades (scaled by notional)', caption='Trades', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, split_on_char=split_on_char, multiplier=multiplier)

    def plot_strategy_trades_notional_sizes(self, strip=None, silent_plot=False, ret_with_df=False, split_on_char=None):
        if False:
            print('Hello World!')
        if strip is not None:
            self._strategy_trade_notional_sizes.index = [k.replace(strip, '') for k in self._strategy_trade_notional_sizes.index]
        style = self._create_style('', '', reduce_plot=False)
        try:
            style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy trade notional size).png'
            style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (Strategy trade notional size).html'
            df = self._strip_dataframe(self._strategy_trade_notional_sizes, strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='bar', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_signals_contracts(self, date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=1):
        if False:
            return 10
        return self._plot_signal(self._strategy_signal_contracts, label='positions (contracts)', caption='Positions', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, ret_with_df=ret_with_df, split_on_char=split_on_charm, multiplier=multiplier)

    def plot_strategy_trades_contracts(self, date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=1):
        if False:
            print('Hello World!')
        return self._plot_signal(self._strategy_trade_contracts, label='trades (contracts)', caption='Contracts', date=date, strip=strip, silent_plot=silent_plot, strip_times=strip_times, ret_with_df=ret_with_df, split_on_char=split_on_charm, multiplier=multiplier)

    def plot_strategy_total_exposures(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style('', 'Strategy Total Exposures')
        df = pd.concat([self._strategy_total_longs, self._strategy_total_shorts, self._strategy_total_exposure], axis=1)
        try:
            df = self._strip_dataframe(self._reduce_plot(df, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_net_exposures(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            i = 10
            return i + 15
        style = self._create_style('', 'Strategy Net Exposures', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._strategy_net_exposure, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_total_exposures_notional(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            for i in range(10):
                print('nop')
        style = self._create_style('(mm)', 'Strategy Total Exposures (mm)', reduce_plot=reduce_plot)
        df = pd.concat([self._strategy_total_longs_notional, self._strategy_total_shorts_notional, self._strategy_total_exposure_notional], axis=1)
        try:
            df = self._strip_dataframe(self._reduce_plot(df / 1000000.0, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def plot_strategy_net_exposures_notional(self, strip=None, silent_plot=False, reduce_plot=True, resample='B', ret_with_df=False, split_on_char=None):
        if False:
            i = 10
            return i + 15
        style = self._create_style('(mm)', 'Strategy Net Exposures (mm)', reduce_plot=reduce_plot)
        try:
            df = self._strip_dataframe(self._reduce_plot(self._strategy_net_exposure_notional / 1000000.0, reduce_plot=reduce_plot, resample=resample), strip)
            return self._chart_return_with_df(df, style, silent_plot, chart_type='line', ret_with_df=ret_with_df, split_on_char=split_on_char)
        except:
            pass

    def _grab_signals(self, strategy_signal, date=None, strip=None):
        if False:
            print('Hello World!')
        if date is None:
            last_day = strategy_signal.iloc[-1].transpose().to_frame()
        else:
            if not isinstance(date, list):
                date = [date]
            last_day = []
            for d in date:
                try:
                    last_day.append(strategy_signal.iloc[d].transpose().to_frame())
                except:
                    pass
            last_day = pd.concat(last_day, axis=1)
            last_day = last_day.sort_index(axis=1)
        if strip is not None:
            last_day.index = [x.replace(strip, '') for x in last_day.index]
        return last_day

    def _plot_signal(self, sig, label=' ', caption='', date=None, strip=None, silent_plot=False, strip_times=False, ret_with_df=False, split_on_char=None, multiplier=100):
        if False:
            return 10
        strategy_signal = multiplier * sig
        last_day = self._grab_signals(strategy_signal, date=date, strip=strip)
        style = self._create_style(label, caption)
        style.legend_y_anchor = 'top'
        if strip_times:
            try:
                last_day.index = [x.date() for x in last_day.index]
            except:
                pass
            try:
                last_day.columns = [x.date() for x in last_day.columns]
            except:
                pass
        return self._chart_return_with_df(last_day, style, silent_plot, chart_type='bar', ret_with_df=ret_with_df, split_on_char=split_on_char)

    def _strip_dataframe(self, data_frame, strip):
        if False:
            i = 10
            return i + 15
        if strip is None:
            return data_frame
        if not isinstance(strip, list):
            strip = [strip]
        for s in strip:
            if s == '.':
                data_frame.columns = [x.split(s)[0] if s in x else x for x in data_frame.columns]
            else:
                data_frame.columns = [x.replace(s, '') if s in x else x for x in data_frame.columns]
        return data_frame

    def _create_style(self, title, file_add, reduce_plot=True):
        if False:
            i = 10
            return i + 15
        style = copy.deepcopy(self.CHART_STYLE)
        if self.SHOW_TITLES:
            style.title = self.FINAL_STRATEGY + ' ' + title
        style.display_legend = True
        style.scale_factor = self.SCALE_FACTOR
        style.width = self.WIDTH
        style.height = self.HEIGHT
        style.source = self.CHART_SOURCE
        style.silent_display = not self.SHOW_CHARTS
        style.legend_bgcolor = 'rgba(0,0,0,0)'
        if not reduce_plot:
            style.plotly_webgl = True
        if self.DEFAULT_PLOT_ENGINE not in ['plotly', 'cufflinks'] and self.SAVE_FIGURES:
            style.file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_add + ') ' + str(style.scale_factor) + '.png'
        style.html_file_output = self.DUMP_PATH + self.FINAL_STRATEGY + ' (' + file_add + ') ' + str(style.scale_factor) + '.html'
        try:
            style.silent_display = self.SILENT_DISPLAY
        except:
            pass
        return style

class PortfolioWeightConstruction(object):
    """Creates dynamics weights for signals and also the portfolio

    """

    def __init__(self, br=None):
        if False:
            return 10
        self._br = br
        self._risk_engine = RiskEngine()
        self._calculations = Calculations()

    def optimize_portfolio_weights(self, returns_df, signal_df, signal_pnl_cols, br=None):
        if False:
            return 10
        if br is None:
            br = self._br
        tc = br.spot_tc_bp
        rc = br.spot_rc_bp
        individual_leverage_df = None
        if br.signal_vol_adjust is True:
            leverage_df = self._risk_engine.calculate_leverage_factor(returns_df, br.signal_vol_target, br.signal_vol_max_leverage, br.signal_vol_periods, br.signal_vol_obs_in_year, br.signal_vol_rebalance_freq, br.signal_vol_resample_freq, br.signal_vol_resample_type, period_shift=br.signal_vol_period_shift)
            signal_df = pd.DataFrame(signal_df.values * leverage_df.values, index=signal_df.index, columns=signal_df.columns)
            individual_leverage_df = leverage_df
        signal_pnl = self._calculations.calculate_signal_returns_with_tc_matrix(signal_df, returns_df, tc=tc, rc=rc)
        signal_pnl.columns = signal_pnl_cols
        adjusted_weights_matrix = None
        if br.portfolio_combination is not None:
            if br.portfolio_combination == 'sum' and br.portfolio_combination_weights is None:
                portfolio = pd.DataFrame(data=signal_pnl.sum(axis=1), index=signal_pnl.index, columns=['Portfolio'])
            elif br.portfolio_combination == 'mean' and br.portfolio_combination_weights is None:
                portfolio = pd.DataFrame(data=signal_pnl.mean(axis=1), index=signal_pnl.index, columns=['Portfolio'])
                adjusted_weights_matrix = self.calculate_signal_weights_for_portfolio(br, signal_pnl, method='mean')
            elif 'weighted' in br.portfolio_combination and isinstance(br.portfolio_combination_weights, dict):
                adjusted_weights_matrix = self.calculate_signal_weights_for_portfolio(br, signal_pnl, method=br.portfolio_combination)
                portfolio = pd.DataFrame(data=signal_pnl.values * adjusted_weights_matrix, index=signal_pnl.index)
                is_all_na = pd.isnull(portfolio).all(axis=1)
                portfolio = pd.DataFrame(portfolio.sum(axis=1), columns=['Portfolio'])
                portfolio[is_all_na] = np.nan
        else:
            portfolio = pd.DataFrame(data=signal_pnl.mean(axis=1), index=signal_pnl.index, columns=['Portfolio'])
            adjusted_weights_matrix = self.calculate_signal_weights_for_portfolio(br, signal_pnl, method='mean')
        portfolio_leverage_df = pd.DataFrame(data=np.ones(len(signal_pnl.index)), index=signal_pnl.index, columns=['Portfolio'])
        if br.portfolio_vol_adjust is True:
            portfolio_leverage_df = self._risk_engine.calculate_leverage_factor(portfolio, br.portfolio_vol_target, br.portfolio_vol_max_leverage, br.portfolio_vol_periods, br.portfolio_vol_obs_in_year, br.portfolio_vol_rebalance_freq, br.portfolio_vol_resample_freq, br.portfolio_vol_resample_type, period_shift=br.portfolio_vol_period_shift)
        length_cols = len(signal_df.columns)
        leverage_matrix = np.transpose(np.repeat(portfolio_leverage_df.values.flatten()[np.newaxis, :], length_cols, 0))
        portfolio_signal = pd.DataFrame(data=np.multiply(leverage_matrix, signal_df.values), index=signal_df.index, columns=signal_df.columns)
        portfolio_signal_before_weighting = portfolio_signal.copy()
        if br.portfolio_combination is not None:
            if 'sum' in br.portfolio_combination:
                pass
            elif br.portfolio_combination == 'mean' or (br.portfolio_combination == 'weighted' and isinstance(br.portfolio_combination_weights, dict)):
                portfolio_signal = pd.DataFrame(data=portfolio_signal.values * adjusted_weights_matrix, index=portfolio_signal.index, columns=portfolio_signal.columns)
        else:
            portfolio_signal = pd.DataFrame(data=portfolio_signal.values * adjusted_weights_matrix, index=portfolio_signal.index, columns=portfolio_signal.columns)
        return (portfolio_signal_before_weighting, portfolio_signal, portfolio_leverage_df, portfolio, individual_leverage_df, signal_pnl)

    def calculate_signal_weights_for_portfolio(self, br, signal_pnl, method='mean'):
        if False:
            return 10
        "Calculates the weights of each signal for the portfolio\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for the backtest specifying start date, finish data, transaction costs etc.\n\n        signal_pnl : pd.DataFrame\n            Contains the daily P&L for the portfolio\n\n        method : String\n            'mean' - assumes equal weighting for each signal\n            'weighted' - can use predefined user weights (eg. if we assign weighting of 1, 1, 0.5, for three signals\n            the third signal will have a weighting of half versus the others)\n\n        weights : dict\n            Portfolio weights\n\n        Returns\n        -------\n        pd.DataFrame\n            Contains the portfolio weights\n        "
        if method == 'mean':
            weights_vector = np.ones(len(signal_pnl.columns))
        elif method == 'weighted' or 'weighted-sum':
            weights_vector = np.array([float(br.portfolio_combination_weights[col]) for col in signal_pnl.columns])
        weights_matrix = np.repeat(weights_vector[np.newaxis, :], len(signal_pnl.index), 0)
        ind = np.isnan(signal_pnl.values)
        weights_matrix[ind] = 0
        if method != 'weighted-sum':
            total_weights = np.sum(weights_matrix, axis=1)
            total_weights = np.transpose(np.repeat(total_weights[np.newaxis, :], len(signal_pnl.columns), 0))
            total_weights[total_weights == 0.0] = 1.0
            adjusted_weights_matrix = weights_matrix / total_weights
            adjusted_weights_matrix[ind] = np.nan
            return adjusted_weights_matrix
        return weights_matrix

class RiskEngine(object):
    """Adjusts signal weighting according to risk constraints (volatility targeting and position limit constraints)

    """

    def calculate_vol_adjusted_index_from_prices(self, prices_df, br):
        if False:
            print('Hello World!')
        'Adjusts an index of prices for a vol target\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for the backtest specifying start date, finish data, transaction costs etc.\n        asset_a_df : pd.DataFrame\n            Asset prices to be traded\n\n        Returns\n        -------\n        pd.Dataframe containing vol adjusted index\n        '
        calculations = Calculations()
        (returns_df, leverage_df) = self.calculate_vol_adjusted_returns(prices_df, br, returns=False)
        if br.cum_index == 'mult':
            return calculations.create_mult_index(returns_df)
        elif br.cum_index == 'add':
            return calculations.create_add_index(returns_df)

    def calculate_vol_adjusted_returns(self, returns_df, br, returns=True):
        if False:
            while True:
                i = 10
        'Adjusts returns for a vol target\n\n        Parameters\n        ----------\n        br : BacktestRequest\n            Parameters for the backtest specifying start date, finish data, transaction costs etc.\n        returns_a_df : pd.DataFrame\n            Asset returns to be traded\n\n        Returns\n        -------\n        pd.DataFrame\n        '
        calculations = Calculations()
        if not returns:
            returns_df = calculations.calculate_returns(returns_df)
        leverage_df = self.calculate_leverage_factor(returns_df, br.portfolio_vol_target, br.portfolio_vol_max_leverage, br.portfolio_vol_periods, br.portfolio_vol_obs_in_year, br.portfolio_vol_rebalance_freq, br.portfolio_vol_resample_freq, br.portfolio_vol_resample_type, period_shift=br.portfolio_vol_period_shift)
        vol_returns_df = calculations.calculate_signal_returns_with_tc_matrix(leverage_df, returns_df, tc=br.spot_tc_bp)
        vol_returns_df.columns = returns_df.columns
        return (vol_returns_df, leverage_df)

    def calculate_leverage_factor(self, returns_df, vol_target, vol_max_leverage, vol_periods=60, vol_obs_in_year=252, vol_rebalance_freq='BM', resample_freq=None, resample_type='mean', returns=True, period_shift=0):
        if False:
            while True:
                i = 10
        'Calculates the time series of leverage for a specified vol target\n\n        Parameters\n        ----------\n        returns_df : DataFrame\n            Asset returns\n        vol_target : float\n            vol target for assets\n        vol_max_leverage : float\n            maximum leverage allowed\n        vol_periods : int\n            number of periods to calculate volatility\n        vol_obs_in_year : int\n            number of observations in the year\n        vol_rebalance_freq : str\n            how often to rebalance\n        resample_type : str\n            do we need to resample the underlying data first? (eg. have we got intraday data?)\n        returns : boolean\n            is this returns time series or prices?\n        period_shift : int\n            should we delay the signal by a number of periods?\n\n        Returns\n        -------\n        pd.Dataframe\n        '
        calculations = Calculations()
        filter = Filter()
        if resample_freq is not None:
            return
        if not returns:
            returns_df = calculations.calculate_returns(returns_df)
        roll_vol_df = calculations.rolling_volatility(returns_df, periods=vol_periods, obs_in_year=vol_obs_in_year).shift(period_shift)
        lev_df = vol_target / roll_vol_df
        if vol_max_leverage is not None:
            lev_df[lev_df > vol_max_leverage] = vol_max_leverage
        if resample_type is not None:
            lev_df = filter.resample_time_series_frequency(lev_df, vol_rebalance_freq, resample_type)
            (returns_df, lev_df) = calculations.join_left_fill_right(returns_df, lev_df)
        lev_df[0:vol_periods] = np.nan
        return lev_df

    def calculate_position_clip_adjustment(self, portfolio_net_exposure, portfolio_total_exposure, br):
        if False:
            return 10
        'Calculate the leverage adjustment that needs to be made in the portfolio such that either the net exposure or\n        the absolute exposure fits within predefined limits\n\n        Parameters\n        ----------\n        portfolio_net_exposure : DataFrame\n            Net exposure of the whole portfolio\n        portfolio_total_exposure : DataFrame\n            Absolute exposure of the whole portfolio\n        br : BacktestRequest\n            Includes parameters for setting position limits\n\n        Returns\n        -------\n        DataFrame\n        '
        position_clip_adjustment = None
        if br.max_net_exposure is not None:
            portfolio_net_exposure = portfolio_net_exposure.shift(br.position_clip_period_shift)
            position_clip_adjustment = pd.DataFrame(data=np.ones(len(portfolio_net_exposure.index)), index=portfolio_net_exposure.index, columns=['Portfolio'])
            portfolio_abs_exposure = portfolio_net_exposure.abs()
            position_clip_adjustment[(portfolio_abs_exposure > br.max_net_exposure).values] = br.max_net_exposure / portfolio_abs_exposure
        if br.max_abs_exposure is not None:
            portfolio_abs_exposure = portfolio_abs_exposure.shift(br.position_clip_period_shift)
            position_clip_adjustment = pd.DataFrame(data=np.ones(len(portfolio_abs_exposure.index)), index=portfolio_abs_exposure.index, columns=['Portfolio'])
            position_clip_adjustment[(portfolio_total_exposure > br.max_abs_exposure).values] = br.max_abs_exposure / portfolio_total_exposure
        if br.position_clip_rebalance_freq is not None:
            calculations = Calculations()
            filter = Filter()
            position_clip_adjustment = filter.resample_time_series_frequency(position_clip_adjustment, br.position_clip_rebalance_freq, br.position_clip_resample_type)
            (a, position_clip_adjustment) = calculations.join_left_fill_right(portfolio_net_exposure, position_clip_adjustment)
        return position_clip_adjustment