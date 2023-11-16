__author__ = 'saeedamen'
from findatapy.market import MarketDataRequest
from finmarketpy.economics import TechParams
from findatapy.util.loggermanager import LoggerManager
from pandas import DataFrame

class BacktestRequest(MarketDataRequest):
    """Contains parameters necessary to define a backtest, including start date, finish date, transaction cost, etc

    Used by TradingModel and Backtest to construct backtested returns for trading strategies

    """

    def __init__(self):
        if False:
            return 10
        super(MarketDataRequest, self).__init__()
        self.__signal_name = None
        self.__plot_start = None
        self.__plot_finish = None
        self.__calc_stats = True
        self.__write_csv = False
        self.__write_csv_pnl = False
        self.__plot_interim = False
        self.__include_benchmark = False
        self.__trading_field = 'close'
        self.__tech_params = TechParams()
        self.__portfolio_weight_construction = None
        self.__portfolio_vol_adjust = False
        self.__portfolio_vol_period_shift = 0
        self.__portfolio_vol_rebalance_freq = None
        self.__portfolio_vol_resample_freq = None
        self.__portfolio_vol_resample_type = 'mean'
        self.__portfolio_vol_target = 0.1
        self.__portfolio_vol_max_leverage = None
        self.__portfolio_vol_periods = 20
        self.__portfolio_vol_obs_in_year = 252
        self.__signal_vol_adjust = False
        self.__signal_vol_period_shift = 0
        self.__signal_vol_rebalance_freq = None
        self.__signal_vol_resample_freq = None
        self.__signal_vol_resample_type = 'mean'
        self.__signal_vol_target = 0.1
        self.__signal_vol_max_leverage = None
        self.__signal_vol_periods = 20
        self.__signal_vol_obs_in_year = 252
        self.__portfolio_notional_size = None
        self.__portfolio_combination = None
        self.__portfolio_combination_weights = None
        self.__max_net_exposure = None
        self.__max_abs_exposure = None
        self.__position_clip_rebalance_freq = None
        self.__position_clip_resample_freq = None
        self.__position_clip_resample_type = 'mean'
        self.__position_clip_period_shift = 0
        self.__take_profit = None
        self.__stop_loss = None
        self.__signal_delay = 0
        self.__ann_factor = 252
        self.__resample_ann_factor = None
        self.__spot_rc_bp = None
        self.__cum_index = 'mult'

    @property
    def plot_start(self):
        if False:
            return 10
        return self.__plot_start

    @plot_start.setter
    def plot_start(self, plot_start):
        if False:
            return 10
        self.__plot_start = plot_start

    @property
    def plot_finish(self):
        if False:
            return 10
        return self.__plot_finish

    @plot_finish.setter
    def plot_finish(self, plot_finish):
        if False:
            return 10
        self.__plot_finish = plot_finish

    @property
    def calc_stats(self):
        if False:
            i = 10
            return i + 15
        return self.__calc_stats

    @calc_stats.setter
    def calc_stats(self, calc_stats):
        if False:
            while True:
                i = 10
        self.__calc_stats = calc_stats

    @property
    def write_csv(self):
        if False:
            print('Hello World!')
        return self.__write_csv

    @write_csv.setter
    def write_csv(self, write_csv):
        if False:
            i = 10
            return i + 15
        self.__write_csv = write_csv

    @property
    def write_csv_pnl(self):
        if False:
            while True:
                i = 10
        return self.__write_csv_pnl

    @write_csv_pnl.setter
    def write_csv_pnl(self, write_csv_pnl):
        if False:
            for i in range(10):
                print('nop')
        self.__write_csv_pnl = write_csv_pnl

    @property
    def plot_interim(self):
        if False:
            return 10
        return self.__plot_interim

    @plot_interim.setter
    def plot_interim(self, plot_interim):
        if False:
            while True:
                i = 10
        self.__plot_interim = plot_interim

    @property
    def include_benchmark(self):
        if False:
            i = 10
            return i + 15
        return self.__include_benchmark

    @include_benchmark.setter
    def include_benchmark(self, include_benchmark):
        if False:
            while True:
                i = 10
        self.__include_benchmark = include_benchmark

    @property
    def trading_field(self):
        if False:
            return 10
        return self.__trading_field

    @trading_field.setter
    def trading_field(self, trading_field):
        if False:
            for i in range(10):
                print('nop')
        self.__trading_field = trading_field

    @property
    def portfolio_weight_construction(self):
        if False:
            print('Hello World!')
        return self.__portfolio_weight_construction

    @portfolio_weight_construction.setter
    def portfolio_weight_construction(self, portfolio_weight_construction):
        if False:
            print('Hello World!')
        self.__portfolio_weight_construction = portfolio_weight_construction

    @property
    def portfolio_vol_adjust(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__portfolio_vol_adjust

    @portfolio_vol_adjust.setter
    def portfolio_vol_adjust(self, portfolio_vol_adjust):
        if False:
            while True:
                i = 10
        self.__portfolio_vol_adjust = portfolio_vol_adjust

    @property
    def portfolio_vol_rebalance_freq(self):
        if False:
            while True:
                i = 10
        return self.__portfolio_vol_rebalance_freq

    @portfolio_vol_rebalance_freq.setter
    def portfolio_vol_rebalance_freq(self, portfolio_vol_rebalance_freq):
        if False:
            for i in range(10):
                print('nop')
        self.__portfolio_vol_rebalance_freq = portfolio_vol_rebalance_freq

    @property
    def portfolio_vol_resample_type(self):
        if False:
            i = 10
            return i + 15
        return self.__portfolio_vol_resample_type

    @portfolio_vol_resample_type.setter
    def portfolio_vol_resample_type(self, portfolio_vol_resample_type):
        if False:
            i = 10
            return i + 15
        self.__portfolio_vol_resample_type = portfolio_vol_resample_type

    @property
    def portfolio_vol_resample_freq(self):
        if False:
            print('Hello World!')
        return self.__portfolio_vol_resample_freq

    @portfolio_vol_resample_freq.setter
    def portfolio_vol_resample_freq(self, portfolio_vol_resample_freq):
        if False:
            for i in range(10):
                print('nop')
        self.__portfolio_vol_resample_freq = portfolio_vol_resample_freq

    @property
    def portfolio_vol_period_shift(self):
        if False:
            print('Hello World!')
        return self.__portfolio_vol_period_shift

    @portfolio_vol_period_shift.setter
    def portfolio_vol_period_shift(self, portfolio_vol_period_shift):
        if False:
            for i in range(10):
                print('nop')
        self.__portfolio_vol_period_shift = portfolio_vol_period_shift

    @property
    def portfolio_vol_target(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__portfolio_vol_target

    @portfolio_vol_target.setter
    def portfolio_vol_target(self, portfolio_vol_target):
        if False:
            while True:
                i = 10
        self.__portfolio_vol_target = portfolio_vol_target

    @property
    def portfolio_vol_max_leverage(self):
        if False:
            return 10
        return self.__portfolio_vol_max_leverage

    @portfolio_vol_max_leverage.setter
    def portfolio_vol_max_leverage(self, portfolio_vol_max_leverage):
        if False:
            return 10
        self.__portfolio_vol_max_leverage = portfolio_vol_max_leverage

    @property
    def portfolio_vol_periods(self):
        if False:
            print('Hello World!')
        return self.__portfolio_vol_periods

    @portfolio_vol_periods.setter
    def portfolio_vol_periods(self, portfolio_vol_periods):
        if False:
            i = 10
            return i + 15
        self.__portfolio_vol_periods = portfolio_vol_periods

    @property
    def portfolio_vol_obs_in_year(self):
        if False:
            print('Hello World!')
        return self.__portfolio_vol_obs_in_year

    @portfolio_vol_obs_in_year.setter
    def portfolio_vol_obs_in_year(self, portfolio_vol_obs_in_year):
        if False:
            print('Hello World!')
        self.__portfolio_vol_obs_in_year = portfolio_vol_obs_in_year

    @property
    def signal_vol_adjust(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__signal_vol_adjust

    @signal_vol_adjust.setter
    def signal_vol_adjust(self, signal_vol_adjust):
        if False:
            return 10
        self.__signal_vol_adjust = signal_vol_adjust

    @property
    def signal_vol_rebalance_freq(self):
        if False:
            print('Hello World!')
        return self.__signal_vol_rebalance_freq

    @signal_vol_rebalance_freq.setter
    def signal_vol_rebalance_freq(self, signal_vol_rebalance_freq):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_vol_rebalance_freq = signal_vol_rebalance_freq

    @property
    def signal_vol_resample_type(self):
        if False:
            i = 10
            return i + 15
        return self.__signal_vol_resample_type

    @signal_vol_resample_type.setter
    def signal_vol_resample_type(self, signal_vol_resample_type):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_vol_resample_type = signal_vol_resample_type

    @property
    def signal_vol_resample_freq(self):
        if False:
            return 10
        return self.__signal_vol_resample_freq

    @signal_vol_resample_freq.setter
    def signal_vol_resample_freq(self, signal_vol_resample_freq):
        if False:
            while True:
                i = 10
        self.__signal_vol_resample_freq = signal_vol_resample_freq

    @property
    def signal_vol_period_shift(self):
        if False:
            i = 10
            return i + 15
        return self.__signal_vol_period_shift

    @signal_vol_period_shift.setter
    def signal_vol_period_shift(self, signal_vol_period_shift):
        if False:
            while True:
                i = 10
        self.__signal_vol_period_shift = signal_vol_period_shift

    @property
    def signal_vol_target(self):
        if False:
            while True:
                i = 10
        return self.__signal_vol_target

    @signal_vol_target.setter
    def signal_vol_target(self, signal_vol_target):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_vol_target = signal_vol_target

    @property
    def signal_vol_max_leverage(self):
        if False:
            i = 10
            return i + 15
        return self.__signal_vol_max_leverage

    @signal_vol_max_leverage.setter
    def signal_vol_max_leverage(self, signal_vol_max_leverage):
        if False:
            while True:
                i = 10
        self.__signal_vol_max_leverage = signal_vol_max_leverage

    @property
    def signal_vol_periods(self):
        if False:
            return 10
        return self.__signal_vol_periods

    @signal_vol_periods.setter
    def signal_vol_periods(self, signal_vol_periods):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_vol_periods = signal_vol_periods

    @property
    def signal_vol_obs_in_year(self):
        if False:
            i = 10
            return i + 15
        return self.__signal_vol_obs_in_year

    @signal_vol_obs_in_year.setter
    def signal_vol_obs_in_year(self, signal_vol_obs_in_year):
        if False:
            i = 10
            return i + 15
        self.__signal_vol_obs_in_year = signal_vol_obs_in_year

    @property
    def portfolio_notional_size(self):
        if False:
            i = 10
            return i + 15
        return self.__portfolio_notional_size

    @portfolio_notional_size.setter
    def portfolio_notional_size(self, portfolio_notional_size):
        if False:
            print('Hello World!')
        self.__portfolio_notional_size = float(portfolio_notional_size)

    @property
    def portfolio_combination(self):
        if False:
            return 10
        return self.__portfolio_combination

    @portfolio_combination.setter
    def portfolio_combination(self, portfolio_combination):
        if False:
            while True:
                i = 10
        self.__portfolio_combination = portfolio_combination

    @property
    def portfolio_combination_weights(self):
        if False:
            while True:
                i = 10
        return self.__portfolio_combination_weights

    @portfolio_combination_weights.setter
    def portfolio_combination_weights(self, portfolio_combination_weights):
        if False:
            return 10
        self.__portfolio_combination_weights = portfolio_combination_weights

    @property
    def max_net_exposure(self):
        if False:
            return 10
        return self.__max_net_exposure

    @max_net_exposure.setter
    def max_net_exposure(self, max_net_exposure):
        if False:
            while True:
                i = 10
        self.__max_net_exposure = max_net_exposure

    @property
    def max_abs_exposure(self):
        if False:
            while True:
                i = 10
        return self.__max_abs_exposure

    @max_abs_exposure.setter
    def max_abs_exposure(self, max_abs_exposure):
        if False:
            i = 10
            return i + 15
        self.__max_abs_exposure = max_abs_exposure

    @property
    def position_clip_rebalance_freq(self):
        if False:
            while True:
                i = 10
        return self.__position_clip_rebalance_freq

    @position_clip_rebalance_freq.setter
    def position_clip_rebalance_freq(self, position_clip_rebalance_freq):
        if False:
            return 10
        self.__position_clip_rebalance_freq = position_clip_rebalance_freq

    @property
    def position_clip_resample_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__position_clip_resample_type

    @position_clip_resample_type.setter
    def position_clip_resample_type(self, position_clip_resample_type):
        if False:
            for i in range(10):
                print('nop')
        self.__position_clip_resample_type = position_clip_resample_type

    @property
    def position_clip_resample_freq(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__position_clip_resample_freq

    @position_clip_resample_freq.setter
    def position_clip_resample_freq(self, position_clip_resample_freq):
        if False:
            while True:
                i = 10
        self.__position_clip_resample_freq = position_clip_resample_freq

    @property
    def position_clip_period_shift(self):
        if False:
            return 10
        return self.__position_clip_period_shift

    @position_clip_period_shift.setter
    def position_clip_period_shift(self, position_clip_period_shift):
        if False:
            while True:
                i = 10
        self.__position_clip_period_shift = position_clip_period_shift

    @property
    def stop_loss(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__stop_loss

    @stop_loss.setter
    def stop_loss(self, stop_loss):
        if False:
            print('Hello World!')
        self.__stop_loss = stop_loss

    @property
    def take_profit(self):
        if False:
            i = 10
            return i + 15
        return self.__take_profit

    @take_profit.setter
    def take_profit(self, take_profit):
        if False:
            while True:
                i = 10
        self.__take_profit = take_profit

    @property
    def tech_params(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__tech_params

    @tech_params.setter
    def tech_params(self, tech_params):
        if False:
            while True:
                i = 10
        self.__tech_params = tech_params

    @property
    def spot_tc_bp(self):
        if False:
            i = 10
            return i + 15
        return self.__spot_tc_bp

    @spot_tc_bp.setter
    def spot_tc_bp(self, spot_tc_bp):
        if False:
            i = 10
            return i + 15
        if isinstance(spot_tc_bp, dict):
            spot_tc_bp = spot_tc_bp.copy()
            for k in spot_tc_bp.keys():
                spot_tc_bp[k] = float(spot_tc_bp[k]) / (2.0 * 100.0 * 100.0)
            self.__spot_tc_bp = spot_tc_bp
        elif isinstance(spot_tc_bp, DataFrame):
            self.__spot_tc_bp = spot_tc_bp
        else:
            self.__spot_tc_bp = float(spot_tc_bp) / (2.0 * 100.0 * 100.0)

    @property
    def spot_rc_bp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__spot_rc_bp

    @spot_rc_bp.setter
    def spot_rc_bp(self, spot_rc_bp):
        if False:
            print('Hello World!')
        if isinstance(spot_rc_bp, dict):
            spot_rc_bp = spot_rc_bp.copy()
            for k in spot_rc_bp.keys():
                spot_rc_bp[k] = float(spot_rc_bp[k]) / (100.0 * 100.0)
            self.__spot_rc_bp = spot_rc_bp
        elif isinstance(spot_rc_bp, DataFrame):
            self.__spot_rc_bp = spot_rc_bp
        else:
            self.__spot_rc_bp = float(spot_rc_bp) / (100.0 * 100.0)

    @property
    def signal_name(self):
        if False:
            while True:
                i = 10
        return self.__signal_name

    @signal_name.setter
    def signal_name(self, signal_name):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_name = signal_name

    @property
    def asset(self):
        if False:
            i = 10
            return i + 15
        return self.__asset

    @asset.setter
    def asset(self, asset):
        if False:
            i = 10
            return i + 15
        valid_asset = ['fx', 'multi-asset']
        if not asset in valid_asset:
            LoggerManager().getLogger(__name__).warning(asset & ' is not a defined asset.')
        self.__asset = asset

    @property
    def instrument(self):
        if False:
            i = 10
            return i + 15
        return self.__instrument

    @instrument.setter
    def instrument(self, instrument):
        if False:
            print('Hello World!')
        valid_instrument = ['spot', 'futures', 'options']
        if not instrument in valid_instrument:
            LoggerManager().getLogger(__name__).warning(instrument & ' is not a defined trading instrument.')
        self.__instrument = instrument

    @property
    def signal_delay(self):
        if False:
            print('Hello World!')
        return self.__signal_delay

    @signal_delay.setter
    def signal_delay(self, signal_delay):
        if False:
            for i in range(10):
                print('nop')
        self.__signal_delay = signal_delay

    @property
    def ann_factor(self):
        if False:
            return 10
        return self.__ann_factor

    @ann_factor.setter
    def ann_factor(self, ann_factor):
        if False:
            i = 10
            return i + 15
        self.__ann_factor = ann_factor

    @property
    def resample_ann_factor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__resample_ann_factor

    @resample_ann_factor.setter
    def resample_ann_factor(self, resample_ann_factor):
        if False:
            while True:
                i = 10
        self.__resample_ann_factor = resample_ann_factor

    @property
    def cum_index(self):
        if False:
            print('Hello World!')
        return self.__cum_index

    @cum_index.setter
    def cum_index(self, cum_index):
        if False:
            i = 10
            return i + 15
        self.__cum_index = cum_index