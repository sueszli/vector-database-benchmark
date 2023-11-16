""" Edge positioning package """
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, NamedTuple
import numpy as np
import utils_find_1st as utf1st
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT, UNLIMITED_STAKE_AMOUNT, Config
from freqtrade.data.history import get_timerange, load_data, refresh_data
from freqtrade.enums import CandleType, ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.strategy.interface import IStrategy
from freqtrade.util import dt_now
logger = logging.getLogger(__name__)

class PairInfo(NamedTuple):
    stoploss: float
    winrate: float
    risk_reward_ratio: float
    required_risk_reward: float
    expectancy: float
    nb_trades: int
    avg_trade_duration: float

class Edge:
    """
    Calculates Win Rate, Risk Reward Ratio, Expectancy
    against historical data for a give set of markets and a strategy
    it then adjusts stoploss and position size accordingly
    and force it into the strategy
    Author: https://github.com/mishaker
    """
    _cached_pairs: Dict[str, Any] = {}

    def __init__(self, config: Config, exchange, strategy) -> None:
        if False:
            print('Hello World!')
        self.config = config
        self.exchange = exchange
        self.strategy: IStrategy = strategy
        self.edge_config = self.config.get('edge', {})
        self._cached_pairs: Dict[str, Any] = {}
        self._final_pairs: list = []
        if self.config['max_open_trades'] != float('inf'):
            logger.critical('max_open_trades should be -1 in config !')
        if self.config['stake_amount'] != UNLIMITED_STAKE_AMOUNT:
            raise OperationalException('Edge works only with unlimited stake amount')
        self._capital_ratio: float = self.config['tradable_balance_ratio']
        self._allowed_risk: float = self.edge_config.get('allowed_risk')
        self._since_number_of_days: int = self.edge_config.get('calculate_since_number_of_days', 14)
        self._last_updated: int = 0
        self._refresh_pairs = True
        self._stoploss_range_min = float(self.edge_config.get('stoploss_range_min', -0.01))
        self._stoploss_range_max = float(self.edge_config.get('stoploss_range_max', -0.05))
        self._stoploss_range_step = float(self.edge_config.get('stoploss_range_step', -0.001))
        self._stoploss_range = np.arange(self._stoploss_range_min, self._stoploss_range_max, self._stoploss_range_step)
        self._timerange: TimeRange = TimeRange.parse_timerange(f"{(dt_now() - timedelta(days=self._since_number_of_days)).strftime('%Y%m%d')}-")
        if config.get('fee'):
            self.fee = config['fee']
        else:
            try:
                self.fee = self.exchange.get_fee(symbol=expand_pairlist(self.config['exchange']['pair_whitelist'], list(self.exchange.markets))[0])
            except IndexError:
                self.fee = None

    def calculate(self, pairs: List[str]) -> bool:
        if False:
            return 10
        if self.fee is None and pairs:
            self.fee = self.exchange.get_fee(pairs[0])
        heartbeat = self.edge_config.get('process_throttle_secs')
        if self._last_updated > 0 and self._last_updated + heartbeat > int(dt_now().timestamp()):
            return False
        data: Dict[str, Any] = {}
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using local backtesting data (using whitelist in given config) ...')
        if self._refresh_pairs:
            timerange_startup = deepcopy(self._timerange)
            timerange_startup.subtract_start(timeframe_to_seconds(self.strategy.timeframe) * self.strategy.startup_candle_count)
            refresh_data(datadir=self.config['datadir'], pairs=pairs, exchange=self.exchange, timeframe=self.strategy.timeframe, timerange=timerange_startup, data_format=self.config['dataformat_ohlcv'], candle_type=self.config.get('candle_type_def', CandleType.SPOT))
            res = defaultdict(list)
            for (pair, timeframe, _) in self.strategy.gather_informative_pairs():
                res[timeframe].append(pair)
            for (timeframe, inf_pairs) in res.items():
                timerange_startup = deepcopy(self._timerange)
                timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * self.strategy.startup_candle_count)
                refresh_data(datadir=self.config['datadir'], pairs=inf_pairs, exchange=self.exchange, timeframe=timeframe, timerange=timerange_startup, data_format=self.config['dataformat_ohlcv'], candle_type=self.config.get('candle_type_def', CandleType.SPOT))
        data = load_data(datadir=self.config['datadir'], pairs=pairs, timeframe=self.strategy.timeframe, timerange=self._timerange, startup_candles=self.strategy.startup_candle_count, data_format=self.config['dataformat_ohlcv'], candle_type=self.config.get('candle_type_def', CandleType.SPOT))
        if not data:
            self._cached_pairs = {}
            logger.critical('No data found. Edge is stopped ...')
            return False
        prior_rm = self.config['runmode']
        self.config['runmode'] = RunMode.EDGE
        preprocessed = self.strategy.advise_all_indicators(data)
        self.config['runmode'] = prior_rm
        (min_date, max_date) = get_timerange(preprocessed)
        logger.info(f'Measuring data from {min_date.strftime(DATETIME_PRINT_FORMAT)} up to {max_date.strftime(DATETIME_PRINT_FORMAT)} ({(max_date - min_date).days} days)..')
        headers = ['date', 'open', 'high', 'low', 'close', 'enter_long', 'exit_long']
        trades: list = []
        for (pair, pair_data) in preprocessed.items():
            pair_data = pair_data.sort_values(by=['date'])
            pair_data = pair_data.reset_index(drop=True)
            df_analyzed = self.strategy.ft_advise_signals(pair_data, {'pair': pair})[headers].copy()
            trades += self._find_trades_for_stoploss_range(df_analyzed, pair, self._stoploss_range)
        if len(trades) == 0:
            logger.info('No trades found.')
            return False
        trades_df = self._fill_calculable_fields(DataFrame(trades))
        self._cached_pairs = self._process_expectancy(trades_df)
        self._last_updated = int(dt_now().timestamp())
        return True

    def stake_amount(self, pair: str, free_capital: float, total_capital: float, capital_in_trade: float) -> float:
        if False:
            return 10
        stoploss = self.get_stoploss(pair)
        available_capital = (total_capital + capital_in_trade) * self._capital_ratio
        allowed_capital_at_risk = available_capital * self._allowed_risk
        max_position_size = abs(allowed_capital_at_risk / stoploss)
        position_size = min(min(max_position_size, free_capital), available_capital)
        if pair in self._cached_pairs:
            logger.info('winrate: %s, expectancy: %s, position size: %s, pair: %s, capital in trade: %s, free capital: %s, total capital: %s, stoploss: %s, available capital: %s.', self._cached_pairs[pair].winrate, self._cached_pairs[pair].expectancy, position_size, pair, capital_in_trade, free_capital, total_capital, stoploss, available_capital)
        return round(position_size, 15)

    def get_stoploss(self, pair: str) -> float:
        if False:
            while True:
                i = 10
        if pair in self._cached_pairs:
            return self._cached_pairs[pair].stoploss
        else:
            logger.warning(f'Tried to access stoploss of non-existing pair {pair}, strategy stoploss is returned instead.')
            return self.strategy.stoploss

    def adjust(self, pairs: List[str]) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filters out and sorts "pairs" according to Edge calculated pairs\n        '
        final = []
        for (pair, info) in self._cached_pairs.items():
            if info.expectancy > float(self.edge_config.get('minimum_expectancy', 0.2)) and info.winrate > float(self.edge_config.get('minimum_winrate', 0.6)) and (pair in pairs):
                final.append(pair)
        if self._final_pairs != final:
            self._final_pairs = final
            if self._final_pairs:
                logger.info('Minimum expectancy and minimum winrate are met only for %s, so other pairs are filtered out.', self._final_pairs)
            else:
                logger.info('Edge removed all pairs as no pair with minimum expectancy and minimum winrate was found !')
        return self._final_pairs

    def accepted_pairs(self) -> List[Dict[str, Any]]:
        if False:
            while True:
                i = 10
        '\n        return a list of accepted pairs along with their winrate, expectancy and stoploss\n        '
        final = []
        for (pair, info) in self._cached_pairs.items():
            if info.expectancy > float(self.edge_config.get('minimum_expectancy', 0.2)) and info.winrate > float(self.edge_config.get('minimum_winrate', 0.6)):
                final.append({'Pair': pair, 'Winrate': info.winrate, 'Expectancy': info.expectancy, 'Stoploss': info.stoploss})
        return final

    def _fill_calculable_fields(self, result: DataFrame) -> DataFrame:
        if False:
            return 10
        '\n        The result frame contains a number of columns that are calculable\n        from other columns. These are left blank till all rows are added,\n        to be populated in single vector calls.\n\n        Columns to be populated are:\n        - Profit\n        - trade duration\n        - profit abs\n        :param result Dataframe\n        :return: result Dataframe\n        '
        stake = 0.015
        result['trade_duration'] = result['close_date'] - result['open_date']
        result['trade_duration'] = result['trade_duration'].map(lambda x: int(x.total_seconds() / 60))
        result['buy_vol'] = stake / result['open_rate']
        result['buy_fee'] = stake * self.fee
        result['buy_spend'] = stake + result['buy_fee']
        result['sell_sum'] = result['buy_vol'] * result['close_rate']
        result['sell_fee'] = result['sell_sum'] * self.fee
        result['sell_take'] = result['sell_sum'] - result['sell_fee']
        result['profit_ratio'] = (result['sell_take'] - result['buy_spend']) / result['buy_spend']
        result['profit_abs'] = result['sell_take'] - result['buy_spend']
        return result

    def _process_expectancy(self, results: DataFrame) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        This calculates WinRate, Required Risk Reward, Risk Reward and Expectancy of all pairs\n        The calculation will be done per pair and per strategy.\n        '
        min_trades_number = self.edge_config.get('min_trade_number', 10)
        results = results.groupby(['pair', 'stoploss']).filter(lambda x: len(x) > min_trades_number)
        if self.edge_config.get('remove_pumps', False):
            results = results[results['profit_abs'] < 2 * results['profit_abs'].std() + results['profit_abs'].mean()]
        max_trade_duration = self.edge_config.get('max_trade_duration_minute', 1440)
        results = results[results.trade_duration < max_trade_duration]
        if results.empty:
            return {}
        groupby_aggregator = {'profit_abs': [('nb_trades', 'count'), ('profit_sum', lambda x: x[x > 0].sum()), ('loss_sum', lambda x: abs(x[x < 0].sum())), ('nb_win_trades', lambda x: x[x > 0].count())], 'trade_duration': [('avg_trade_duration', 'mean')]}
        df = results.groupby(['pair', 'stoploss'])[['profit_abs', 'trade_duration']].agg(groupby_aggregator).reset_index(col_level=1)
        df.columns = df.columns.droplevel(0)
        df['nb_loss_trades'] = df['nb_trades'] - df['nb_win_trades']
        df['average_win'] = np.where(df['nb_win_trades'] == 0, 0.0, df['profit_sum'] / df['nb_win_trades'])
        df['average_loss'] = np.where(df['nb_loss_trades'] == 0, 0.0, df['loss_sum'] / df['nb_loss_trades'])
        df['winrate'] = df['nb_win_trades'] / df['nb_trades']
        df['risk_reward_ratio'] = df['average_win'] / df['average_loss']
        df['required_risk_reward'] = 1 / df['winrate'] - 1
        df['expectancy'] = df['risk_reward_ratio'] * df['winrate'] - (1 - df['winrate'])
        df = df.sort_values(by=['expectancy', 'stoploss'], ascending=False).groupby('pair').first().sort_values(by=['expectancy'], ascending=False).reset_index()
        final = {}
        for x in df.itertuples():
            final[x.pair] = PairInfo(x.stoploss, x.winrate, x.risk_reward_ratio, x.required_risk_reward, x.expectancy, x.nb_trades, x.avg_trade_duration)
        return final

    def _find_trades_for_stoploss_range(self, df, pair: str, stoploss_range) -> list:
        if False:
            for i in range(10):
                print('nop')
        buy_column = df['enter_long'].values
        sell_column = df['exit_long'].values
        date_column = df['date'].values
        ohlc_columns = df[['open', 'high', 'low', 'close']].values
        result: list = []
        for stoploss in stoploss_range:
            result += self._detect_next_stop_or_sell_point(buy_column, sell_column, date_column, ohlc_columns, round(stoploss, 6), pair)
        return result

    def _detect_next_stop_or_sell_point(self, buy_column, sell_column, date_column, ohlc_columns, stoploss, pair: str):
        if False:
            while True:
                i = 10
        '\n        Iterate through ohlc_columns in order to find the next trade\n        Next trade opens from the first buy signal noticed to\n        The sell or stoploss signal after it.\n        It then cuts OHLC, buy_column, sell_column and date_column.\n        Cut from (the exit trade index) + 1.\n\n        Author: https://github.com/mishaker\n        '
        result: list = []
        start_point = 0
        while True:
            open_trade_index = utf1st.find_1st(buy_column, 1, utf1st.cmp_equal)
            if open_trade_index == -1 or open_trade_index == len(buy_column) - 1:
                break
            else:
                open_trade_index += 1
            open_price = ohlc_columns[open_trade_index, 0]
            stop_price = open_price * (stoploss + 1)
            stop_index = utf1st.find_1st(ohlc_columns[open_trade_index:, 2], stop_price, utf1st.cmp_smaller)
            if stop_index == -1:
                stop_index = float('inf')
            sell_index = utf1st.find_1st(sell_column[open_trade_index:], 1, utf1st.cmp_equal)
            if sell_index == -1:
                sell_index = float('inf')
            if stop_index == sell_index == float('inf'):
                break
            if stop_index <= sell_index:
                exit_index = open_trade_index + stop_index
                exit_type = ExitType.STOP_LOSS
                exit_price = stop_price
            elif stop_index > sell_index:
                exit_index = open_trade_index + sell_index + 1
                if len(ohlc_columns) - 1 < exit_index:
                    break
                exit_type = ExitType.EXIT_SIGNAL
                exit_price = ohlc_columns[exit_index, 0]
            trade = {'pair': pair, 'stoploss': stoploss, 'profit_ratio': '', 'profit_abs': '', 'open_date': date_column[open_trade_index], 'close_date': date_column[exit_index], 'trade_duration': '', 'open_rate': round(open_price, 15), 'close_rate': round(exit_price, 15), 'exit_type': exit_type}
            result.append(trade)
            buy_column = buy_column[exit_index:]
            sell_column = sell_column[exit_index:]
            date_column = date_column[exit_index:]
            ohlc_columns = ohlc_columns[exit_index:]
            start_point += exit_index
        return result