from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from typing import IO, List, Tuple, Union
from qlib.data.dataset.utils import convert_index_format
from qlib.utils import lazy_sort_index
from ...utils.resam import resam_ts_data, ts_data_last
from ...data.data import D
from ...strategy.base import BaseStrategy
from ...backtest.decision import BaseTradeDecision, Order, TradeDecisionWO, TradeRange
from ...backtest.exchange import Exchange, OrderHelper
from ...backtest.utils import CommonInfrastructure, LevelInfrastructure
from qlib.utils.file import get_io_object
from qlib.backtest.utils import get_start_end_idx

class TWAPStrategy(BaseStrategy):
    """TWAP Strategy for trading

    NOTE:
        - This TWAP strategy will celling round when trading. This will make the TWAP trading strategy produce the order
          earlier when the total trade unit of amount is less than the trading step
    """

    def reset(self, outer_trade_decision: BaseTradeDecision=None, **kwargs):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        outer_trade_decision : BaseTradeDecision, optional\n        '
        super(TWAPStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount_remain = {}
            for order in outer_trade_decision.get_decision():
                self.trade_amount_remain[order.stock_id] = order.amount

    def generate_trade_decision(self, execute_result=None):
        if False:
            while True:
                i = 10
        if len(self.outer_trade_decision.get_decision()) == 0:
            return TradeDecisionWO(order_list=[], strategy=self)
        trade_step = self.trade_calendar.get_trade_step()
        (start_idx, end_idx) = get_start_end_idx(self.trade_calendar, self.outer_trade_decision)
        trade_len = end_idx - start_idx + 1
        if trade_step < start_idx or trade_step > end_idx:
            return TradeDecisionWO(order_list=[], strategy=self)
        rel_trade_step = trade_step - start_idx
        if execute_result is not None:
            for (order, _, _, _) in execute_result:
                self.trade_amount_remain[order.stock_id] -= order.deal_amount
        (trade_start_time, trade_end_time) = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
            if self.trade_exchange.check_stock_suspended(stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time):
                continue
            amount_expect = order.amount / trade_len * (rel_trade_step + 1)
            amount_remain = self.trade_amount_remain[order.stock_id]
            amount_finished = order.amount - amount_remain
            amount_delta = amount_expect - amount_finished
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time)
            if _amount_trade_unit is None:
                amount_delta_target = amount_delta
            else:
                amount_delta_target = min(np.round(amount_delta / _amount_trade_unit) * _amount_trade_unit, amount_remain)
            if rel_trade_step == trade_len - 1:
                amount_delta_target = amount_remain
            if amount_delta_target > 1e-05:
                _order = Order(stock_id=order.stock_id, amount=amount_delta_target, start_time=trade_start_time, end_time=trade_end_time, direction=order.direction)
                order_list.append(_order)
        return TradeDecisionWO(order_list=order_list, strategy=self)

class SBBStrategyBase(BaseStrategy):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy.
    """
    TREND_MID = 0
    TREND_SHORT = 1
    TREND_LONG = 2

    def reset(self, outer_trade_decision: BaseTradeDecision=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        outer_trade_decision : BaseTradeDecision, optional\n        '
        super(SBBStrategyBase, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_trend = {}
            self.trade_amount = {}
            for order in outer_trade_decision.get_decision():
                self.trade_trend[order.stock_id] = self.TREND_MID
                self.trade_amount[order.stock_id] = order.amount

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('pred_price_trend method is not implemented!')

    def generate_trade_decision(self, execute_result=None):
        if False:
            print('Hello World!')
        trade_step = self.trade_calendar.get_trade_step()
        trade_len = self.trade_calendar.get_trade_len()
        if execute_result is not None:
            for (order, _, _, _) in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount
        (trade_start_time, trade_end_time) = self.trade_calendar.get_step_time(trade_step)
        (pred_start_time, pred_end_time) = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
            if trade_step % 2 == 0:
                _pred_trend = self._pred_price_trend(order.stock_id, pred_start_time, pred_end_time)
            else:
                _pred_trend = self.trade_trend[order.stock_id]
            if not self.trade_exchange.is_stock_tradable(stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time):
                if trade_step % 2 == 0:
                    self.trade_trend[order.stock_id] = _pred_trend
                continue
            _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time)
            if _pred_trend == self.TREND_MID:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)
                else:
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    _order_amount = (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
                if order.direction == order.SELL:
                    if self.trade_amount[order.stock_id] > 1e-05 and (_order_amount < 1e-05 or trade_step == trade_len - 1):
                        _order_amount = self.trade_amount[order.stock_id]
                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])
                if _order_amount > 1e-05:
                    _order = Order(stock_id=order.stock_id, amount=_order_amount, start_time=trade_start_time, end_time=trade_end_time, direction=order.direction)
                    order_list.append(_order)
            else:
                _order_amount = None
                if _amount_trade_unit is None:
                    _order_amount = 2 * self.trade_amount[order.stock_id] / (trade_len - trade_step + 1)
                else:
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    _order_amount = (trade_unit_cnt + trade_len - trade_step) // (trade_len - trade_step + 1) * 2 * _amount_trade_unit
                if order.direction == order.SELL:
                    if self.trade_amount[order.stock_id] > 1e-05 and (_order_amount < 1e-05 or trade_step == trade_len - 1):
                        _order_amount = self.trade_amount[order.stock_id]
                _order_amount = min(_order_amount, self.trade_amount[order.stock_id])
                if _order_amount > 1e-05:
                    if trade_step % 2 == 0:
                        if _pred_trend == self.TREND_SHORT and order.direction == order.SELL or (_pred_trend == self.TREND_LONG and order.direction == order.BUY):
                            _order = Order(stock_id=order.stock_id, amount=_order_amount, start_time=trade_start_time, end_time=trade_end_time, direction=order.direction)
                            order_list.append(_order)
                    elif _pred_trend == self.TREND_SHORT and order.direction == order.BUY or (_pred_trend == self.TREND_LONG and order.direction == order.SELL):
                        _order = Order(stock_id=order.stock_id, amount=_order_amount, start_time=trade_start_time, end_time=trade_end_time, direction=order.direction)
                        order_list.append(_order)
            if trade_step % 2 == 0:
                self.trade_trend[order.stock_id] = _pred_trend
        return TradeDecisionWO(order_list, self)

class SBBStrategyEMA(SBBStrategyBase):
    """
    (S)elect the (B)etter one among every two adjacent trading (B)ars to sell or buy with (EMA) signal.
    """

    def __init__(self, outer_trade_decision: BaseTradeDecision=None, instruments: Union[List, str]='csi300', freq: str='day', trade_exchange: Exchange=None, level_infra: LevelInfrastructure=None, common_infra: CommonInfrastructure=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        instruments : Union[List, str], optional\n            instruments of EMA signal, by default "csi300"\n        freq : str, optional\n            freq of EMA signal, by default "day"\n            Note: `freq` may be different from `time_per_step`\n        '
        if instruments is None:
            warnings.warn('`instruments` is not set, will load all stocks')
            self.instruments = 'all'
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq
        super(SBBStrategyEMA, self).__init__(outer_trade_decision, level_infra, common_infra, trade_exchange=trade_exchange, **kwargs)

    def _reset_signal(self):
        if False:
            return 10
        trade_len = self.trade_calendar.get_trade_len()
        fields = ['EMA($close, 10)-EMA($close, 20)']
        (signal_start_time, _) = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        (_, signal_end_time) = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        signal_df = D.features(self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq)
        signal_df.columns = ['signal']
        self.signal = {}
        if not signal_df.empty:
            for (stock_id, stock_val) in signal_df.groupby(level='instrument'):
                self.signal[stock_id] = stock_val['signal'].droplevel(level='instrument')

    def reset_level_infra(self, level_infra):
        if False:
            for i in range(10):
                print('nop')
        '\n        reset level-shared infra\n        - After reset the trade calendar, the signal will be changed\n        '
        super().reset_level_infra(level_infra)
        self._reset_signal()

    def _pred_price_trend(self, stock_id, pred_start_time=None, pred_end_time=None):
        if False:
            return 10
        if stock_id not in self.signal:
            return self.TREND_MID
        else:
            _sample_signal = resam_ts_data(self.signal[stock_id], pred_start_time, pred_end_time, method=ts_data_last)
            if _sample_signal is None or np.isnan(_sample_signal) or _sample_signal == 0:
                return self.TREND_MID
            elif _sample_signal > 0:
                return self.TREND_LONG
            else:
                return self.TREND_SHORT

class ACStrategy(BaseStrategy):

    def __init__(self, lamb: float=1e-06, eta: float=2.5e-06, window_size: int=20, outer_trade_decision: BaseTradeDecision=None, instruments: Union[List, str]='csi300', freq: str='day', trade_exchange: Exchange=None, level_infra: LevelInfrastructure=None, common_infra: CommonInfrastructure=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        instruments : Union[List, str], optional\n            instruments of Volatility, by default "csi300"\n        freq : str, optional\n            freq of Volatility, by default "day"\n            Note: `freq` may be different from `time_per_step`\n        '
        self.lamb = lamb
        self.eta = eta
        self.window_size = window_size
        if instruments is None:
            warnings.warn('`instruments` is not set, will load all stocks')
            self.instruments = 'all'
        if isinstance(instruments, str):
            self.instruments = D.instruments(instruments)
        self.freq = freq
        super(ACStrategy, self).__init__(outer_trade_decision, level_infra, common_infra, trade_exchange=trade_exchange, **kwargs)

    def _reset_signal(self):
        if False:
            while True:
                i = 10
        trade_len = self.trade_calendar.get_trade_len()
        fields = [f'Power(Sum(Power(Log($close/Ref($close, 1)), 2), {self.window_size})/{self.window_size - 1}-Power(Sum(Log($close/Ref($close, 1)), {self.window_size}), 2)/({self.window_size}*{self.window_size - 1}), 0.5)']
        (signal_start_time, _) = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        (_, signal_end_time) = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        signal_df = D.features(self.instruments, fields, start_time=signal_start_time, end_time=signal_end_time, freq=self.freq)
        signal_df.columns = ['volatility']
        self.signal = {}
        if not signal_df.empty:
            for (stock_id, stock_val) in signal_df.groupby(level='instrument'):
                self.signal[stock_id] = stock_val['volatility'].droplevel(level='instrument')

    def reset_level_infra(self, level_infra):
        if False:
            return 10
        '\n        reset level-shared infra\n        - After reset the trade calendar, the signal will be changed\n        '
        super().reset_level_infra(level_infra)
        self._reset_signal()

    def reset(self, outer_trade_decision: BaseTradeDecision=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        outer_trade_decision : BaseTradeDecision, optional\n        '
        super(ACStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            self.trade_amount = {}
            for order in outer_trade_decision.get_decision():
                self.trade_amount[order.stock_id] = order.amount

    def generate_trade_decision(self, execute_result=None):
        if False:
            print('Hello World!')
        trade_step = self.trade_calendar.get_trade_step()
        trade_len = self.trade_calendar.get_trade_len()
        if execute_result is not None:
            for (order, _, _, _) in execute_result:
                self.trade_amount[order.stock_id] -= order.deal_amount
        (trade_start_time, trade_end_time) = self.trade_calendar.get_step_time(trade_step)
        (pred_start_time, pred_end_time) = self.trade_calendar.get_step_time(trade_step, shift=1)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
            if not self.trade_exchange.is_stock_tradable(stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time):
                continue
            _order_amount = None
            sig_sam = resam_ts_data(self.signal[order.stock_id], pred_start_time, pred_end_time, method=ts_data_last) if order.stock_id in self.signal else None
            if sig_sam is None or np.isnan(sig_sam):
                _amount_trade_unit = self.trade_exchange.get_amount_of_trade_unit(stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time)
                if _amount_trade_unit is None:
                    _order_amount = self.trade_amount[order.stock_id] / (trade_len - trade_step)
                else:
                    trade_unit_cnt = int(self.trade_amount[order.stock_id] // _amount_trade_unit)
                    _order_amount = (trade_unit_cnt + trade_len - trade_step - 1) // (trade_len - trade_step) * _amount_trade_unit
            else:
                kappa_tild = self.lamb / self.eta * sig_sam * sig_sam
                kappa = np.arccosh(kappa_tild / 2 + 1)
                amount_ratio = (np.sinh(kappa * (trade_len - trade_step)) - np.sinh(kappa * (trade_len - trade_step - 1))) / np.sinh(kappa * trade_len)
                _order_amount = order.amount * amount_ratio
                _order_amount = self.trade_exchange.round_amount_by_trade_unit(_order_amount, stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time)
            if order.direction == order.SELL:
                if self.trade_amount[order.stock_id] > 1e-05 and (_order_amount < 1e-05 or trade_step == trade_len - 1):
                    _order_amount = self.trade_amount[order.stock_id]
            _order_amount = min(_order_amount, self.trade_amount[order.stock_id])
            if _order_amount > 1e-05:
                _order = Order(stock_id=order.stock_id, amount=_order_amount, start_time=trade_start_time, end_time=trade_end_time, direction=order.direction, factor=order.factor)
                order_list.append(_order)
        return TradeDecisionWO(order_list, self)

class RandomOrderStrategy(BaseStrategy):

    def __init__(self, trade_range: Union[Tuple[int, int], TradeRange], sample_ratio: float=1.0, volume_ratio: float=0.01, market: str='all', direction: int=Order.BUY, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        trade_range : Tuple\n            please refer to the `trade_range` parameter of BaseStrategy\n        sample_ratio : float\n            the ratio of all orders are sampled\n        volume_ratio : float\n            the volume of the total day\n            raito of the total volume of a specific day\n        market : str\n            stock pool for sampling\n        '
        super().__init__(*args, **kwargs)
        self.sample_ratio = sample_ratio
        self.volume_ratio = volume_ratio
        self.market = market
        self.direction = direction
        exch: Exchange = self.common_infra.get('trade_exchange')
        self.volume = D.features(D.instruments(market), ['Mean(Ref($volume, 1), 10)'], start_time=exch.start_time, end_time=exch.end_time)
        self.volume_df = self.volume.iloc[:, 0].unstack()
        self.trade_range = trade_range

    def generate_trade_decision(self, execute_result=None):
        if False:
            while True:
                i = 10
        trade_step = self.trade_calendar.get_trade_step()
        (step_time_start, step_time_end) = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        if step_time_start in self.volume_df:
            for (stock_id, volume) in self.volume_df[step_time_start].dropna().sample(frac=self.sample_ratio).items():
                order_list.append(self.common_infra.get('trade_exchange').get_order_helper().create(code=stock_id, amount=volume * self.volume_ratio, direction=self.direction))
        return TradeDecisionWO(order_list, self, self.trade_range)

class FileOrderStrategy(BaseStrategy):
    """
    Motivation:
    - This class provides an interface for user to read orders from csv files.
    """

    def __init__(self, file: Union[IO, str, Path, pd.DataFrame], trade_range: Union[Tuple[int, int], TradeRange]=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n\n        Parameters\n        ----------\n        file : Union[IO, str, Path, pd.DataFrame]\n            this parameters will specify the info of expected orders\n\n            Here is an example of the content\n\n            1) Amount (**adjusted**) based strategy\n\n                datetime,instrument,amount,direction\n                20200102,  SH600519,  1000,     sell\n                20200103,  SH600519,  1000,      buy\n                20200106,  SH600519,  1000,     sell\n\n        trade_range : Tuple[int, int]\n            the intra day time index range of the orders\n            the left and right is closed.\n\n            If you want to get the trade_range in intra-day\n            - `qlib/utils/time.py:def get_day_min_idx_range` can help you create the index range easier\n            # TODO: this is a trade_range level limitation. We'll implement a more detailed limitation later.\n\n        "
        super().__init__(*args, **kwargs)
        if isinstance(file, pd.DataFrame):
            self.order_df = file
        else:
            with get_io_object(file) as f:
                self.order_df = pd.read_csv(f, dtype={'datetime': str})
        self.order_df['datetime'] = self.order_df['datetime'].apply(pd.Timestamp)
        self.order_df = self.order_df.set_index(['datetime', 'instrument'])
        self.order_df = lazy_sort_index(convert_index_format(self.order_df, level='datetime'))
        self.trade_range = trade_range

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        execute_result :\n            execute_result will be ignored in FileOrderStrategy\n        '
        oh: OrderHelper = self.common_infra.get('trade_exchange').get_order_helper()
        (start, _) = self.trade_calendar.get_step_time()
        try:
            df = self.order_df.loc(axis=0)[start]
        except KeyError:
            return TradeDecisionWO([], self)
        else:
            order_list = []
            for (idx, row) in df.iterrows():
                order_list.append(oh.create(code=idx, amount=row['amount'], direction=Order.parse_dir(row['direction'])))
            return TradeDecisionWO(order_list, self, self.trade_range)