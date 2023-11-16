from datetime import datetime, date
from typing import Union, List, Sequence, Optional, Tuple
import six
import numpy as np
import pandas as pd
from rqalpha.const import INSTRUMENT_TYPE, TRADING_CALENDAR_TYPE, EXECUTION_PHASE
from rqalpha.utils import risk_free_helper, TimeRange, merge_trading_period
from rqalpha.data.trading_dates_mixin import TradingDatesMixin
from rqalpha.model.bar import BarObject, NANDict, PartialBarObject
from rqalpha.model.tick import TickObject
from rqalpha.model.instrument import Instrument
from rqalpha.model.order import ALGO_ORDER_STYLES
from rqalpha.utils.functools import lru_cache
from rqalpha.utils.datetime_func import convert_int_to_datetime, convert_date_to_int
from rqalpha.utils.typing import DateLike, StrOrIter
from rqalpha.interface import AbstractDataSource, AbstractPriceBoard
from rqalpha.core.execution_context import ExecutionContext

class DataProxy(TradingDatesMixin):

    def __init__(self, data_source, price_board):
        if False:
            print('Hello World!')
        self._data_source = data_source
        self._price_board = price_board
        try:
            trading_calendars = data_source.get_trading_calendars()
        except NotImplementedError:
            trading_calendars = {TRADING_CALENDAR_TYPE.EXCHANGE: data_source.get_trading_calendar()}
        TradingDatesMixin.__init__(self, trading_calendars)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        return getattr(self._data_source, item)

    def get_trading_minutes_for(self, order_book_id, dt):
        if False:
            print('Hello World!')
        instrument = self.instruments(order_book_id)
        minutes = self._data_source.get_trading_minutes_for(instrument, dt)
        return [] if minutes is None else minutes

    def get_yield_curve(self, start_date, end_date, tenor=None):
        if False:
            while True:
                i = 10
        if isinstance(tenor, six.string_types):
            tenor = [tenor]
        return self._data_source.get_yield_curve(start_date, end_date, tenor)

    def get_risk_free_rate(self, start_date, end_date):
        if False:
            while True:
                i = 10
        tenors = risk_free_helper.get_tenors_for(start_date, end_date)
        _s = start_date if self.is_trading_date(start_date) else self.get_next_trading_date(start_date, n=1)
        yc = self._data_source.get_yield_curve(_s, _s)
        if yc is None or yc.empty:
            return np.nan
        yc = yc.iloc[0]
        for tenor in tenors[::-1]:
            rate = yc.get(tenor)
            if rate and (not np.isnan(rate)):
                return rate
        else:
            return np.nan

    def get_dividend(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        instrument = self.instruments(order_book_id)
        return self._data_source.get_dividend(instrument)

    def get_split(self, order_book_id):
        if False:
            while True:
                i = 10
        instrument = self.instruments(order_book_id)
        return self._data_source.get_split(instrument)

    def get_dividend_by_book_date(self, order_book_id, date):
        if False:
            i = 10
            return i + 15
        table = self._data_source.get_dividend(self.instruments(order_book_id))
        if table is None or len(table) == 0:
            return
        try:
            dates = table['book_closure_date']
        except ValueError:
            dates = table['ex_dividend_date']
            date = self.get_next_trading_date(date)
        dt = date.year * 10000 + date.month * 100 + date.day
        left_pos = dates.searchsorted(dt)
        right_pos = dates.searchsorted(dt, side='right')
        if left_pos >= right_pos:
            return None
        return table[left_pos:right_pos]

    def get_split_by_ex_date(self, order_book_id, date):
        if False:
            i = 10
            return i + 15
        df = self.get_split(order_book_id)
        if df is None or len(df) == 0:
            return
        dt = convert_date_to_int(date)
        pos = df['ex_date'].searchsorted(dt)
        if pos == len(df) or df['ex_date'][pos] != dt:
            return None
        return df['split_factor'][pos]

    @lru_cache(10240)
    def _get_prev_close(self, order_book_id, dt):
        if False:
            for i in range(10):
                print('nop')
        instrument = self.instruments(order_book_id)
        prev_trading_date = self.get_previous_trading_date(dt)
        bar = self._data_source.history_bars(instrument, 1, '1d', 'close', prev_trading_date, skip_suspended=False, include_now=False, adjust_orig=dt)
        if bar is None or len(bar) < 1:
            return np.nan
        return bar[0]

    def get_prev_close(self, order_book_id, dt):
        if False:
            i = 10
            return i + 15
        return self._get_prev_close(order_book_id, dt.replace(hour=0, minute=0, second=0))

    @lru_cache(10240)
    def _get_prev_settlement(self, instrument, dt):
        if False:
            for i in range(10):
                print('nop')
        bar = self._data_source.history_bars(instrument, 1, '1d', fields='prev_settlement', dt=dt, skip_suspended=False, adjust_orig=dt)
        if bar is None or len(bar) == 0:
            return np.nan
        return bar[0]

    @lru_cache(10240)
    def _get_settlement(self, instrument, dt):
        if False:
            return 10
        bar = self._data_source.history_bars(instrument, 1, '1d', 'settlement', dt, skip_suspended=False)
        if bar is None or len(bar) == 0:
            raise LookupError("'{}', dt={}".format(instrument.order_book_id, dt))
        return bar[0]

    def get_prev_settlement(self, order_book_id, dt):
        if False:
            i = 10
            return i + 15
        instrument = self.instruments(order_book_id)
        if instrument.type not in (INSTRUMENT_TYPE.FUTURE, INSTRUMENT_TYPE.OPTION):
            return np.nan
        return self._get_prev_settlement(instrument, dt)

    def get_settlement(self, instrument, dt):
        if False:
            for i in range(10):
                print('nop')
        if instrument.type != INSTRUMENT_TYPE.FUTURE:
            raise LookupError("'{}', instrument_type={}".format(instrument.order_book_id, instrument.type))
        return self._get_settlement(instrument, dt)

    def get_settle_price(self, order_book_id, date):
        if False:
            i = 10
            return i + 15
        instrument = self.instruments(order_book_id)
        if instrument.type != 'Future':
            return np.nan
        return self._data_source.get_settle_price(instrument, date)

    @lru_cache(512)
    def get_bar(self, order_book_id: str, dt: date, frequency: str='1d') -> BarObject:
        if False:
            i = 10
            return i + 15
        instrument = self.instruments(order_book_id)
        if dt is None:
            return BarObject(instrument, NANDict, dt)
        bar = self._data_source.get_bar(instrument, dt, frequency)
        if bar:
            return BarObject(instrument, bar)
        return BarObject(instrument, NANDict, dt)

    def get_open_auction_bar(self, order_book_id, dt):
        if False:
            while True:
                i = 10
        instrument = self.instruments(order_book_id)
        try:
            bar = self._data_source.get_open_auction_bar(instrument, dt)
        except NotImplementedError:
            tick = self.current_snapshot(order_book_id, '1d', dt)
            bar = {k: getattr(tick, k) for k in ['datetime', 'open', 'limit_up', 'limit_down', 'volume', 'total_turnover']}
        return PartialBarObject(instrument, bar)

    def history(self, order_book_id, bar_count, frequency, field, dt):
        if False:
            while True:
                i = 10
        data = self.history_bars(order_book_id, bar_count, frequency, ['datetime', field], dt, skip_suspended=False, adjust_orig=dt)
        if data is None:
            return None
        return pd.Series(data[field], index=[convert_int_to_datetime(t) for t in data['datetime']])

    def fast_history(self, order_book_id, bar_count, frequency, field, dt):
        if False:
            i = 10
            return i + 15
        return self.history_bars(order_book_id, bar_count, frequency, field, dt, skip_suspended=False, adjust_type='pre', adjust_orig=dt)

    def history_bars(self, order_book_id, bar_count, frequency, field, dt, skip_suspended=True, include_now=False, adjust_type='pre', adjust_orig=None):
        if False:
            i = 10
            return i + 15
        instrument = self.instruments(order_book_id)
        if adjust_orig is None:
            adjust_orig = dt
        return self._data_source.history_bars(instrument, bar_count, frequency, field, dt, skip_suspended=skip_suspended, include_now=include_now, adjust_type=adjust_type, adjust_orig=adjust_orig)

    def history_ticks(self, order_book_id, count, dt):
        if False:
            while True:
                i = 10
        instrument = self.instruments(order_book_id)
        return self._data_source.history_ticks(instrument, count, dt)

    def current_snapshot(self, order_book_id, frequency, dt):
        if False:
            i = 10
            return i + 15

        def tick_fields_for(ins):
            if False:
                for i in range(10):
                    print('nop')
            _STOCK_FIELD_NAMES = ['datetime', 'open', 'high', 'low', 'last', 'volume', 'total_turnover', 'prev_close', 'limit_up', 'limit_down']
            _FUTURE_FIELD_NAMES = _STOCK_FIELD_NAMES + ['open_interest', 'prev_settlement']
            if ins.type not in [INSTRUMENT_TYPE.FUTURE, INSTRUMENT_TYPE.OPTION]:
                return _STOCK_FIELD_NAMES
            else:
                return _FUTURE_FIELD_NAMES
        instrument = self.instruments(order_book_id)
        if frequency == '1d':
            bar = self._data_source.get_bar(instrument, dt, '1d')
            if not bar:
                return None
            d = {k: bar[k] for k in tick_fields_for(instrument) if k in bar.dtype.names}
            d['last'] = bar['open'] if ExecutionContext.phase() == EXECUTION_PHASE.OPEN_AUCTION else bar['close']
            d['prev_close'] = self._get_prev_close(order_book_id, dt)
            return TickObject(instrument, d)
        return self._data_source.current_snapshot(instrument, frequency, dt)

    def available_data_range(self, frequency):
        if False:
            print('Hello World!')
        return self._data_source.available_data_range(frequency)

    def get_commission_info(self, order_book_id):
        if False:
            print('Hello World!')
        instrument = self.instruments(order_book_id)
        return self._data_source.get_commission_info(instrument)

    def get_merge_ticks(self, order_book_id_list, trading_date, last_dt=None):
        if False:
            for i in range(10):
                print('nop')
        return self._data_source.get_merge_ticks(order_book_id_list, trading_date, last_dt)

    def is_suspended(self, order_book_id, dt, count=1):
        if False:
            print('Hello World!')
        if count == 1:
            return self._data_source.is_suspended(order_book_id, [dt])[0]
        trading_dates = self.get_n_trading_dates_until(dt, count)
        return self._data_source.is_suspended(order_book_id, trading_dates)

    def is_st_stock(self, order_book_id, dt, count=1):
        if False:
            while True:
                i = 10
        if count == 1:
            return self._data_source.is_st_stock(order_book_id, [dt])[0]
        trading_dates = self.get_n_trading_dates_until(dt, count)
        return self._data_source.is_st_stock(order_book_id, trading_dates)

    def get_tick_size(self, order_book_id):
        if False:
            while True:
                i = 10
        return self.instruments(order_book_id).tick_size()

    def get_last_price(self, order_book_id):
        if False:
            print('Hello World!')
        return float(self._price_board.get_last_price(order_book_id))

    def all_instruments(self, types, dt=None):
        if False:
            for i in range(10):
                print('nop')
        li = []
        for i in self._data_source.get_instruments(types=types):
            if dt is None or i.listing_at(dt):
                li.append(i)
        return li

    @lru_cache(2048)
    def instrument(self, sym_or_id):
        if False:
            i = 10
            return i + 15
        return next(iter(self._data_source.get_instruments(id_or_syms=[sym_or_id])), None)

    def instruments(self, sym_or_ids):
        if False:
            return 10
        if isinstance(sym_or_ids, str):
            return next(iter(self._data_source.get_instruments(id_or_syms=[sym_or_ids])), None)
        else:
            return list(self._data_source.get_instruments(id_or_syms=sym_or_ids))

    def get_future_contracts(self, underlying, date):
        if False:
            while True:
                i = 10
        return sorted((i.order_book_id for i in self.all_instruments([INSTRUMENT_TYPE.FUTURE], date) if i.underlying_symbol == underlying and (not Instrument.is_future_continuous_contract(i.order_book_id))))

    def get_trading_period(self, sym_or_ids, default_trading_period=None):
        if False:
            for i in range(10):
                print('nop')
        trading_period = default_trading_period or []
        for instrument in self.instruments(sym_or_ids):
            trading_period.extend(instrument.trading_hours or [])
        return merge_trading_period(trading_period)

    def is_night_trading(self, sym_or_ids):
        if False:
            while True:
                i = 10
        return any((instrument.trade_at_night for instrument in self.instruments(sym_or_ids)))

    def get_algo_bar(self, id_or_ins, order_style, dt):
        if False:
            print('Hello World!')
        if not isinstance(order_style, ALGO_ORDER_STYLES):
            raise RuntimeError('get_algo_bar only support VWAPOrder and TWAPOrder')
        if not isinstance(id_or_ins, Instrument):
            id_or_ins = self.instrument(id_or_ins)
        if id_or_ins is None:
            return (np.nan, 0)
        day_bar = self.get_bar(order_book_id=id_or_ins.order_book_id, dt=dt, frequency='1d')
        if day_bar.volume == 0:
            return (np.nan, 0)
        bar = self._data_source.get_algo_bar(id_or_ins, order_style.start_min, order_style.end_min, dt)
        return (bar[order_style.TYPE], bar['volume']) if bar else (np.nan, 0)