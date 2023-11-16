from six import iteritems
import numpy as np
import pandas as pd
from pandas import NaT
from trading_calendars import TradingCalendar
from zipline.data.bar_reader import OHLCV, NoDataOnDate, NoDataForSid
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.input_validation import expect_types, validate_keys
from zipline.utils.pandas_utils import check_indexes_all_same

class InMemoryDailyBarReader(CurrencyAwareSessionBarReader):
    """
    A SessionBarReader backed by a dictionary of in-memory DataFrames.

    Parameters
    ----------
    frames : dict[str -> pd.DataFrame]
        Dictionary from field name ("open", "high", "low", "close", or
        "volume") to DataFrame containing data for that field.
    calendar : str or trading_calendars.TradingCalendar
        Calendar (or name of calendar) to which data is aligned.
    currency_codes : pd.Series
        Map from sid -> listing currency for that sid.
    verify_indices : bool, optional
        Whether or not to verify that input data is correctly aligned to the
        given calendar. Default is True.
    """

    @expect_types(frames=dict, calendar=TradingCalendar, verify_indices=bool, currency_codes=pd.Series)
    def __init__(self, frames, calendar, currency_codes, verify_indices=True):
        if False:
            while True:
                i = 10
        self._frames = frames
        self._values = {key: frame.values for (key, frame) in iteritems(frames)}
        self._calendar = calendar
        self._currency_codes = currency_codes
        validate_keys(frames, set(OHLCV), type(self).__name__)
        if verify_indices:
            verify_frames_aligned(list(frames.values()), calendar)
        self._sessions = frames['close'].index
        self._sids = frames['close'].columns

    @classmethod
    def from_panel(cls, panel, calendar, currency_codes):
        if False:
            for i in range(10):
                print('nop')
        'Helper for construction from a pandas.Panel.\n        '
        return cls(dict(panel.iteritems()), calendar, currency_codes)

    @property
    def last_available_dt(self):
        if False:
            i = 10
            return i + 15
        return self._calendar[-1]

    @property
    def trading_calendar(self):
        if False:
            return 10
        return self._calendar

    @property
    def sessions(self):
        if False:
            return 10
        return self._sessions

    def load_raw_arrays(self, columns, start_dt, end_dt, assets):
        if False:
            print('Hello World!')
        if start_dt not in self._sessions:
            raise NoDataOnDate(start_dt)
        if end_dt not in self._sessions:
            raise NoDataOnDate(end_dt)
        asset_indexer = self._sids.get_indexer(assets)
        if -1 in asset_indexer:
            bad_assets = assets[asset_indexer == -1]
            raise NoDataForSid(bad_assets)
        date_indexer = self._sessions.slice_indexer(start_dt, end_dt)
        out = []
        for c in columns:
            out.append(self._values[c][date_indexer, asset_indexer])
        return out

    def get_value(self, sid, dt, field):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        day : datetime64-like\n            Midnight of the day for which data is requested.\n        field : string\n            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')\n\n        Returns\n        -------\n        float\n            The spot price for colname of the given sid on the given day.\n            Raises a NoDataOnDate exception if the given day and sid is before\n            or after the date range of the equity.\n            Returns -1 if the day is within the date range, but the price is\n            0.\n        "
        return self.frames[field].loc[dt, sid]

    def get_last_traded_dt(self, asset, dt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        asset : zipline.asset.Asset\n            The asset identifier.\n        dt : datetime64-like\n            Midnight of the day for which data is requested.\n\n        Returns\n        -------\n        pd.Timestamp : The last know dt for the asset and dt;\n                       NaT if no trade is found before the given dt.\n        '
        try:
            return self.frames['close'].loc[:, asset.sid].last_valid_index()
        except IndexError:
            return NaT

    @property
    def first_trading_day(self):
        if False:
            i = 10
            return i + 15
        return self._sessions[0]

    def currency_codes(self, sids):
        if False:
            while True:
                i = 10
        codes = self._currency_codes
        return np.array([codes[sid] for sid in sids])

def verify_frames_aligned(frames, calendar):
    if False:
        print('Hello World!')
    '\n    Verify that DataFrames in ``frames`` have the same indexing scheme and are\n    aligned to ``calendar``.\n\n    Parameters\n    ----------\n    frames : list[pd.DataFrame]\n    calendar : trading_calendars.TradingCalendar\n\n    Raises\n    ------\n    ValueError\n        If frames have different indexes/columns, or if frame indexes do not\n        match a contiguous region of ``calendar``.\n    '
    indexes = [f.index for f in frames]
    check_indexes_all_same(indexes, message="DataFrame indexes don't match:")
    columns = [f.columns for f in frames]
    check_indexes_all_same(columns, message="DataFrame columns don't match:")
    (start, end) = indexes[0][[0, -1]]
    cal_sessions = calendar.sessions_in_range(start, end)
    check_indexes_all_same([indexes[0], cal_sessions], "DataFrame index doesn't match {} calendar:".format(calendar.name))