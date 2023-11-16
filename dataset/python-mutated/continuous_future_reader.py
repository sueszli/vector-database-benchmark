import numpy as np
import pandas as pd
from zipline.data.session_bars import SessionBarReader

class ContinuousFutureSessionBarReader(SessionBarReader):

    def __init__(self, bar_reader, roll_finders):
        if False:
            while True:
                i = 10
        self._bar_reader = bar_reader
        self._roll_finders = roll_finders

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        fields : list of str\n            'sid'\n        start_dt: Timestamp\n           Beginning of the window range.\n        end_dt: Timestamp\n           End of the window range.\n        sids : list of int\n           The asset identifiers in the window.\n\n        Returns\n        -------\n        list of np.ndarray\n            A list with an entry per field of ndarrays with shape\n            (minutes in range, sids) with a dtype of float64, containing the\n            values for the respective field over start and end dt range.\n        "
        rolls_by_asset = {}
        for asset in assets:
            rf = self._roll_finders[asset.roll_style]
            rolls_by_asset[asset] = rf.get_rolls(asset.root_symbol, start_date, end_date, asset.offset)
        num_sessions = len(self.trading_calendar.sessions_in_range(start_date, end_date))
        shape = (num_sessions, len(assets))
        results = []
        tc = self._bar_reader.trading_calendar
        sessions = tc.sessions_in_range(start_date, end_date)
        partitions_by_asset = {}
        for asset in assets:
            partitions = []
            partitions_by_asset[asset] = partitions
            rolls = rolls_by_asset[asset]
            start = start_date
            for roll in rolls:
                (sid, roll_date) = roll
                start_loc = sessions.get_loc(start)
                if roll_date is not None:
                    end = roll_date - sessions.freq
                    end_loc = sessions.get_loc(end)
                else:
                    end = end_date
                    end_loc = len(sessions) - 1
                partitions.append((sid, start, end, start_loc, end_loc))
                if roll_date is not None:
                    start = sessions[end_loc + 1]
        for column in columns:
            if column != 'volume' and column != 'sid':
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.int64)
            for (i, asset) in enumerate(assets):
                partitions = partitions_by_asset[asset]
                for (sid, start, end, start_loc, end_loc) in partitions:
                    if column != 'sid':
                        result = self._bar_reader.load_raw_arrays([column], start, end, [sid])[0][:, 0]
                    else:
                        result = int(sid)
                    out[start_loc:end_loc + 1, i] = result
            results.append(out)
        return results

    @property
    def last_available_dt(self):
        if False:
            return 10
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The last session for which the reader can provide data.\n        '
        return self._bar_reader.last_available_dt

    @property
    def trading_calendar(self):
        if False:
            return 10
        "\n        Returns the zipline.utils.calendar.trading_calendar used to read\n        the data.  Can be None (if the writer didn't specify it).\n        "
        return self._bar_reader.trading_calendar

    @property
    def first_trading_day(self):
        if False:
            return 10
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The first trading day (session) for which the reader can provide\n            data.\n        '
        return self._bar_reader.first_trading_day

    def get_value(self, continuous_future, dt, field):
        if False:
            i = 10
            return i + 15
        "\n        Retrieve the value at the given coordinates.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        dt : pd.Timestamp\n            The timestamp for the desired data point.\n        field : string\n            The OHLVC name for the desired data point.\n\n        Returns\n        -------\n        value : float|int\n            The value at the given coordinates, ``float`` for OHLC, ``int``\n            for 'volume'.\n\n        Raises\n        ------\n        NoDataOnDate\n            If the given dt is not a valid market minute (in minute mode) or\n            session (in daily mode) according to this reader's tradingcalendar.\n        "
        rf = self._roll_finders[continuous_future.roll_style]
        sid = rf.get_contract_center(continuous_future.root_symbol, dt, continuous_future.offset)
        return self._bar_reader.get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        if False:
            i = 10
            return i + 15
        '\n        Get the latest minute on or before ``dt`` in which ``asset`` traded.\n\n        If there are no trades on or before ``dt``, returns ``pd.NaT``.\n\n        Parameters\n        ----------\n        asset : zipline.asset.Asset\n            The asset for which to get the last traded minute.\n        dt : pd.Timestamp\n            The minute at which to start searching for the last traded minute.\n\n        Returns\n        -------\n        last_traded : pd.Timestamp\n            The dt of the last trade for the given asset, using the input\n            dt as a vantage point.\n        '
        rf = self._roll_finders[asset.roll_style]
        sid = rf.get_contract_center(asset.root_symbol, dt, asset.offset)
        if sid is None:
            return pd.NaT
        contract = rf.asset_finder.retrieve_asset(sid)
        return self._bar_reader.get_last_traded_dt(contract, dt)

    @property
    def sessions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        sessions : DatetimeIndex\n           All session labels (unioning the range for all assets) which the\n           reader can provide.\n        '
        return self._bar_reader.sessions

class ContinuousFutureMinuteBarReader(SessionBarReader):

    def __init__(self, bar_reader, roll_finders):
        if False:
            for i in range(10):
                print('nop')
        self._bar_reader = bar_reader
        self._roll_finders = roll_finders

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        fields : list of str\n           'open', 'high', 'low', 'close', or 'volume'\n        start_dt: Timestamp\n           Beginning of the window range.\n        end_dt: Timestamp\n           End of the window range.\n        sids : list of int\n           The asset identifiers in the window.\n\n        Returns\n        -------\n        list of np.ndarray\n            A list with an entry per field of ndarrays with shape\n            (minutes in range, sids) with a dtype of float64, containing the\n            values for the respective field over start and end dt range.\n        "
        rolls_by_asset = {}
        tc = self.trading_calendar
        start_session = tc.minute_to_session_label(start_date)
        end_session = tc.minute_to_session_label(end_date)
        for asset in assets:
            rf = self._roll_finders[asset.roll_style]
            rolls_by_asset[asset] = rf.get_rolls(asset.root_symbol, start_session, end_session, asset.offset)
        sessions = tc.sessions_in_range(start_date, end_date)
        minutes = tc.minutes_in_range(start_date, end_date)
        num_minutes = len(minutes)
        shape = (num_minutes, len(assets))
        results = []
        partitions_by_asset = {}
        for asset in assets:
            partitions = []
            partitions_by_asset[asset] = partitions
            rolls = rolls_by_asset[asset]
            start = start_date
            for roll in rolls:
                (sid, roll_date) = roll
                start_loc = minutes.searchsorted(start)
                if roll_date is not None:
                    (_, end) = tc.open_and_close_for_session(roll_date - sessions.freq)
                    end_loc = minutes.searchsorted(end)
                else:
                    end = end_date
                    end_loc = len(minutes) - 1
                partitions.append((sid, start, end, start_loc, end_loc))
                if roll[-1] is not None:
                    (start, _) = tc.open_and_close_for_session(tc.minute_to_session_label(minutes[end_loc + 1]))
        for column in columns:
            if column != 'volume':
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.uint32)
            for (i, asset) in enumerate(assets):
                partitions = partitions_by_asset[asset]
                for (sid, start, end, start_loc, end_loc) in partitions:
                    if column != 'sid':
                        result = self._bar_reader.load_raw_arrays([column], start, end, [sid])[0][:, 0]
                    else:
                        result = int(sid)
                    out[start_loc:end_loc + 1, i] = result
            results.append(out)
        return results

    @property
    def last_available_dt(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The last session for which the reader can provide data.\n        '
        return self._bar_reader.last_available_dt

    @property
    def trading_calendar(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the zipline.utils.calendar.trading_calendar used to read\n        the data.  Can be None (if the writer didn't specify it).\n        "
        return self._bar_reader.trading_calendar

    @property
    def first_trading_day(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The first trading day (session) for which the reader can provide\n            data.\n        '
        return self._bar_reader.first_trading_day

    def get_value(self, continuous_future, dt, field):
        if False:
            while True:
                i = 10
        "\n        Retrieve the value at the given coordinates.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        dt : pd.Timestamp\n            The timestamp for the desired data point.\n        field : string\n            The OHLVC name for the desired data point.\n\n        Returns\n        -------\n        value : float|int\n            The value at the given coordinates, ``float`` for OHLC, ``int``\n            for 'volume'.\n\n        Raises\n        ------\n        NoDataOnDate\n            If the given dt is not a valid market minute (in minute mode) or\n            session (in daily mode) according to this reader's tradingcalendar.\n        "
        rf = self._roll_finders[continuous_future.roll_style]
        sid = rf.get_contract_center(continuous_future.root_symbol, dt, continuous_future.offset)
        return self._bar_reader.get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the latest minute on or before ``dt`` in which ``asset`` traded.\n\n        If there are no trades on or before ``dt``, returns ``pd.NaT``.\n\n        Parameters\n        ----------\n        asset : zipline.asset.Asset\n            The asset for which to get the last traded minute.\n        dt : pd.Timestamp\n            The minute at which to start searching for the last traded minute.\n\n        Returns\n        -------\n        last_traded : pd.Timestamp\n            The dt of the last trade for the given asset, using the input\n            dt as a vantage point.\n        '
        rf = self._roll_finders[asset.roll_style]
        sid = rf.get_contract_center(asset.root_symbol, dt, asset.offset)
        if sid is None:
            return pd.NaT
        contract = rf.asset_finder.retrieve_asset(sid)
        return self._bar_reader.get_last_traded_dt(contract, dt)

    @property
    def sessions(self):
        if False:
            print('Hello World!')
        return self._bar_reader.sessions