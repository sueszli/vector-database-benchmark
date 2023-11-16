from abc import ABCMeta, abstractmethod, abstractproperty
from six import with_metaclass

class NoDataOnDate(Exception):
    """
    Raised when a spot price cannot be found for the sid and date.
    """
    pass

class NoDataBeforeDate(NoDataOnDate):
    pass

class NoDataAfterDate(NoDataOnDate):
    pass

class NoDataForSid(Exception):
    """
    Raised when the requested sid is missing from the pricing data.
    """
    pass
OHLCV = ('open', 'high', 'low', 'close', 'volume')

class BarReader(with_metaclass(ABCMeta, object)):

    @abstractproperty
    def data_frequency(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def load_raw_arrays(self, columns, start_date, end_date, assets):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        columns : list of str\n           'open', 'high', 'low', 'close', or 'volume'\n        start_date: Timestamp\n           Beginning of the window range.\n        end_date: Timestamp\n           End of the window range.\n        assets : list of int\n           The asset identifiers in the window.\n\n        Returns\n        -------\n        list of np.ndarray\n            A list with an entry per field of ndarrays with shape\n            (minutes in range, sids) with a dtype of float64, containing the\n            values for the respective field over start and end dt range.\n        "
        pass

    @abstractproperty
    def last_available_dt(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The last session for which the reader can provide data.\n        '
        pass

    @abstractproperty
    def trading_calendar(self):
        if False:
            return 10
        "\n        Returns the zipline.utils.calendar.trading_calendar used to read\n        the data.  Can be None (if the writer didn't specify it).\n        "
        pass

    @abstractproperty
    def first_trading_day(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        dt : pd.Timestamp\n            The first trading day (session) for which the reader can provide\n            data.\n        '
        pass

    @abstractmethod
    def get_value(self, sid, dt, field):
        if False:
            for i in range(10):
                print('nop')
        "\n        Retrieve the value at the given coordinates.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        dt : pd.Timestamp\n            The timestamp for the desired data point.\n        field : string\n            The OHLVC name for the desired data point.\n\n        Returns\n        -------\n        value : float|int\n            The value at the given coordinates, ``float`` for OHLC, ``int``\n            for 'volume'.\n\n        Raises\n        ------\n        NoDataOnDate\n            If the given dt is not a valid market minute (in minute mode) or\n            session (in daily mode) according to this reader's tradingcalendar.\n        "
        pass

    @abstractmethod
    def get_last_traded_dt(self, asset, dt):
        if False:
            i = 10
            return i + 15
        '\n        Get the latest minute on or before ``dt`` in which ``asset`` traded.\n\n        If there are no trades on or before ``dt``, returns ``pd.NaT``.\n\n        Parameters\n        ----------\n        asset : zipline.asset.Asset\n            The asset for which to get the last traded minute.\n        dt : pd.Timestamp\n            The minute at which to start searching for the last traded minute.\n\n        Returns\n        -------\n        last_traded : pd.Timestamp\n            The dt of the last trade for the given asset, using the input\n            dt as a vantage point.\n        '
        pass