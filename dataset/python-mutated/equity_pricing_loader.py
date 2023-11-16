from collections import defaultdict
from interface import implements
from numpy import iinfo, uint32, multiply
from zipline.data.fx import ExplodingFXRateReader
from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import repeat_first_axis
from .base import PipelineLoader
from .utils import shift_dates
from ..data.equity_pricing import EquityPricing
UINT32_MAX = iinfo(uint32).max

class EquityPricingLoader(implements(PipelineLoader)):
    """A PipelineLoader for loading daily OHLCV data.

    Parameters
    ----------
    raw_price_reader : zipline.data.session_bars.SessionBarReader
        Reader providing raw prices.
    adjustments_reader : zipline.data.adjustments.SQLiteAdjustmentReader
        Reader providing price/volume adjustments.
    fx_reader : zipline.data.fx.FXRateReader
       Reader providing currency conversions.
    """

    def __init__(self, raw_price_reader, adjustments_reader, fx_reader):
        if False:
            for i in range(10):
                print('nop')
        self.raw_price_reader = raw_price_reader
        self.adjustments_reader = adjustments_reader
        self.fx_reader = fx_reader

    @classmethod
    def without_fx(cls, raw_price_reader, adjustments_reader):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct an EquityPricingLoader without support for fx rates.\n\n        The returned loader will raise an error if requested to load\n        currency-converted columns.\n\n        Parameters\n        ----------\n        raw_price_reader : zipline.data.session_bars.SessionBarReader\n            Reader providing raw prices.\n        adjustments_reader : zipline.data.adjustments.SQLiteAdjustmentReader\n            Reader providing price/volume adjustments.\n\n        Returns\n        -------\n        loader : EquityPricingLoader\n            A loader that can only provide currency-naive data.\n        '
        return cls(raw_price_reader=raw_price_reader, adjustments_reader=adjustments_reader, fx_reader=ExplodingFXRateReader())

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        if False:
            while True:
                i = 10
        sessions = domain.all_sessions()
        shifted_dates = shift_dates(sessions, dates[0], dates[-1], shift=1)
        (ohlcv_cols, currency_cols) = self._split_column_types(columns)
        del columns
        ohlcv_colnames = [c.name for c in ohlcv_cols]
        raw_ohlcv_arrays = self.raw_price_reader.load_raw_arrays(ohlcv_colnames, shifted_dates[0], shifted_dates[-1], sids)
        self._inplace_currency_convert(ohlcv_cols, raw_ohlcv_arrays, shifted_dates, sids)
        adjustments = self.adjustments_reader.load_pricing_adjustments(ohlcv_colnames, dates, sids)
        out = {}
        for (c, c_raw, c_adjs) in zip(ohlcv_cols, raw_ohlcv_arrays, adjustments):
            out[c] = AdjustedArray(c_raw.astype(c.dtype), c_adjs, c.missing_value)
        for c in currency_cols:
            codes_1d = self.raw_price_reader.currency_codes(sids)
            codes = repeat_first_axis(codes_1d, len(dates))
            out[c] = AdjustedArray(codes, adjustments={}, missing_value=None)
        return out

    @property
    def currency_aware(self):
        if False:
            for i in range(10):
                print('nop')
        return not isinstance(self.fx_reader, ExplodingFXRateReader)

    def _inplace_currency_convert(self, columns, arrays, dates, sids):
        if False:
            for i in range(10):
                print('nop')
        '\n        Currency convert raw data loaded for ``column``.\n\n        Parameters\n        ----------\n        columns : list[zipline.pipeline.data.BoundColumn]\n            List of columns whose raw data has been loaded.\n        arrays : list[np.array]\n            List of arrays, parallel to ``columns`` containing data for the\n            column.\n        dates : pd.DatetimeIndex\n            Labels for rows of ``arrays``. These are the dates that should\n            be used to fetch fx rates for conversion.\n        sids : np.array[int64]\n            Labels for columns of ``arrays``.\n\n        Returns\n        -------\n        None\n\n        Side Effects\n        ------------\n        Modifies ``arrays`` in place by applying currency conversions.\n        '
        by_spec = defaultdict(list)
        for (column, array) in zip(columns, arrays):
            by_spec[column.currency_conversion].append(array)
        by_spec.pop(None, None)
        if not by_spec:
            return
        fx_reader = self.fx_reader
        base_currencies = self.raw_price_reader.currency_codes(sids)
        for (spec, arrays) in by_spec.items():
            rates = fx_reader.get_rates(rate=spec.field, quote=spec.currency.code, bases=base_currencies, dts=dates)
            for arr in arrays:
                multiply(arr, rates, out=arr)

    def _split_column_types(self, columns):
        if False:
            print('Hello World!')
        'Split out currency columns from OHLCV columns.\n\n        Parameters\n        ----------\n        columns : list[zipline.pipeline.data.BoundColumn]\n            Columns to be loaded by ``load_adjusted_array``.\n\n        Returns\n        -------\n        ohlcv_columns : list[zipline.pipeline.data.BoundColumn]\n            Price and volume columns from ``columns``.\n        currency_columns : list[zipline.pipeline.data.BoundColumn]\n            Currency code column from ``columns``, if present.\n        '
        currency_name = EquityPricing.currency.name
        ohlcv = []
        currency = []
        for c in columns:
            if c.name == currency_name:
                currency.append(c)
            else:
                ohlcv.append(c)
        return (ohlcv, currency)
USEquityPricingLoader = EquityPricingLoader