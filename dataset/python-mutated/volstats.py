__author__ = 'saeedamen'
import math
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from findatapy.timeseries import Calculations, Filter, Timezone
from findatapy.timeseries import Calendar

class VolStats(object):
    """Arranging underlying volatility market in easier to read format. Also provides methods for calculating various
    volatility metrics, such as realized_vol volatility and volatility risk premium. Has extensive support for estimating
    implied_vol volatility addons.

    """

    def __init__(self, market_df=None, intraday_spot_df=None):
        if False:
            print('Hello World!')
        self._market_df = market_df
        self._intraday_spot_df = intraday_spot_df
        self._calculations = Calculations()
        self._timezone = Timezone()
        self._filter = Filter()

    def calculate_realized_vol(self, asset, spot_df=None, returns_df=None, tenor_label='ON', freq='daily', freq_min_mult=1, hour_of_day=10, minute_of_day=0, field='close', returns_calc='simple', timezone_hour_minute='America/New_York'):
        if False:
            while True:
                i = 10
        "Calculates rolling realized vol with daily cutoffs either using daily spot data or intraday spot data\n        (which is assumed to be in UTC timezone)\n\n        Parameters\n        ----------\n        asset : str\n            asset to be calculated\n\n        spot_df : pd.DataFrame\n            minute spot returns (freq_min_mult should be the same as the frequency and should have timezone set)\n\n        tenor_label : str\n            tenor to calculate\n\n        freq_min_mult : int\n            frequency multiply for data (1 = 1 min)\n\n        hour_of_day : closing time of data in the timezone specified\n            eg. 10 which is 1000 time (default = 10)\n\n        minute_of_day : closing time of data in the timezone specified\n            eg. 0 which is 0 time (default = 0)\n\n        field : str\n            By default 'close'\n\n        returns_calc : str\n            'simple' calculate simple returns\n            'log' calculate log returns\n\n        timezone_hour_minute : str\n            The timezone for the closing hour/minute (default: 'America/New_York')\n\n        Returns\n        -------\n        pd.DataFrame of realized volatility\n        "
        if returns_df is None:
            if spot_df is None:
                if freq == 'daily':
                    spot_df = self._market_df[asset + '.' + field]
                else:
                    spot_df = self._intraday_spot_df[asset + '.' + field]
            if returns_calc == 'simple':
                returns_df = self._calculations.calculate_returns(spot_df)
            else:
                returns_df = self._calculations.calculate_log_returns(spot_df)
        cal = Calendar()
        tenor_days = cal.get_business_days_tenor(tenor_label)
        if freq == 'intraday':
            mult = int(1440.0 / float(freq_min_mult))
            realized_rolling = self._calculations.rolling_volatility(returns_df, tenor_days * mult, obs_in_year=252 * mult)
            realized_rolling = self._timezone.convert_index_aware_to_alt(realized_rolling, timezone_hour_minute)
            realized_vol = self._filter.filter_time_series_by_time_of_day(hour_of_day, minute_of_day, realized_rolling)
            realized_vol = self._timezone.convert_index_aware_to_UTC_time(realized_vol)
            realized_vol = self._timezone.set_as_no_timezone(realized_vol)
        elif freq == 'daily':
            realized_vol = self._calculations.rolling_volatility(spot_df, tenor_days, obs_in_year=252)
        realized_vol.index = realized_vol.index.date
        realized_vol = pd.DataFrame(realized_vol)
        realized_vol.columns = [asset + 'H' + tenor_label + '.close']
        return realized_vol

    def calculate_vol_risk_premium(self, asset, tenor_label='ON', implied_vol=None, realized_vol=None, field='close', adj_ON_friday=False):
        if False:
            for i in range(10):
                print('nop')
        "Calculates volatility risk premium given implied and realized quotes (ie. implied - realized) and tenor\n\n        Calculates both a version which is aligned (VRP), where the implied and realized volatilities cover\n        the same period (note: you will have a gap for recent points, where you can't grab future implied volatilities),\n        and an unaligned version (VRPV), which is the typical one used in the market\n\n        Parameters\n        ----------\n        asset : str\n            asset to calculate value for\n\n        tenor_label : str\n            tenor to calculate\n\n        implied_vol : pd.DataFrame\n            implied vol quotes where columns are of the form eg. EURUSDV1M.close\n\n        realized_vol : pd.DataFrame\n            realized vol eg. EURUSDH1M.close\n\n        field : str\n            the field of the data to use (default: 'close')\n\n        Returns\n        -------\n        pd.DataFrame of vrp (both lagged - VRPV & contemporanous - VRP)\n        "
        cal = Calendar()
        tenor_days = cal.get_business_days_tenor(tenor_label)
        if tenor_label == 'ON' and adj_ON_friday:
            implied_vol = self.adjust_implied_ON_fri_vol(implied_vol)
        implied_vol = implied_vol.copy(deep=True)
        implied_unaligned = implied_vol.copy(deep=True)
        cols_to_change = implied_vol.columns.values
        new_cols = []
        for i in range(0, len(cols_to_change)):
            temp_col = list(cols_to_change[i])
            temp_col[6] = 'U'
            new_cols.append(''.join(temp_col))
        implied_vol.columns = new_cols
        implied_vol.index = [pd.Timestamp(x) + pd.tseries.offsets.BDay(tenor_days) for x in implied_vol.index]
        vrp = implied_vol.join(realized_vol, how='outer')
        vrp[asset + 'VRP' + tenor_label + '.close'] = vrp[asset + 'U' + tenor_label + '.' + field] - vrp[asset + 'H' + tenor_label + '.' + field]
        vrp = vrp.join(implied_unaligned, how='outer')
        vrp[asset + 'VRPV' + tenor_label + '.close'] = vrp[asset + 'V' + tenor_label + '.' + field] - vrp[asset + 'H' + tenor_label + '.' + field]
        return vrp

    def calculate_implied_vol_addon(self, asset, implied_vol=None, tenor_label='ON', model_window=20, model_algo='weighted-median-model', field='close', adj_ON_friday=True):
        if False:
            return 10
        'Calculates the implied volatility add on for specific tenors. The implied volatility addon can be seen as\n        a proxy for the event weights of large scheduled events for that day, such as the US employment report.\n\n        If there are multiple large events in the same period covered by the option, then this approach is not going\n        to be able to disentangle this.\n\n        Parameters\n        ----------\n        asset : str\n            Asset to be traded (eg. EURUSD)\n\n        tenor: str\n            eg. ON\n\n        Returns\n        ------\n        Implied volatility addon\n        '
        part = 'V'
        if implied_vol is None:
            implied_vol = self._market_df[asset + 'V' + tenor_label + '.' + field]
        implied_vol = implied_vol.copy(deep=True)
        implied_vol = pd.DataFrame(implied_vol)
        if tenor_label == 'ON' and adj_ON_friday:
            implied_vol = self.adjust_implied_ON_fri_vol(implied_vol)
        implied_vol = implied_vol.dropna()
        if model_algo == 'weighted-median-model':
            vol_data_20D_avg = self._calculations.rolling_median(implied_vol, model_window)
            vol_data_10D_avg = self._calculations.rolling_median(implied_vol, model_window)
            vol_data_5D_avg = self._calculations.rolling_median(implied_vol, model_window)
            vol_data_avg = (vol_data_20D_avg + vol_data_10D_avg + vol_data_5D_avg) / 3
            vol_data_addon = implied_vol - vol_data_avg
        elif model_algo == 'weighted-mean-model':
            vol_data_20D_avg = self._calculations.rolling_average(implied_vol, model_window)
            vol_data_10D_avg = self._calculations.rolling_average(implied_vol, model_window)
            vol_data_5D_avg = self._calculations.rolling_average(implied_vol, model_window)
            vol_data_avg = (vol_data_20D_avg + vol_data_10D_avg + vol_data_5D_avg) / 3
            vol_data_addon = implied_vol - vol_data_avg
        vol_data_addon = pd.DataFrame(vol_data_addon)
        implied_vol = pd.DataFrame(implied_vol)
        new_cols = implied_vol.columns.values
        new_cols = [w.replace(part + tenor_label, 'ADD' + tenor_label) for w in new_cols]
        vol_data_addon.columns = new_cols
        return vol_data_addon

    def adjust_implied_ON_fri_vol(self, data_frame):
        if False:
            while True:
                i = 10
        cols_ON = [x for x in data_frame.columns if 'VON' in x]
        for c in cols_ON:
            data_frame[c][data_frame.index.dayofweek == 4] = data_frame[c][data_frame.index.dayofweek == 4] * math.sqrt(3)
        return data_frame