__author__ = 'saeedamen'
import numpy
import pandas
from findatapy.timeseries import Calculations, Filter
from findatapy.util.commonman import CommonMan
from findatapy.util.configmanager import ConfigManager
from findatapy.util.loggermanager import LoggerManager

class Seasonality(object):
    """Does simple seasonality _calculations on data.

    """

    def __init__(self):
        if False:
            return 10
        self.config = ConfigManager()
        self.logger = LoggerManager().getLogger(__name__)
        return

    def time_of_day_seasonality(self, data_frame, years=False, seconds=False):
        if False:
            return 10
        calculations = Calculations()
        if years is False:
            if seconds:
                return calculations.average_by_hour_min_sec_of_day_pretty_output(data_frame)
            else:
                return calculations.average_by_hour_min_of_day_pretty_output(data_frame)
        set_year = set(data_frame.index.year)
        year = sorted(list(set_year))
        intraday_seasonality = None
        commonman = CommonMan()
        for i in year:
            if seconds:
                temp_seasonality = calculations.average_by_hour_min_sec_of_day_pretty_output(data_frame[data_frame.index.year == i])
            else:
                temp_seasonality = calculations.average_by_hour_min_of_day_pretty_output(data_frame[data_frame.index.year == i])
            temp_seasonality.columns = commonman.postfix_list(temp_seasonality.columns.values, ' ' + str(i))
            if intraday_seasonality is None:
                intraday_seasonality = temp_seasonality
            else:
                intraday_seasonality = intraday_seasonality.join(temp_seasonality)
        return intraday_seasonality

    def bus_day_of_month_seasonality_from_prices(self, data_frame, month_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], cum=True, cal='FX', partition_by_month=True, add_average=False, resample_freq='B'):
        if False:
            return 10
        return self.bus_day_of_month_seasonality(self, data_frame, month_list=month_list, cum=cum, cal=cal, partition_by_month=partition_by_month, add_average=add_average, price_index=True, resample_freq=resample_freq)

    def bus_day_of_month_seasonality(self, data_frame, month_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], cum=True, cal='FX', partition_by_month=True, add_average=False, price_index=False, resample_freq='B'):
        if False:
            i = 10
            return i + 15
        calculations = Calculations()
        filter = Filter()
        if price_index:
            data_frame = data_frame.resample(resample_freq).mean()
            data_frame = calculations.calculate_returns(data_frame)
        data_frame.index = pandas.to_datetime(data_frame.index)
        data_frame = filter.filter_time_series_by_holidays(data_frame, cal)
        if resample_freq == 'B':
            monthly_seasonality = calculations.average_by_month_day_by_bus_day(data_frame, cal)
        elif resample_freq == 'D':
            monthly_seasonality = calculations.average_by_month_day_by_day(data_frame)
        monthly_seasonality = monthly_seasonality.loc[month_list]
        if partition_by_month:
            monthly_seasonality = monthly_seasonality.unstack(level=0)
            if add_average:
                monthly_seasonality['Avg'] = monthly_seasonality.mean(axis=1)
        if cum is True:
            if partition_by_month:
                monthly_seasonality.loc[0] = numpy.zeros(len(monthly_seasonality.columns))
                monthly_seasonality = monthly_seasonality.sort_index()
            monthly_seasonality = calculations.create_mult_index(monthly_seasonality)
        return monthly_seasonality

    def monthly_seasonality_from_prices(self, data_frame, cum=True, add_average=False):
        if False:
            while True:
                i = 10
        return self.monthly_seasonality(data_frame, cum, add_average, price_index=True)

    def monthly_seasonality(self, data_frame, cum=True, add_average=False, price_index=False):
        if False:
            return 10
        calculations = Calculations()
        if price_index:
            data_frame = data_frame.resample('BM').mean()
            data_frame = calculations.calculate_returns(data_frame)
        data_frame.index = pandas.to_datetime(data_frame.index)
        monthly_seasonality = calculations.average_by_month(data_frame)
        if add_average:
            monthly_seasonality['Avg'] = monthly_seasonality.mean(axis=1)
        if cum is True:
            monthly_seasonality.loc[0] = numpy.zeros(len(monthly_seasonality.columns))
            monthly_seasonality = monthly_seasonality.sort_index()
            monthly_seasonality = calculations.create_mult_index(monthly_seasonality)
        return monthly_seasonality

    def adjust_rolling_seasonality(self, data_frame, window=None, likely_period=None):
        if False:
            for i in range(10):
                print('nop')
        'Adjusted time series which exhibit strong seasonality. If time series do not exhibit any seasonality will return\n        NaN values.\n\n        Parameters\n        ----------\n        data_frame : DataFrame\n            Data to be seasonally adjusted\n        window : int\n            Number of points to use in our rolling window (eg. for monthly data 60 = 5 year * 12 months)\n        likely_period\n            What is the likely period of the data (eg. for monthly data, likely to be 12, for daily, likely to be 252)\n\n        Returns\n        -------\n        DataFrame\n            Time series seasonally adjusted\n        '
        data_frame = data_frame.copy()
        for c in data_frame.columns:
            data_frame[c] = data_frame[c].rolling(center=False, window=window, min_periods=window).apply(lambda x: self._remove_seasonality(x, likely_period=likely_period))
        return data_frame

    def _remove_seasonality(self, series, likely_period=None):
        if False:
            return 10
        from seasonal import fit_seasons, adjust_seasons
        (seasons, trend) = fit_seasons(series, period=likely_period)
        adjusted = adjust_seasons(series, seasons=seasons, period=likely_period)
        if adjusted is None:
            return numpy.nan
        return adjusted[-1]
if __name__ == '__main__':
    pass