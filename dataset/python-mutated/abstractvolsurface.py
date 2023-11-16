__author__ = 'saeedamen'
import abc
import numpy as np
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})
from financepy.utils.date import Date

class AbstractVolSurface(ABC):
    """Holds data for an asset class vol surface

    """

    @abc.abstractmethod
    def build_vol_surface(self):
        if False:
            return 10
        return

    @abc.abstractmethod
    def extract_vol_surface(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def _extremes(self, min, max, data):
        if False:
            for i in range(10):
                print('nop')
        if min is None:
            min = np.min(data)
        else:
            new_min = np.min(data)
            if new_min < min:
                min = new_min
        if max is None:
            max = np.max(data)
        else:
            new_max = np.max(data)
            if new_max > max:
                max = new_max
        return (min, max)

    def extract_vol_surface_across_dates(self, dates, num_strike_intervals=60, vol_surface_type='vol_surface_strike_space', reverse_plot=True):
        if False:
            print('Hello World!')
        vol_surface_dict = {}
        min_x = None
        max_x = None
        min_z = None
        max_z = None
        for i in range(0, len(dates)):
            self.build_vol_surface(dates[i])
            df_vol_surface = self.extract_vol_surface(num_strike_intervals=num_strike_intervals)[vol_surface_type]
            if reverse_plot:
                vol_surface_dict[dates[i]] = df_vol_surface.iloc[:, ::-1]
            else:
                vol_surface_dict[dates[i]] = df_vol_surface
            (min_x, max_x) = self._extremes(min_x, max_x, df_vol_surface.index.values)
            (min_z, max_z) = self._extremes(min_z, max_z, df_vol_surface.values)
        extremes_dict = {'min_x': min_x, 'max_x': max_x, 'min_z': min_z, 'max_z': max_z}
        return (vol_surface_dict, extremes_dict)

    def _get_tenor_index(self, tenor):
        if False:
            while True:
                i = 10
        return self._tenors.index(tenor)

    def _get_tenor_expiry(self, tenor):
        if False:
            i = 10
            return i + 15
        return

    def _findate(self, date):
        if False:
            for i in range(10):
                print('nop')
        return Date(date.day, date.month, date.year)