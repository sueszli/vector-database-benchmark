import datetime
import random
from unittest import TestCase
import pytest
import pandas as pd
import plotly.tools as tls
from plotly import optional_imports
matplotlylib = optional_imports.get_module('plotly.matplotlylib')
if matplotlylib:
    from matplotlib.dates import date2num
    import matplotlib.pyplot as plt

@pytest.mark.skip
class TestDateTimes(TestCase):

    def test_normal_mpl_dates(self):
        if False:
            print('Hello World!')
        datetime_format = '%Y-%m-%d %H:%M:%S'
        y = [1, 2, 3, 4]
        date_strings = ['2010-01-04 00:00:00', '2010-01-04 10:00:00', '2010-01-04 23:00:59', '2010-01-05 00:00:00']
        dates = [datetime.datetime.strptime(date_string, datetime_format) for date_string in date_strings]
        mpl_dates = date2num(dates)
        (fig, ax) = plt.subplots()
        ax.plot_date(mpl_dates, y)
        pfig = tls.mpl_to_plotly(fig)
        print(date_strings)
        print(pfig['data'][0]['x'])
        self.assertEqual(fig.axes[0].lines[0].get_xydata()[0][0], 733776.0)
        self.assertEqual(tuple(pfig['data'][0]['x']), tuple(date_strings))

    def test_pandas_time_series_date_formatter(self):
        if False:
            i = 10
            return i + 15
        ndays = 3
        x = pd.date_range('1/1/2001', periods=ndays, freq='D')
        y = [random.randint(0, 10) for i in range(ndays)]
        s = pd.DataFrame(y, columns=['a'])
        s['Date'] = x
        s.plot(x='Date')
        fig = plt.gcf()
        pfig = tls.mpl_to_plotly(fig)
        expected_x = ('2001-01-01 00:00:00', '2001-01-02 00:00:00', '2001-01-03 00:00:00')
        expected_x0 = 11323.0
        x0 = fig.axes[0].lines[0].get_xydata()[0][0]
        self.assertEqual(x0, expected_x0)
        self.assertEqual(pfig['data'][0]['x'], expected_x)