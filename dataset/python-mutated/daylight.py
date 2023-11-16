""" Provide 2013 Warsaw daylight hours.

License: free to use and redistribute (see `this FAQ`_ for details).

Sourced from http://www.sunrisesunset.com

This module contains one pandas Dataframe: ``daylight_warsaw_2013``.

.. rubric:: ``daylight_warsaw_2013``

:bokeh-dataframe:`bokeh.sampledata.daylight.daylight_warsaw_2013`

.. bokeh-sampledata-xref:: daylight

.. _this FAQ: https://www.sunrisesunset.com/faqs.asp#other_usage
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
import pandas as pd
from ..util.sampledata import package_csv
__all__ = ('daylight_warsaw_2013',)

def _read_data() -> DataFrame:
    if False:
        print('Hello World!')
    '\n\n    '
    df = package_csv('daylight', 'daylight_warsaw_2013.csv', parse_dates=False)
    df['Date'] = pd.to_datetime(df.Date).map(lambda x: pd.to_datetime(x).date())
    df['Sunrise'] = pd.to_datetime(df.Sunrise, format='%H:%M:%S').map(lambda x: x.time())
    df['Sunset'] = pd.to_datetime(df.Sunset, format='%H:%M:%S').map(lambda x: x.time())
    return df
daylight_warsaw_2013 = _read_data()