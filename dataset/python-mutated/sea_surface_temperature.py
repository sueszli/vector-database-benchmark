""" Time series of historical average sea surface temperatures.

License: free to use and redistribute (see `this table`_ for details).

Sourced from http://www.neracoos.org/erddap/tabledap/index.html (table *B01_sbe37_all*)

This module contains one pandas Dataframe: ``sea_surface_temperature``.

.. rubric:: ``sea_surface_temperature``

:bokeh-dataframe:`bokeh.sampledata.sea_surface_temperature.sea_surface_temperature`

.. bokeh-sampledata-xref:: sea_surface_temperature

.. _this table: http://www.neracoos.org/erddap/info/B01_sbe37_all/index.html
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
from ..util.sampledata import package_csv
__all__ = ('sea_surface_temperature',)

def _read_data() -> DataFrame:
    if False:
        return 10
    df = package_csv('sea_surface_temperature', 'sea_surface_temperature.csv.gz', parse_dates=True, index_col=0)
    df = df.rename(columns={'temperature (celsius)': 'temperature'})
    df.index.name = 'time'
    return df
sea_surface_temperature = _read_data()