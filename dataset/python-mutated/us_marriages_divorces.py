""" Provide U.S. marriage and divorce statistics between 1867 and 2014

License: `Public Domain`_

Sourced from http://www.cdc.gov/nchs/

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.us_marriages_divorces.data`

.. bokeh-sampledata-xref:: us_marriages_divorces
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
from ..util.sampledata import package_csv
__all__ = ('data',)

def _read_data() -> DataFrame:
    if False:
        while True:
            i = 10
    '\n\n    '
    data = package_csv('us_marriages_divorces', 'us_marriages_divorces.csv')
    return data.interpolate(method='linear', axis=0).ffill().bfill()
data = _read_data()