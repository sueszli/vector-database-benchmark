""" Mean daily temperatures in Lincoln, Nebraska, 2016.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.lincoln.data`

.. bokeh-sampledata-xref:: lincoln
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
from ..util.sampledata import package_csv
if TYPE_CHECKING:
    import pandas as pd
__all__ = ('data',)

def _read_data() -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    data = package_csv('lincoln', 'lincoln_weather.csv')
    return data
data = _read_data()