""" CO2 emmisions of selected countries in the years 2000 and 2010.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.emissions.data`

.. bokeh-sampledata-xref:: emissions
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
from ..util.sampledata import external_csv
if TYPE_CHECKING:
    import pandas as pd
__all__ = ('data',)

def _read_data() -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    data = external_csv('emissions', 'emissions.csv')
    return data
data = _read_data()