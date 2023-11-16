""" Correlations in mineral content for forensic glass samples.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.forensic_glass.data`

.. bokeh-sampledata-xref:: forensic_glass
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
        while True:
            i = 10
    data = package_csv('forensic_glass', 'forensic_glass.csv')
    return data
data = _read_data()