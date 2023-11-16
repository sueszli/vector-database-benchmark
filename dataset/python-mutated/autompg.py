""" A version of the Auto MPG data set.

License: `CC0`_

Sourced from https://archive.ics.uci.edu/ml/datasets/auto+mpg

This module contains two pandas Dataframes: ``autompg`` and ``autompg_clean``.
The "clean" version has cleaned up the ``"mfr"`` and ``"origin"`` fields.

.. rubric:: ``autompg``

:bokeh-dataframe:`bokeh.sampledata.autompg.autompg`

.. rubric:: ``autompg_clean``

:bokeh-dataframe:`bokeh.sampledata.autompg.autompg_clean`

.. bokeh-sampledata-xref:: autompg
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
from ..util.sampledata import package_csv
if TYPE_CHECKING:
    import pandas as pd
__all__ = ('autompg', 'autompg_clean')

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        return 10
    '\n\n    '
    df = df.copy()
    df['mfr'] = [x.split()[0] for x in df.name]
    df.loc[df.mfr == 'chevy', 'mfr'] = 'chevrolet'
    df.loc[df.mfr == 'chevroelt', 'mfr'] = 'chevrolet'
    df.loc[df.mfr == 'maxda', 'mfr'] = 'mazda'
    df.loc[df.mfr == 'mercedes-benz', 'mfr'] = 'mercedes'
    df.loc[df.mfr == 'toyouta', 'mfr'] = 'toyota'
    df.loc[df.mfr == 'vokswagen', 'mfr'] = 'volkswagen'
    df.loc[df.mfr == 'vw', 'mfr'] = 'volkswagen'
    ORIGINS = ['North America', 'Europe', 'Asia']
    df.origin = [ORIGINS[x - 1] for x in df.origin]
    return df
autompg = package_csv('autompg', 'auto-mpg.csv')
autompg_clean = _clean_data(autompg)