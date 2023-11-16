""" A version of the Auto MPG data set.

License: `CC0`_

Sourced from https://archive.ics.uci.edu/ml/datasets/auto+mpg

This module contains one pandas Dataframe: ``autompg``.

.. rubric:: ``autompg2``

:bokeh-dataframe:`bokeh.sampledata.autompg2.autompg2`

.. bokeh-sampledata-xref:: autompg2
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
from ..util.sampledata import package_csv
__all__ = ('autompg2',)

def _capitalize_words(string: str) -> str:
    if False:
        print('Hello World!')
    '\n\n    '
    return ' '.join((word.capitalize() for word in string.split(' ')))

def _read_data() -> DataFrame:
    if False:
        i = 10
        return i + 15
    '\n\n    '
    df = package_csv('autompg2', 'auto-mpg2.csv').copy()
    df['manufacturer'] = df['manufacturer'].map(_capitalize_words)
    df['model'] = df['model'].map(_capitalize_words)
    df['drv'] = df['drv'].replace({'f': 'front', 'r': 'rear', '4': '4x4'})
    return df
autompg2 = _read_data()