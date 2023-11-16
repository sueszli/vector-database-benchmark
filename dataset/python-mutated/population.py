""" Historical and projected population data by age, gender, and country.

License: `CC BY 3.0 IGO`_

Sourced from: https://population.un.org/wpp/Download/Standard/Population/

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.population.data`

.. bokeh-sampledata-xref:: population
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
from ..util.sampledata import external_csv
__all__ = ('data',)

def _read_data() -> DataFrame:
    if False:
        return 10
    '\n\n    '
    df = external_csv('population', 'WPP2012_SA_DB03_POPULATION_QUINQUENNIAL.csv', encoding='CP1250')
    df = df[df.Sex != 'Both']
    df = df.drop(['VarID', 'Variant', 'MidPeriod', 'SexID', 'AgeGrpSpan'], axis=1)
    df = df.rename(columns={'Time': 'Year'})
    df.Value *= 1000
    return df
data = _read_data()