""" Time series of commits for a GitHub user between 2012 and 2016.

License: `Public Domain`_

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.commits.data`

.. bokeh-sampledata-xref:: commits
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, cast
from ..util.sampledata import package_csv
if TYPE_CHECKING:
    import pandas as pd
__all__ = ('data',)

def _read_data() -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n\n    '
    import pandas as pd
    data = package_csv('commits', 'commits.txt.gz', parse_dates=True, header=None, names=['day', 'datetime'], index_col='datetime')
    data.index = cast(Any, pd.to_datetime(data.index, utc=True).tz_convert('US/Central'))
    data['time'] = data.index.time
    return data
data = _read_data()