""" Provide (optional) Pandas properties.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any
from .bases import Property
if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core.groupby.groupby import GroupBy
__all__ = ('PandasDataFrame', 'PandasGroupBy')

class PandasDataFrame(Property['DataFrame']):
    """ Accept Pandas DataFrame values.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    """

    def validate(self, value: Any, detail: bool=True) -> None:
        if False:
            while True:
                i = 10
        super().validate(value, detail)
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return
        msg = '' if not detail else f'expected Pandas DataFrame, got {value!r}'
        raise ValueError(msg)

class PandasGroupBy(Property['GroupBy[Any]']):
    """ Accept Pandas DataFrame values.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    """

    def validate(self, value: Any, detail: bool=True) -> None:
        if False:
            print('Hello World!')
        super().validate(value, detail)
        import pandas as pd
        if isinstance(value, pd.core.groupby.GroupBy):
            return
        msg = '' if not detail else f'expected Pandas GroupBy, got {value!r}'
        raise ValueError(msg)