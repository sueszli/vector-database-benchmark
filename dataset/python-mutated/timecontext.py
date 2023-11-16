from __future__ import annotations
from typing import TYPE_CHECKING
import pyspark.sql.functions as F
import ibis.common.exceptions as com
from ibis.backends.base.df.timecontext import TimeContext, get_time_col
if TYPE_CHECKING:
    from pyspark.sql.dataframe import DataFrame

def filter_by_time_context(df: DataFrame, timecontext: TimeContext | None, adjusted_timecontext: TimeContext | None=None) -> DataFrame:
    if False:
        print('Hello World!')
    'Filter a Dataframe by given time context.'
    if not timecontext or (timecontext and adjusted_timecontext and (timecontext == adjusted_timecontext)):
        return df
    time_col = get_time_col()
    if time_col in df.columns:
        (begin, end) = timecontext
        return df.filter((F.col(time_col) >= begin.to_pydatetime()) & (F.col(time_col) < end.to_pydatetime()))
    else:
        raise com.TranslationError(f"'time' column missing in Dataframe {df}.To use time context, a Timestamp column name 'time' mustpresent in the table. ")

def combine_time_context(timecontexts: list[TimeContext]) -> TimeContext | None:
    if False:
        for i in range(10):
            print('nop')
    'Return a combined time context of `timecontexts`.\n\n    The combined time context starts from the earliest begin time\n    of `timecontexts`, and ends with the latest end time of `timecontexts`\n    The motivation is to generate a time context that is a superset\n    to all time contexts.\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> timecontexts = [\n    ...     (pd.Timestamp("20200102"), pd.Timestamp("20200103")),\n    ...     (pd.Timestamp("20200101"), pd.Timestamp("20200106")),\n    ...     (pd.Timestamp("20200109"), pd.Timestamp("20200110")),\n    ... ]\n    >>> combine_time_context(timecontexts)\n    (Timestamp(...), Timestamp(...))\n    >>> timecontexts = [None]\n    >>> print(combine_time_context(timecontexts))\n    None\n    '
    begin = min((t[0] for t in timecontexts if t), default=None)
    end = max((t[1] for t in timecontexts if t), default=None)
    if begin and end:
        return (begin, end)
    return None