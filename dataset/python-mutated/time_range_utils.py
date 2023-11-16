from __future__ import annotations
from datetime import datetime
from typing import Any, cast
from superset import app
from superset.common.query_object import QueryObject
from superset.utils.core import FilterOperator, get_xaxis_label
from superset.utils.date_parser import get_since_until

def get_since_until_from_time_range(time_range: str | None=None, time_shift: str | None=None, extras: dict[str, Any] | None=None) -> tuple[datetime | None, datetime | None]:
    if False:
        i = 10
        return i + 15
    return get_since_until(relative_start=(extras or {}).get('relative_start', app.config['DEFAULT_RELATIVE_START_TIME']), relative_end=(extras or {}).get('relative_end', app.config['DEFAULT_RELATIVE_END_TIME']), time_range=time_range, time_shift=time_shift)

def get_since_until_from_query_object(query_object: QueryObject) -> tuple[datetime | None, datetime | None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    this function will return since and until by tuple if\n    1) the time_range is in the query object.\n    2) the xaxis column is in the columns field\n       and its corresponding `temporal_range` filter is in the adhoc filters.\n    :param query_object: a valid query object\n    :return: since and until by tuple\n    '
    if query_object.time_range:
        return get_since_until_from_time_range(time_range=query_object.time_range, time_shift=query_object.time_shift, extras=query_object.extras)
    time_range = None
    for flt in query_object.filter:
        if flt.get('op') == FilterOperator.TEMPORAL_RANGE.value and flt.get('col') == get_xaxis_label(query_object.columns) and isinstance(flt.get('val'), str):
            time_range = cast(str, flt.get('val'))
    return get_since_until_from_time_range(time_range=time_range, time_shift=query_object.time_shift, extras=query_object.extras)