from superset.common.query_object_factory import QueryObjectFactory
from superset.constants import NO_TIME_RANGE

def test_process_time_range():
    if False:
        i = 10
        return i + 15
    '\n    correct empty time range\n    '
    assert QueryObjectFactory._process_time_range(None) == NO_TIME_RANGE
    '\n    Use the first temporal filter as time range\n    '
    filters = [{'col': 'dttm', 'op': 'TEMPORAL_RANGE', 'val': '2001 : 2002'}, {'col': 'dttm2', 'op': 'TEMPORAL_RANGE', 'val': '2002 : 2003'}]
    assert QueryObjectFactory._process_time_range(None, filters) == '2001 : 2002'
    '\n    Use the BASE_AXIS temporal filter as time range\n    '
    columns = [{'columnType': 'BASE_AXIS', 'label': 'dttm2', 'sqlExpression': 'dttm'}]
    assert QueryObjectFactory._process_time_range(None, filters, columns) == '2002 : 2003'