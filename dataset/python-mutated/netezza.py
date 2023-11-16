from superset.constants import TimeGrain
from superset.db_engine_specs.postgres import PostgresBaseEngineSpec

class NetezzaEngineSpec(PostgresBaseEngineSpec):
    engine = 'netezza'
    default_driver = 'nzpy'
    engine_name = 'IBM Netezza Performance Server'
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "DATE_TRUNC('second', {col})", TimeGrain.MINUTE: "DATE_TRUNC('minute', {col})", TimeGrain.HOUR: "DATE_TRUNC('hour', {col})", TimeGrain.DAY: "DATE_TRUNC('day', {col})", TimeGrain.WEEK: "DATE_TRUNC('week', {col})", TimeGrain.MONTH: "DATE_TRUNC('month', {col})", TimeGrain.QUARTER: "DATE_TRUNC('quarter', {col})", TimeGrain.YEAR: "DATE_TRUNC('year', {col})"}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return "(timestamp 'epoch' + {col} * interval '1 second')"