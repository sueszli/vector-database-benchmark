from enum import Enum

class DestinationType(Enum):
    BIGQUERY = 'bigquery'
    CLICKHOUSE = 'clickhouse'
    MSSQL = 'mssql'
    MYSQL = 'mysql'
    ORACLE = 'oracle'
    POSTGRES = 'postgres'
    REDSHIFT = 'redshift'
    SNOWFLAKE = 'snowflake'
    TIDB = 'tidb'
    DUCKDB = 'duckdb'

    @classmethod
    def from_string(cls, string_value: str) -> 'DestinationType':
        if False:
            for i in range(10):
                print('nop')
        return DestinationType[string_value.upper()]

    @staticmethod
    def testable_destinations():
        if False:
            i = 10
            return i + 15
        return [dest for dest in list(DestinationType) if dest != DestinationType.DUCKDB]