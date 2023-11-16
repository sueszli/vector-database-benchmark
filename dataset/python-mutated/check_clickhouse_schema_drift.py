from typing import Dict, List, Tuple
import structlog
from clickhouse_driver.errors import Error as ClickhouseError
from django.conf import settings
from statshog.defaults.django import statsd
from posthog.client import sync_execute
logger = structlog.get_logger(__name__)

def get_clickhouse_schema() -> List[Tuple[str, str, str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the ClickHouse schema of all tables that\n    are not materialized views (aka: .inner_id.%)\n    '
    return sync_execute("\n        SELECT\n            name as table_name,\n            create_table_query,\n            hostname() as hostname\n        FROM\n            clusterAllReplicas('{cluster}', system, tables)\n        WHERE\n            database == '{database}'\n        AND\n            table_name NOT LIKE '.inner_id.%'\n        ".format(cluster=settings.CLICKHOUSE_CLUSTER, database=settings.CLICKHOUSE_DATABASE))

def get_clickhouse_nodes() -> List[Tuple[str]]:
    if False:
        print('Hello World!')
    '\n    Get the ClickHouse nodes part of the cluster\n    '
    return sync_execute("\n        SELECT\n            host_name\n        FROM\n            system.clusters\n        WHERE\n            cluster == '{cluster}'\n\n        ".format(cluster=settings.CLICKHOUSE_CLUSTER))

def get_clickhouse_schema_drift(clickhouse_nodes: List[Tuple[str]], clickhouse_schema: List[Tuple[str, str, str]]) -> List:
    if False:
        while True:
            i = 10
    diff = []
    if len(clickhouse_nodes) <= 1:
        return diff
    tables = {}
    for (table_name, schema, node_name) in clickhouse_schema:
        if table_name not in tables:
            tables[table_name] = {}
        if schema not in tables[table_name]:
            tables[table_name][schema] = []
        tables[table_name][schema].append(node_name)
    for (table_name, table_schemas) in tables.items():
        if len(table_schemas) > 1:
            diff.append(table_name)
        schema_count = sum((len(v) for v in table_schemas.values()))
        if schema_count != len(clickhouse_nodes):
            diff.append(table_name)
    return diff

def check_clickhouse_schema_drift(clickhouse_nodes: List[Tuple[str]]=[], clickhouse_schema: List[Tuple[str, str, str]]=[]) -> None:
    if False:
        i = 10
        return i + 15
    try:
        if not clickhouse_nodes:
            clickhouse_nodes = get_clickhouse_nodes()
        if not clickhouse_schema:
            clickhouse_schema = get_clickhouse_schema()
    except ClickhouseError:
        logger.error('check_clickhouse_schema_drift_error', exc_info=True)
        return
    drift = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    logger.info('check_clickhouse_schema_drift', table_count=len(drift), tables=drift)
    for table_name in drift:
        statsd.gauge('clickhouse_schema_drift_table.{}'.format(table_name), 1)
    statsd.gauge('clickhouse_schema_drift_table_count', len(drift))