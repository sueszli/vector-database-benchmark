from unittest.mock import Mock, call, patch
from clickhouse_driver.errors import Error as ClickhouseError
from posthog.tasks.check_clickhouse_schema_drift import check_clickhouse_schema_drift, get_clickhouse_schema_drift

def test_get_clickhouse_schema_drift() -> None:
    if False:
        print('Hello World!')
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2'), ('table2', 'schema2', 'host3')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == []
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2'), ('table2', 'schema2-different-bis', 'host3')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2-different', 'host2'), ('table2', 'schema2-different-bis', 'host3')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1-different', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2'), ('table2', 'schema2-different', 'host3')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table1', 'table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1-different', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2'), ('table2', 'schema2-different', 'host3')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table1', 'table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table1', 'schema1', 'host3'), ('table2', 'schema2', 'host1')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table1', 'table2']
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table2', 'schema2', 'host1')]
    diff = get_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert diff == ['table1', 'table2']

@patch('posthog.clickhouse.client.connection.ch_pool.get_client')
def test_check_clickhouse_schema_drift_error_from_clickhouse(mock_ch: Mock) -> None:
    if False:
        while True:
            i = 10
    mock_ch.side_effect = ClickhouseError('Broken to connect')
    check_clickhouse_schema_drift()

@patch('statshog.defaults.django.statsd.gauge')
def test_check_clickhouse_schema_drift_without_drift(mock_statsd: Mock) -> None:
    if False:
        i = 10
        return i + 15
    clickhouse_nodes = [('node1',), ('node2',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table2', 'schema2', 'host1'), ('table1', 'schema1', 'host2'), ('table2', 'schema2', 'host2')]
    check_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert mock_statsd.call_args_list == [call('clickhouse_schema_drift_table_count', 0)]

@patch('statshog.defaults.django.statsd.gauge')
def test_check_clickhouse_schema_drift_with_drift(mock_statsd: Mock) -> None:
    if False:
        i = 10
        return i + 15
    clickhouse_nodes = [('node1',), ('node2',), ('node3',)]
    clickhouse_schema = [('table1', 'schema1', 'host1'), ('table1', 'schema1', 'host2'), ('table2', 'schema2', 'host1'), ('table2', 'schema2', 'host2')]
    check_clickhouse_schema_drift(clickhouse_nodes, clickhouse_schema)
    assert mock_statsd.call_args_list == [call('clickhouse_schema_drift_table.table1', 1), call('clickhouse_schema_drift_table.table2', 1), call('clickhouse_schema_drift_table_count', 2)]