from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from airflow.providers.apache.hive.hooks.hive import HiveMetastoreHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class HivePartitionSensor(BaseSensorOperator):
    """
    Waits for a partition to show up in Hive.

    Note: Because ``partition`` supports general logical operators, it
    can be inefficient. Consider using NamedHivePartitionSensor instead if
    you don't need the full flexibility of HivePartitionSensor.

    :param table: The name of the table to wait for, supports the dot
        notation (my_database.my_table)
    :param partition: The partition clause to wait for. This is passed as
        is to the metastore Thrift client ``get_partitions_by_filter`` method,
        and apparently supports SQL like notation as in ``ds='2015-01-01'
        AND type='value'`` and comparison operators as in ``"ds>=2015-01-01"``
    :param metastore_conn_id: reference to the
        :ref: `metastore thrift service connection id <howto/connection:hive_metastore>`
    """
    template_fields: Sequence[str] = ('schema', 'table', 'partition')
    ui_color = '#C5CAE9'

    def __init__(self, *, table: str, partition: str | None="ds='{{ ds }}'", metastore_conn_id: str='metastore_default', schema: str='default', poke_interval: int=60 * 3, **kwargs: Any):
        if False:
            while True:
                i = 10
        super().__init__(poke_interval=poke_interval, **kwargs)
        if not partition:
            partition = "ds='{{ ds }}'"
        self.metastore_conn_id = metastore_conn_id
        self.table = table
        self.partition = partition
        self.schema = schema

    def poke(self, context: Context) -> bool:
        if False:
            print('Hello World!')
        if '.' in self.table:
            (self.schema, self.table) = self.table.split('.')
        self.log.info('Poking for table %s.%s, partition %s', self.schema, self.table, self.partition)
        if not hasattr(self, 'hook'):
            hook = HiveMetastoreHook(metastore_conn_id=self.metastore_conn_id)
        return hook.check_for_partition(self.schema, self.table, self.partition)