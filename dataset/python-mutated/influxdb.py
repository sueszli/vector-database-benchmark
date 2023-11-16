from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.influxdb.hooks.influxdb import InfluxDBHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class InfluxDBOperator(BaseOperator):
    """
    Executes sql code in a specific InfluxDB database.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:InfluxDBOperator`

    :param sql: the sql code to be executed. Can receive a str representing a
        sql statement
    :param influxdb_conn_id: Reference to :ref:`Influxdb connection id <howto/connection:influxdb>`.
    """
    template_fields: Sequence[str] = ('sql',)

    def __init__(self, *, sql: str, influxdb_conn_id: str='influxdb_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.influxdb_conn_id = influxdb_conn_id
        self.sql = sql

    def execute(self, context: Context) -> None:
        if False:
            return 10
        self.log.info('Executing: %s', self.sql)
        self.hook = InfluxDBHook(conn_id=self.influxdb_conn_id)
        self.hook.query(self.sql)