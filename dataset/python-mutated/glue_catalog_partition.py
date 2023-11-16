from __future__ import annotations
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Sequence
from deprecated import deprecated
from airflow.configuration import conf
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning, AirflowSkipException
from airflow.providers.amazon.aws.hooks.glue_catalog import GlueCatalogHook
from airflow.providers.amazon.aws.triggers.glue import GlueCatalogPartitionTrigger
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class GlueCatalogPartitionSensor(BaseSensorOperator):
    """
    Waits for a partition to show up in AWS Glue Catalog.

    :param table_name: The name of the table to wait for, supports the dot
        notation (my_database.my_table)
    :param expression: The partition clause to wait for. This is passed as
        is to the AWS Glue Catalog API's get_partitions function,
        and supports SQL like notation as in ``ds='2015-01-01'
        AND type='value'`` and comparison operators as in ``"ds>=2015-01-01"``.
        See https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api-catalog-partitions.html
        #aws-glue-api-catalog-partitions-GetPartitions
    :param aws_conn_id: ID of the Airflow connection where
        credentials and extra configuration are stored
    :param region_name: Optional aws region name (example: us-east-1). Uses region from connection
        if not specified.
    :param database_name: The name of the catalog database where the partitions reside.
    :param poke_interval: Time in seconds that the job should wait in
        between each tries
    :param deferrable: If true, then the sensor will wait asynchronously for the partition to
        show up in the AWS Glue Catalog.
        (default: False, but can be overridden in config file by setting default_deferrable to True)
    """
    template_fields: Sequence[str] = ('database_name', 'table_name', 'expression')
    ui_color = '#C5CAE9'

    def __init__(self, *, table_name: str, expression: str="ds='{{ ds }}'", aws_conn_id: str='aws_default', region_name: str | None=None, database_name: str='default', poke_interval: int=60 * 3, deferrable: bool=conf.getboolean('operators', 'default_deferrable', fallback=False), **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(poke_interval=poke_interval, **kwargs)
        self.aws_conn_id = aws_conn_id
        self.region_name = region_name
        self.table_name = table_name
        self.expression = expression
        self.database_name = database_name
        self.deferrable = deferrable

    def execute(self, context: Context) -> Any:
        if False:
            return 10
        if self.deferrable:
            self.defer(trigger=GlueCatalogPartitionTrigger(database_name=self.database_name, table_name=self.table_name, expression=self.expression, aws_conn_id=self.aws_conn_id, waiter_delay=int(self.poke_interval)), method_name='execute_complete', timeout=timedelta(seconds=self.timeout))
        else:
            super().execute(context=context)

    def poke(self, context: Context):
        if False:
            while True:
                i = 10
        'Check for existence of the partition in the AWS Glue Catalog table.'
        if '.' in self.table_name:
            (self.database_name, self.table_name) = self.table_name.split('.')
        self.log.info('Poking for table %s. %s, expression %s', self.database_name, self.table_name, self.expression)
        return self.hook.check_for_partition(self.database_name, self.table_name, self.expression)

    def execute_complete(self, context: Context, event: dict | None=None) -> None:
        if False:
            i = 10
            return i + 15
        if event is None or event['status'] != 'success':
            message = f'Trigger error: event is {event}'
            if self.soft_fail:
                raise AirflowSkipException(message)
            raise AirflowException(message)
        self.log.info('Partition exists in the Glue Catalog')

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> GlueCatalogHook:
        if False:
            i = 10
            return i + 15
        'Get the GlueCatalogHook.'
        return self.hook

    @cached_property
    def hook(self) -> GlueCatalogHook:
        if False:
            i = 10
            return i + 15
        return GlueCatalogHook(aws_conn_id=self.aws_conn_id, region_name=self.region_name)