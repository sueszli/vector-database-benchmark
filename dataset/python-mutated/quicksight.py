from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.providers.amazon.aws.hooks.quicksight import QuickSightHook
from airflow.providers.amazon.aws.hooks.sts import StsHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class QuickSightSensor(BaseSensorOperator):
    """
    Watches for the status of an Amazon QuickSight Ingestion.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:QuickSightSensor`

    :param data_set_id:  ID of the dataset used in the ingestion.
    :param ingestion_id: ID for the ingestion.
    :param aws_conn_id: The Airflow connection used for AWS credentials. (templated)
         If this is None or empty then the default boto3 behaviour is used. If
         running Airflow in a distributed manner and aws_conn_id is None or
         empty, then the default boto3 configuration would be used (and must be
         maintained on each worker node).
    """
    template_fields: Sequence[str] = ('data_set_id', 'ingestion_id', 'aws_conn_id')

    def __init__(self, *, data_set_id: str, ingestion_id: str, aws_conn_id: str='aws_default', **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.data_set_id = data_set_id
        self.ingestion_id = ingestion_id
        self.aws_conn_id = aws_conn_id
        self.success_status = 'COMPLETED'
        self.errored_statuses = ('FAILED', 'CANCELLED')

    def poke(self, context: Context) -> bool:
        if False:
            return 10
        '\n        Pokes until the QuickSight Ingestion has successfully finished.\n\n        :param context: The task context during execution.\n        :return: True if it COMPLETED and False if not.\n        '
        self.log.info('Poking for Amazon QuickSight Ingestion ID: %s', self.ingestion_id)
        aws_account_id = self.sts_hook.get_account_number()
        quicksight_ingestion_state = self.quicksight_hook.get_status(aws_account_id, self.data_set_id, self.ingestion_id)
        self.log.info('QuickSight Status: %s', quicksight_ingestion_state)
        if quicksight_ingestion_state in self.errored_statuses:
            error = self.quicksight_hook.get_error_info(aws_account_id, self.data_set_id, self.ingestion_id)
            message = f'The QuickSight Ingestion failed. Error info: {error}'
            if self.soft_fail:
                raise AirflowSkipException(message)
            raise AirflowException(message)
        return quicksight_ingestion_state == self.success_status

    @cached_property
    def quicksight_hook(self):
        if False:
            return 10
        return QuickSightHook(aws_conn_id=self.aws_conn_id)

    @cached_property
    def sts_hook(self):
        if False:
            for i in range(10):
                print('nop')
        return StsHook(aws_conn_id=self.aws_conn_id)