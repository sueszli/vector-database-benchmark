from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from deprecated import deprecated
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.amazon.aws.hooks.redshift_cluster import RedshiftHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class RedshiftClusterSensor(BaseSensorOperator):
    """
    Waits for a Redshift cluster to reach a specific status.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:RedshiftClusterSensor`

    :param cluster_identifier: The identifier for the cluster being pinged.
    :param target_status: The cluster status desired.
    """
    template_fields: Sequence[str] = ('cluster_identifier', 'target_status')

    def __init__(self, *, cluster_identifier: str, target_status: str='available', aws_conn_id: str='aws_default', **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.cluster_identifier = cluster_identifier
        self.target_status = target_status
        self.aws_conn_id = aws_conn_id

    def poke(self, context: Context):
        if False:
            print('Hello World!')
        current_status = self.hook.cluster_status(self.cluster_identifier)
        self.log.info("Poked cluster %s for status '%s', found status '%s'", self.cluster_identifier, self.target_status, current_status)
        return current_status == self.target_status

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> RedshiftHook:
        if False:
            for i in range(10):
                print('nop')
        'Create and return a RedshiftHook.'
        return self.hook

    @cached_property
    def hook(self) -> RedshiftHook:
        if False:
            i = 10
            return i + 15
        return RedshiftHook(aws_conn_id=self.aws_conn_id)