from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from deprecated import deprecated
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning, AirflowSkipException
from airflow.providers.amazon.aws.hooks.glue_crawler import GlueCrawlerHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class GlueCrawlerSensor(BaseSensorOperator):
    """
    Waits for an AWS Glue crawler to reach any of the statuses below.

    'FAILED', 'CANCELLED', 'SUCCEEDED'

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:GlueCrawlerSensor`

    :param crawler_name: The AWS Glue crawler unique name
    :param aws_conn_id: aws connection to use, defaults to 'aws_default'
    """
    template_fields: Sequence[str] = ('crawler_name',)

    def __init__(self, *, crawler_name: str, aws_conn_id: str='aws_default', **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.crawler_name = crawler_name
        self.aws_conn_id = aws_conn_id
        self.success_statuses = 'SUCCEEDED'
        self.errored_statuses = ('FAILED', 'CANCELLED')

    def poke(self, context: Context):
        if False:
            return 10
        self.log.info('Poking for AWS Glue crawler: %s', self.crawler_name)
        crawler_state = self.hook.get_crawler(self.crawler_name)['State']
        if crawler_state == 'READY':
            self.log.info('State: %s', crawler_state)
            crawler_status = self.hook.get_crawler(self.crawler_name)['LastCrawl']['Status']
            if crawler_status == self.success_statuses:
                self.log.info('Status: %s', crawler_status)
                return True
            else:
                message = f'Status: {crawler_status}'
                if self.soft_fail:
                    raise AirflowSkipException(message)
                raise AirflowException(message)
        else:
            return False

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> GlueCrawlerHook:
        if False:
            for i in range(10):
                print('nop')
        'Return a new or pre-existing GlueCrawlerHook.'
        return self.hook

    @cached_property
    def hook(self) -> GlueCrawlerHook:
        if False:
            while True:
                i = 10
        return GlueCrawlerHook(aws_conn_id=self.aws_conn_id)