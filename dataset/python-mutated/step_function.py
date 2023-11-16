from __future__ import annotations
import json
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from deprecated import deprecated
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning, AirflowSkipException
from airflow.providers.amazon.aws.hooks.step_function import StepFunctionHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class StepFunctionExecutionSensor(BaseSensorOperator):
    """
    Poll the Step Function State Machine Execution until it reaches a terminal state; fails if the task fails.

    On successful completion of the Execution the Sensor will do an XCom Push
    of the State Machine's output to `output`

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:StepFunctionExecutionSensor`

    :param execution_arn: execution_arn to check the state of
    :param aws_conn_id: aws connection to use, defaults to 'aws_default'
    """
    INTERMEDIATE_STATES = ('RUNNING',)
    FAILURE_STATES = ('FAILED', 'TIMED_OUT', 'ABORTED')
    SUCCESS_STATES = ('SUCCEEDED',)
    template_fields: Sequence[str] = ('execution_arn',)
    template_ext: Sequence[str] = ()
    ui_color = '#66c3ff'

    def __init__(self, *, execution_arn: str, aws_conn_id: str='aws_default', region_name: str | None=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.execution_arn = execution_arn
        self.aws_conn_id = aws_conn_id
        self.region_name = region_name

    def poke(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        execution_status = self.hook.describe_execution(self.execution_arn)
        state = execution_status['status']
        output = json.loads(execution_status['output']) if 'output' in execution_status else None
        if state in self.FAILURE_STATES:
            message = f'Step Function sensor failed. State Machine Output: {output}'
            if self.soft_fail:
                raise AirflowSkipException(message)
            raise AirflowException(message)
        if state in self.INTERMEDIATE_STATES:
            return False
        self.log.info('Doing xcom_push of output')
        self.xcom_push(context, 'output', output)
        return True

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> StepFunctionHook:
        if False:
            while True:
                i = 10
        'Create and return a StepFunctionHook.'
        return self.hook

    @cached_property
    def hook(self) -> StepFunctionHook:
        if False:
            print('Hello World!')
        return StepFunctionHook(aws_conn_id=self.aws_conn_id, region_name=self.region_name)