from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.amazon.aws.hooks.base_aws import AwsGenericHook
from airflow.providers.amazon.aws.utils.waiter_with_logging import wait
if TYPE_CHECKING:
    from mypy_boto3_appflow.client import AppflowClient

class AppflowHook(AwsGenericHook['AppflowClient']):
    """
    Interact with Amazon AppFlow.

    Provide thin wrapper around :external+boto3:py:class:`boto3.client("appflow") <Appflow.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
        - `Amazon Appflow API Reference <https://docs.aws.amazon.com/appflow/1.0/APIReference/Welcome.html>`__
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        kwargs['client_type'] = 'appflow'
        super().__init__(*args, **kwargs)

    def run_flow(self, flow_name: str, poll_interval: int=20, wait_for_completion: bool=True, max_attempts: int=60) -> str:
        if False:
            return 10
        '\n        Execute an AppFlow run.\n\n        :param flow_name: The flow name\n        :param poll_interval: Time (seconds) to wait between two consecutive calls to check the run status\n        :param wait_for_completion: whether to wait for the run to end to return\n        :param max_attempts: the number of polls to do before timing out/returning a failure.\n        :return: The run execution ID\n        '
        response_start = self.conn.start_flow(flowName=flow_name)
        execution_id = response_start['executionId']
        self.log.info('executionId: %s', execution_id)
        if wait_for_completion:
            wait(waiter=self.get_waiter('run_complete', {'EXECUTION_ID': execution_id}), waiter_delay=poll_interval, waiter_max_attempts=max_attempts, args={'flowName': flow_name}, failure_message='error while waiting for flow to complete', status_message='waiting for flow completion, status', status_args=[f"flowExecutions[?executionId=='{execution_id}'].executionStatus", f"flowExecutions[?executionId=='{execution_id}'].executionResult.errorInfo"])
            self._log_execution_description(flow_name, execution_id)
        return execution_id

    def _log_execution_description(self, flow_name: str, execution_id: str):
        if False:
            for i in range(10):
                print('nop')
        response_desc = self.conn.describe_flow_execution_records(flowName=flow_name)
        last_execs = {fe['executionId']: fe for fe in response_desc['flowExecutions']}
        exec_details = last_execs[execution_id]
        self.log.info('Run complete, execution details: %s', exec_details)

    def update_flow_filter(self, flow_name: str, filter_tasks, set_trigger_ondemand: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the flow task filter; all filters will be removed if an empty array is passed to filter_tasks.\n\n        :param flow_name: The flow name\n        :param filter_tasks: List flow tasks to be added\n        :param set_trigger_ondemand: If True, set the trigger to on-demand; otherwise, keep the trigger as is\n        :return: None\n        '
        response = self.conn.describe_flow(flowName=flow_name)
        connector_type = response['sourceFlowConfig']['connectorType']
        tasks = []
        for task in response['tasks']:
            if task['taskType'] == 'Filter' and task.get('connectorOperator', {}).get(connector_type) != 'PROJECTION':
                self.log.info('Removing task: %s', task)
            else:
                tasks.append(task)
        tasks += filter_tasks
        if set_trigger_ondemand:
            del response['triggerConfig']['triggerProperties']
        self.conn.update_flow(flowName=response['flowName'], destinationFlowConfigList=response['destinationFlowConfigList'], sourceFlowConfig=response['sourceFlowConfig'], triggerConfig=response['triggerConfig'], description=response.get('description', 'Flow description.'), tasks=tasks)