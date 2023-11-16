"""Interact with AWS DataSync, using the AWS ``boto3`` library."""
from __future__ import annotations
import time
from urllib.parse import urlsplit
from airflow.exceptions import AirflowBadRequest, AirflowException, AirflowTaskTimeout
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook

class DataSyncHook(AwsBaseHook):
    """
    Interact with AWS DataSync.

    Provide thick wrapper around :external+boto3:py:class:`boto3.client("datasync") <DataSync.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    :param wait_interval_seconds: Time to wait between two
        consecutive calls to check TaskExecution status. Defaults to 30 seconds.
    :raises ValueError: If wait_interval_seconds is not between 0 and 15*60 seconds.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
    """
    TASK_EXECUTION_INTERMEDIATE_STATES = ('INITIALIZING', 'QUEUED', 'LAUNCHING', 'PREPARING', 'TRANSFERRING', 'VERIFYING')
    TASK_EXECUTION_FAILURE_STATES = ('ERROR',)
    TASK_EXECUTION_SUCCESS_STATES = ('SUCCESS',)

    def __init__(self, wait_interval_seconds: int=30, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, client_type='datasync', **kwargs)
        self.locations: list = []
        self.tasks: list = []
        if 0 <= wait_interval_seconds <= 15 * 60:
            self.wait_interval_seconds = wait_interval_seconds
        else:
            raise ValueError(f'Invalid wait_interval_seconds {wait_interval_seconds}')

    def create_location(self, location_uri: str, **create_location_kwargs) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Create a new location.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.create_location_s3`\n            - :external+boto3:py:meth:`DataSync.Client.create_location_smb`\n            - :external+boto3:py:meth:`DataSync.Client.create_location_nfs`\n            - :external+boto3:py:meth:`DataSync.Client.create_location_efs`\n\n        :param location_uri: Location URI used to determine the location type (S3, SMB, NFS, EFS).\n        :param create_location_kwargs: Passed to ``DataSync.Client.create_location_*`` methods.\n        :return: LocationArn of the created Location.\n        :raises AirflowException: If location type (prefix from ``location_uri``) is invalid.\n        '
        schema = urlsplit(location_uri).scheme
        if schema == 'smb':
            location = self.get_conn().create_location_smb(**create_location_kwargs)
        elif schema == 's3':
            location = self.get_conn().create_location_s3(**create_location_kwargs)
        elif schema == 'nfs':
            location = self.get_conn().create_location_nfs(**create_location_kwargs)
        elif schema == 'efs':
            location = self.get_conn().create_location_efs(**create_location_kwargs)
        else:
            raise AirflowException(f'Invalid/Unsupported location type: {schema}')
        self._refresh_locations()
        return location['LocationArn']

    def get_location_arns(self, location_uri: str, case_sensitive: bool=False, ignore_trailing_slash: bool=True) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all LocationArns which match a LocationUri.\n\n        :param location_uri: Location URI to search for, eg ``s3://mybucket/mypath``\n        :param case_sensitive: Do a case sensitive search for location URI.\n        :param ignore_trailing_slash: Ignore / at the end of URI when matching.\n        :return: List of LocationArns.\n        :raises AirflowBadRequest: if ``location_uri`` is empty\n        '
        if not location_uri:
            raise AirflowBadRequest('location_uri not specified')
        if not self.locations:
            self._refresh_locations()
        result = []
        if not case_sensitive:
            location_uri = location_uri.lower()
        if ignore_trailing_slash and location_uri.endswith('/'):
            location_uri = location_uri[:-1]
        for location_from_aws in self.locations:
            location_uri_from_aws = location_from_aws['LocationUri']
            if not case_sensitive:
                location_uri_from_aws = location_uri_from_aws.lower()
            if ignore_trailing_slash and location_uri_from_aws.endswith('/'):
                location_uri_from_aws = location_uri_from_aws[:-1]
            if location_uri == location_uri_from_aws:
                result.append(location_from_aws['LocationArn'])
        return result

    def _refresh_locations(self) -> None:
        if False:
            while True:
                i = 10
        'Refresh the local list of Locations.'
        locations = self.get_conn().list_locations()
        self.locations = locations['Locations']
        while 'NextToken' in locations:
            locations = self.get_conn().list_locations(NextToken=locations['NextToken'])
            self.locations.extend(locations['Locations'])

    def create_task(self, source_location_arn: str, destination_location_arn: str, **create_task_kwargs) -> str:
        if False:
            print('Hello World!')
        'Create a Task between the specified source and destination LocationArns.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.create_task`\n\n        :param source_location_arn: Source LocationArn. Must exist already.\n        :param destination_location_arn: Destination LocationArn. Must exist already.\n        :param create_task_kwargs: Passed to ``boto.create_task()``. See AWS boto3 datasync documentation.\n        :return: TaskArn of the created Task\n        '
        task = self.get_conn().create_task(SourceLocationArn=source_location_arn, DestinationLocationArn=destination_location_arn, **create_task_kwargs)
        self._refresh_tasks()
        return task['TaskArn']

    def update_task(self, task_arn: str, **update_task_kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update a Task.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.update_task`\n\n        :param task_arn: The TaskArn to update.\n        :param update_task_kwargs: Passed to ``boto.update_task()``, See AWS boto3 datasync documentation.\n        '
        self.get_conn().update_task(TaskArn=task_arn, **update_task_kwargs)

    def delete_task(self, task_arn: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete a Task.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.delete_task`\n\n        :param task_arn: The TaskArn to delete.\n        '
        self.get_conn().delete_task(TaskArn=task_arn)

    def _refresh_tasks(self) -> None:
        if False:
            print('Hello World!')
        'Refresh the local list of Tasks.'
        tasks = self.get_conn().list_tasks()
        self.tasks = tasks['Tasks']
        while 'NextToken' in tasks:
            tasks = self.get_conn().list_tasks(NextToken=tasks['NextToken'])
            self.tasks.extend(tasks['Tasks'])

    def get_task_arns_for_location_arns(self, source_location_arns: list, destination_location_arns: list) -> list:
        if False:
            i = 10
            return i + 15
        '\n        Return list of TaskArns which use both a specified source and destination LocationArns.\n\n        :param source_location_arns: List of source LocationArns.\n        :param destination_location_arns: List of destination LocationArns.\n        :raises AirflowBadRequest: if ``source_location_arns`` or ``destination_location_arns`` are empty.\n        '
        if not source_location_arns:
            raise AirflowBadRequest('source_location_arns not specified')
        if not destination_location_arns:
            raise AirflowBadRequest('destination_location_arns not specified')
        if not self.tasks:
            self._refresh_tasks()
        result = []
        for task in self.tasks:
            task_arn = task['TaskArn']
            task_description = self.get_task_description(task_arn)
            if task_description['SourceLocationArn'] in source_location_arns:
                if task_description['DestinationLocationArn'] in destination_location_arns:
                    result.append(task_arn)
        return result

    def start_task_execution(self, task_arn: str, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Start a TaskExecution for the specified task_arn.\n\n        Each task can have at most one TaskExecution.\n        Additional keyword arguments send to ``start_task_execution`` boto3 method.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.start_task_execution`\n\n        :param task_arn: TaskArn\n        :return: TaskExecutionArn\n        :raises ClientError: If a TaskExecution is already busy running for this ``task_arn``.\n        :raises AirflowBadRequest: If ``task_arn`` is empty.\n        '
        if not task_arn:
            raise AirflowBadRequest('task_arn not specified')
        task_execution = self.get_conn().start_task_execution(TaskArn=task_arn, **kwargs)
        return task_execution['TaskExecutionArn']

    def cancel_task_execution(self, task_execution_arn: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Cancel a TaskExecution for the specified ``task_execution_arn``.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.cancel_task_execution`\n\n        :param task_execution_arn: TaskExecutionArn.\n        :raises AirflowBadRequest: If ``task_execution_arn`` is empty.\n        '
        if not task_execution_arn:
            raise AirflowBadRequest('task_execution_arn not specified')
        self.get_conn().cancel_task_execution(TaskExecutionArn=task_execution_arn)

    def get_task_description(self, task_arn: str) -> dict:
        if False:
            return 10
        '\n        Get description for the specified ``task_arn``.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.describe_task`\n\n        :param task_arn: TaskArn\n        :return: AWS metadata about a task.\n        :raises AirflowBadRequest: If ``task_arn`` is empty.\n        '
        if not task_arn:
            raise AirflowBadRequest('task_arn not specified')
        return self.get_conn().describe_task(TaskArn=task_arn)

    def describe_task_execution(self, task_execution_arn: str) -> dict:
        if False:
            while True:
                i = 10
        '\n        Get description for the specified ``task_execution_arn``.\n\n        .. seealso::\n            - :external+boto3:py:meth:`DataSync.Client.describe_task_execution`\n\n        :param task_execution_arn: TaskExecutionArn\n        :return: AWS metadata about a task execution.\n        :raises AirflowBadRequest: If ``task_execution_arn`` is empty.\n        '
        return self.get_conn().describe_task_execution(TaskExecutionArn=task_execution_arn)

    def get_current_task_execution_arn(self, task_arn: str) -> str | None:
        if False:
            print('Hello World!')
        '\n        Get current TaskExecutionArn (if one exists) for the specified ``task_arn``.\n\n        :param task_arn: TaskArn\n        :return: CurrentTaskExecutionArn for this ``task_arn`` or None.\n        :raises AirflowBadRequest: if ``task_arn`` is empty.\n        '
        if not task_arn:
            raise AirflowBadRequest('task_arn not specified')
        task_description = self.get_task_description(task_arn)
        if 'CurrentTaskExecutionArn' in task_description:
            return task_description['CurrentTaskExecutionArn']
        return None

    def wait_for_task_execution(self, task_execution_arn: str, max_iterations: int=60) -> bool:
        if False:
            while True:
                i = 10
        '\n        Wait for Task Execution status to be complete (SUCCESS/ERROR).\n\n        The ``task_execution_arn`` must exist, or a boto3 ClientError will be raised.\n\n        :param task_execution_arn: TaskExecutionArn\n        :param max_iterations: Maximum number of iterations before timing out.\n        :return: Result of task execution.\n        :raises AirflowTaskTimeout: If maximum iterations is exceeded.\n        :raises AirflowBadRequest: If ``task_execution_arn`` is empty.\n        '
        if not task_execution_arn:
            raise AirflowBadRequest('task_execution_arn not specified')
        for _ in range(max_iterations):
            task_execution = self.get_conn().describe_task_execution(TaskExecutionArn=task_execution_arn)
            status = task_execution['Status']
            self.log.info('status=%s', status)
            if status in self.TASK_EXECUTION_SUCCESS_STATES:
                return True
            elif status in self.TASK_EXECUTION_FAILURE_STATES:
                return False
            elif status is None or status in self.TASK_EXECUTION_INTERMEDIATE_STATES:
                time.sleep(self.wait_interval_seconds)
            else:
                raise AirflowException(f'Unknown status: {status}')
            time.sleep(self.wait_interval_seconds)
        else:
            raise AirflowTaskTimeout('Max iterations exceeded!')