"""
A client for AWS Batch services.

.. seealso::

    - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html
    - https://docs.aws.amazon.com/batch/latest/APIReference/Welcome.html
"""
from __future__ import annotations
import itertools
import random
import time
from typing import TYPE_CHECKING, Callable
import botocore.client
import botocore.exceptions
import botocore.waiter
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from airflow.typing_compat import Protocol, runtime_checkable
if TYPE_CHECKING:
    from airflow.providers.amazon.aws.utils.task_log_fetcher import AwsTaskLogFetcher

@runtime_checkable
class BatchProtocol(Protocol):
    """
    A structured Protocol for ``boto3.client('batch') -> botocore.client.Batch``.

    This is used for type hints on :py:meth:`.BatchClient.client`; it covers
    only the subset of client methods required.

    .. seealso::

        - https://mypy.readthedocs.io/en/latest/protocols.html
        - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html
    """

    def describe_jobs(self, jobs: list[str]) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get job descriptions from AWS Batch.\n\n        :param jobs: a list of JobId to describe\n\n        :return: an API response to describe jobs\n        '
        ...

    def get_waiter(self, waiterName: str) -> botocore.waiter.Waiter:
        if False:
            i = 10
            return i + 15
        '\n        Get an AWS Batch service waiter.\n\n        :param waiterName: The name of the waiter.  The name should match\n            the name (including the casing) of the key name in the waiter\n            model file (typically this is CamelCasing).\n\n        :return: a waiter object for the named AWS Batch service\n\n        .. note::\n            AWS Batch might not have any waiters (until botocore PR-1307 is released).\n\n            .. code-block:: python\n\n                import boto3\n\n                boto3.client("batch").waiter_names == []\n\n        .. seealso::\n\n            - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#waiters\n            - https://github.com/boto/botocore/pull/1307\n        '
        ...

    def submit_job(self, jobName: str, jobQueue: str, jobDefinition: str, arrayProperties: dict, parameters: dict, containerOverrides: dict, tags: dict) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Submit a Batch job.\n\n        :param jobName: the name for the AWS Batch job\n\n        :param jobQueue: the queue name on AWS Batch\n\n        :param jobDefinition: the job definition name on AWS Batch\n\n        :param arrayProperties: the same parameter that boto3 will receive\n\n        :param parameters: the same parameter that boto3 will receive\n\n        :param containerOverrides: the same parameter that boto3 will receive\n\n        :param tags: the same parameter that boto3 will receive\n\n        :return: an API response\n        '
        ...

    def terminate_job(self, jobId: str, reason: str) -> dict:
        if False:
            return 10
        '\n        Terminate a Batch job.\n\n        :param jobId: a job ID to terminate\n\n        :param reason: a reason to terminate job ID\n\n        :return: an API response\n        '
        ...

class BatchClientHook(AwsBaseHook):
    """
    Interact with AWS Batch.

    Provide thick wrapper around :external+boto3:py:class:`boto3.client("batch") <Batch.Client>`.

    :param max_retries: exponential back-off retries, 4200 = 48 hours;
        polling is only used when waiters is None
    :param status_retries: number of HTTP retries to get job status, 10;
        polling is only used when waiters is None

    .. note::
        Several methods use a default random delay to check or poll for job status, i.e.
        ``random.uniform(DEFAULT_DELAY_MIN, DEFAULT_DELAY_MAX)``
        Using a random interval helps to avoid AWS API throttle limits
        when many concurrent tasks request job-descriptions.

        To modify the global defaults for the range of jitter allowed when a
        random delay is used to check Batch job status, modify these defaults, e.g.:
        .. code-block::

            BatchClient.DEFAULT_DELAY_MIN = 0
            BatchClient.DEFAULT_DELAY_MAX = 5

        When explicit delay values are used, a 1 second random jitter is applied to the
        delay (e.g. a delay of 0 sec will be a ``random.uniform(0, 1)`` delay.  It is
        generally recommended that random jitter is added to API requests.  A
        convenience method is provided for this, e.g. to get a random delay of
        10 sec +/- 5 sec: ``delay = BatchClient.add_jitter(10, width=5, minima=0)``

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
        - https://docs.aws.amazon.com/general/latest/gr/api-retries.html
        - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """
    MAX_RETRIES = 4200
    STATUS_RETRIES = 10
    DEFAULT_DELAY_MIN = 1
    DEFAULT_DELAY_MAX = 10
    FAILURE_STATE = 'FAILED'
    SUCCESS_STATE = 'SUCCEEDED'
    RUNNING_STATE = 'RUNNING'
    INTERMEDIATE_STATES = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', RUNNING_STATE)
    COMPUTE_ENVIRONMENT_TERMINAL_STATUS = ('VALID', 'DELETED')
    COMPUTE_ENVIRONMENT_INTERMEDIATE_STATUS = ('CREATING', 'UPDATING', 'DELETING')
    JOB_QUEUE_TERMINAL_STATUS = ('VALID', 'DELETED')
    JOB_QUEUE_INTERMEDIATE_STATUS = ('CREATING', 'UPDATING', 'DELETING')

    def __init__(self, *args, max_retries: int | None=None, status_retries: int | None=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, client_type='batch', **kwargs)
        self.max_retries = max_retries or self.MAX_RETRIES
        self.status_retries = status_retries or self.STATUS_RETRIES

    @property
    def client(self) -> BatchProtocol | botocore.client.BaseClient:
        if False:
            print('Hello World!')
        "\n        An AWS API client for Batch services.\n\n        :return: a boto3 'batch' client for the ``.region_name``\n        "
        return self.conn

    def terminate_job(self, job_id: str, reason: str) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Terminate a Batch job.\n\n        :param job_id: a job ID to terminate\n\n        :param reason: a reason to terminate job ID\n\n        :return: an API response\n        '
        response = self.get_conn().terminate_job(jobId=job_id, reason=reason)
        self.log.info(response)
        return response

    def check_job_success(self, job_id: str) -> bool:
        if False:
            while True:
                i = 10
        "\n        Check the final status of the Batch job.\n\n        Return True if the job 'SUCCEEDED', else raise an AirflowException.\n\n        :param job_id: a Batch job ID\n\n        :raises: AirflowException\n        "
        job = self.get_job_description(job_id)
        job_status = job.get('status')
        if job_status == self.SUCCESS_STATE:
            self.log.info('AWS Batch job (%s) succeeded: %s', job_id, job)
            return True
        if job_status == self.FAILURE_STATE:
            raise AirflowException(f'AWS Batch job ({job_id}) failed: {job}')
        if job_status in self.INTERMEDIATE_STATES:
            raise AirflowException(f'AWS Batch job ({job_id}) is not complete: {job}')
        raise AirflowException(f'AWS Batch job ({job_id}) has unknown status: {job}')

    def wait_for_job(self, job_id: str, delay: int | float | None=None, get_batch_log_fetcher: Callable[[str], AwsTaskLogFetcher | None] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Wait for Batch job to complete.\n\n        :param job_id: a Batch job ID\n\n        :param delay: a delay before polling for job status\n\n        :param get_batch_log_fetcher : a method that returns batch_log_fetcher\n\n        :raises: AirflowException\n        '
        self.delay(delay)
        self.poll_for_job_running(job_id, delay)
        batch_log_fetcher = None
        try:
            if get_batch_log_fetcher:
                batch_log_fetcher = get_batch_log_fetcher(job_id)
                if batch_log_fetcher:
                    batch_log_fetcher.start()
            self.poll_for_job_complete(job_id, delay)
        finally:
            if batch_log_fetcher:
                batch_log_fetcher.stop()
                batch_log_fetcher.join()
        self.log.info('AWS Batch job (%s) has completed', job_id)

    def poll_for_job_running(self, job_id: str, delay: int | float | None=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Poll for job running.\n\n        The status that indicates a job is running or already complete are: 'RUNNING'|'SUCCEEDED'|'FAILED'.\n\n        So the status options that this will wait for are the transitions from:\n        'SUBMITTED'>'PENDING'>'RUNNABLE'>'STARTING'>'RUNNING'|'SUCCEEDED'|'FAILED'\n\n        The completed status options are included for cases where the status\n        changes too quickly for polling to detect a RUNNING status that moves\n        quickly from STARTING to RUNNING to completed (often a failure).\n\n        :param job_id: a Batch job ID\n\n        :param delay: a delay before polling for job status\n\n        :raises: AirflowException\n        "
        self.delay(delay)
        running_status = [self.RUNNING_STATE, self.SUCCESS_STATE, self.FAILURE_STATE]
        self.poll_job_status(job_id, running_status)

    def poll_for_job_complete(self, job_id: str, delay: int | float | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Poll for job completion.\n\n        The status that indicates job completion are: 'SUCCEEDED'|'FAILED'.\n\n        So the status options that this will wait for are the transitions from:\n        'SUBMITTED'>'PENDING'>'RUNNABLE'>'STARTING'>'RUNNING'>'SUCCEEDED'|'FAILED'\n\n        :param job_id: a Batch job ID\n\n        :param delay: a delay before polling for job status\n\n        :raises: AirflowException\n        "
        self.delay(delay)
        complete_status = [self.SUCCESS_STATE, self.FAILURE_STATE]
        self.poll_job_status(job_id, complete_status)

    def poll_job_status(self, job_id: str, match_status: list[str]) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        Poll for job status using an exponential back-off strategy (with max_retries).\n\n        :param job_id: a Batch job ID\n\n        :param match_status: a list of job status to match; the Batch job status are:\n            'SUBMITTED'|'PENDING'|'RUNNABLE'|'STARTING'|'RUNNING'|'SUCCEEDED'|'FAILED'\n\n\n        :raises: AirflowException\n        "
        for retries in range(1 + self.max_retries):
            if retries:
                pause = self.exponential_delay(retries)
                self.log.info('AWS Batch job (%s) status check (%d of %d) in the next %.2f seconds', job_id, retries, self.max_retries, pause)
                self.delay(pause)
            job = self.get_job_description(job_id)
            job_status = job.get('status')
            self.log.info('AWS Batch job (%s) check status (%s) in %s', job_id, job_status, match_status)
            if job_status in match_status:
                return True
        else:
            raise AirflowException(f'AWS Batch job ({job_id}) status checks exceed max_retries')

    def get_job_description(self, job_id: str) -> dict:
        if False:
            print('Hello World!')
        '\n        Get job description (using status_retries).\n\n        :param job_id: a Batch job ID\n\n        :return: an API response for describe jobs\n\n        :raises: AirflowException\n        '
        for retries in range(self.status_retries):
            if retries:
                pause = self.exponential_delay(retries)
                self.log.info('AWS Batch job (%s) description retry (%d of %d) in the next %.2f seconds', job_id, retries, self.status_retries, pause)
                self.delay(pause)
            try:
                response = self.get_conn().describe_jobs(jobs=[job_id])
                return self.parse_job_description(job_id, response)
            except botocore.exceptions.ClientError as err:
                if err.response.get('Error', {}).get('Code') != 'TooManyRequestsException':
                    raise
                self.log.warning('Ignored TooManyRequestsException error, original message: %r. Please consider to setup retries mode in boto3, check Amazon Provider AWS Connection documentation for more details.', str(err))
        else:
            raise AirflowException(f'AWS Batch job ({job_id}) description error: exceeded status_retries ({self.status_retries})')

    @staticmethod
    def parse_job_description(job_id: str, response: dict) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Parse job description to extract description for job_id.\n\n        :param job_id: a Batch job ID\n\n        :param response: an API response for describe jobs\n\n        :return: an API response to describe job_id\n\n        :raises: AirflowException\n        '
        jobs = response.get('jobs', [])
        matching_jobs = [job for job in jobs if job.get('jobId') == job_id]
        if len(matching_jobs) != 1:
            raise AirflowException(f'AWS Batch job ({job_id}) description error: response: {response}')
        return matching_jobs[0]

    def get_job_awslogs_info(self, job_id: str) -> dict[str, str] | None:
        if False:
            print('Hello World!')
        all_info = self.get_job_all_awslogs_info(job_id)
        if not all_info:
            return None
        if len(all_info) > 1:
            self.log.warning(f'AWS Batch job ({job_id}) has more than one log stream, only returning the first one.')
        return all_info[0]

    def get_job_all_awslogs_info(self, job_id: str) -> list[dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse job description to extract AWS CloudWatch information.\n\n        :param job_id: AWS Batch Job ID\n        '
        job_desc = self.get_job_description(job_id=job_id)
        job_node_properties = job_desc.get('nodeProperties', {})
        job_container_desc = job_desc.get('container', {})
        if job_node_properties:
            log_configs = [p.get('container', {}).get('logConfiguration', {}) for p in job_node_properties.get('nodeRangeProperties', {})]
            stream_names = [a.get('container', {}).get('logStreamName') for a in job_desc.get('attempts', [])]
        elif job_container_desc:
            log_configs = [job_container_desc.get('logConfiguration', {})]
            stream_name = job_container_desc.get('logStreamName')
            stream_names = [stream_name] if stream_name is not None else []
        else:
            raise AirflowException(f'AWS Batch job ({job_id}) is not a supported job type. Supported job types: container, array, multinode.')
        if any((c.get('logDriver', 'awslogs') != 'awslogs' for c in log_configs)):
            self.log.warning(f'AWS Batch job ({job_id}) uses non-aws log drivers. AWS CloudWatch logging disabled.')
            return []
        if not stream_names:
            self.log.warning(f"AWS Batch job ({job_id}) doesn't have any AWS CloudWatch Stream.")
            return []
        log_options = [c.get('options', {}) for c in log_configs]
        result = []
        for (stream, option) in itertools.product(stream_names, log_options):
            result.append({'awslogs_stream_name': stream, 'awslogs_group': option.get('awslogs-group', '/aws/batch/job'), 'awslogs_region': option.get('awslogs-region', self.conn_region_name)})
        return result

    @staticmethod
    def add_jitter(delay: int | float, width: int | float=1, minima: int | float=0) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Use delay +/- width for random jitter.\n\n        Adding jitter to status polling can help to avoid\n        AWS Batch API limits for monitoring Batch jobs with\n        a high concurrency in Airflow tasks.\n\n        :param delay: number of seconds to pause;\n            delay is assumed to be a positive number\n\n        :param width: delay +/- width for random jitter;\n            width is assumed to be a positive number\n\n        :param minima: minimum delay allowed;\n            minima is assumed to be a non-negative number\n\n        :return: uniform(delay - width, delay + width) jitter\n            and it is a non-negative number\n        '
        delay = abs(delay)
        width = abs(width)
        minima = abs(minima)
        lower = max(minima, delay - width)
        upper = delay + width
        return random.uniform(lower, upper)

    @staticmethod
    def delay(delay: int | float | None=None) -> None:
        if False:
            print('Hello World!')
        '\n        Pause execution for ``delay`` seconds.\n\n        :param delay: a delay to pause execution using ``time.sleep(delay)``;\n            a small 1 second jitter is applied to the delay.\n\n        .. note::\n            This method uses a default random delay, i.e.\n            ``random.uniform(DEFAULT_DELAY_MIN, DEFAULT_DELAY_MAX)``;\n            using a random interval helps to avoid AWS API throttle limits\n            when many concurrent tasks request job-descriptions.\n        '
        if delay is None:
            delay = random.uniform(BatchClientHook.DEFAULT_DELAY_MIN, BatchClientHook.DEFAULT_DELAY_MAX)
        else:
            delay = BatchClientHook.add_jitter(delay)
        time.sleep(delay)

    @staticmethod
    def exponential_delay(tries: int) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply an exponential back-off delay, with random jitter.\n\n        There is a maximum interval of 10 minutes (with random jitter between 3 and 10 minutes).\n        This is used in the :py:meth:`.poll_for_job_status` method.\n\n        Examples of behavior:\n\n        .. code-block:: python\n\n            def exp(tries):\n                max_interval = 600.0  # 10 minutes in seconds\n                delay = 1 + pow(tries * 0.6, 2)\n                delay = min(max_interval, delay)\n                print(delay / 3, delay)\n\n\n            for tries in range(10):\n                exp(tries)\n\n            #  0.33  1.0\n            #  0.45  1.35\n            #  0.81  2.44\n            #  1.41  4.23\n            #  2.25  6.76\n            #  3.33 10.00\n            #  4.65 13.95\n            #  6.21 18.64\n            #  8.01 24.04\n            # 10.05 30.15\n\n        .. seealso::\n\n            - https://docs.aws.amazon.com/general/latest/gr/api-retries.html\n            - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/\n\n        :param tries: Number of tries\n        '
        max_interval = 600.0
        delay = 1 + pow(tries * 0.6, 2)
        delay = min(max_interval, delay)
        return random.uniform(delay / 3, delay)