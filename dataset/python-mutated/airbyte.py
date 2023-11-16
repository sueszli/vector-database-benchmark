from __future__ import annotations
import time
from typing import Any
from airflow.exceptions import AirflowException
from airflow.providers.http.hooks.http import HttpHook

class AirbyteHook(HttpHook):
    """
    Hook for Airbyte API.

    :param airbyte_conn_id: Optional. The name of the Airflow connection to get
        connection information for Airbyte. Defaults to "airbyte_default".
    :param api_version: Optional. Airbyte API version. Defaults to "v1".
    """
    conn_name_attr = 'airbyte_conn_id'
    default_conn_name = 'airbyte_default'
    conn_type = 'airbyte'
    hook_name = 'Airbyte'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    CANCELLED = 'cancelled'
    PENDING = 'pending'
    FAILED = 'failed'
    ERROR = 'error'
    INCOMPLETE = 'incomplete'

    def __init__(self, airbyte_conn_id: str='airbyte_default', api_version: str='v1') -> None:
        if False:
            while True:
                i = 10
        super().__init__(http_conn_id=airbyte_conn_id)
        self.api_version: str = api_version

    def wait_for_job(self, job_id: str | int, wait_seconds: float=3, timeout: float | None=3600) -> None:
        if False:
            while True:
                i = 10
        '\n        Poll a job to check if it finishes.\n\n        :param job_id: Required. Id of the Airbyte job\n        :param wait_seconds: Optional. Number of seconds between checks.\n        :param timeout: Optional. How many seconds wait for job to be ready.\n            Used only if ``asynchronous`` is False.\n        '
        state = None
        start = time.monotonic()
        while True:
            if timeout and start + timeout < time.monotonic():
                raise AirflowException(f'Timeout: Airbyte job {job_id} is not ready after {timeout}s')
            time.sleep(wait_seconds)
            try:
                job = self.get_job(job_id=int(job_id))
                state = job.json()['job']['status']
            except AirflowException as err:
                self.log.info('Retrying. Airbyte API returned server error when waiting for job: %s', err)
                continue
            if state in (self.RUNNING, self.PENDING, self.INCOMPLETE):
                continue
            if state == self.SUCCEEDED:
                break
            if state == self.ERROR:
                raise AirflowException(f'Job failed:\n{job}')
            elif state == self.CANCELLED:
                raise AirflowException(f'Job was cancelled:\n{job}')
            else:
                raise Exception(f'Encountered unexpected state `{state}` for job_id `{job_id}`')

    def submit_sync_connection(self, connection_id: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Submit a job to a Airbyte server.\n\n        :param connection_id: Required. The ConnectionId of the Airbyte Connection.\n        '
        return self.run(endpoint=f'api/{self.api_version}/connections/sync', json={'connectionId': connection_id}, headers={'accept': 'application/json'})

    def get_job(self, job_id: int) -> Any:
        if False:
            return 10
        '\n        Get the resource representation for a job in Airbyte.\n\n        :param job_id: Required. Id of the Airbyte job\n        '
        return self.run(endpoint=f'api/{self.api_version}/jobs/get', json={'id': job_id}, headers={'accept': 'application/json'})

    def cancel_job(self, job_id: int) -> Any:
        if False:
            print('Hello World!')
        '\n        Cancel the job when task is cancelled.\n\n        :param job_id: Required. Id of the Airbyte job\n        '
        return self.run(endpoint=f'api/{self.api_version}/jobs/cancel', json={'id': job_id}, headers={'accept': 'application/json'})

    def test_connection(self):
        if False:
            return 10
        'Tests the Airbyte connection by hitting the health API.'
        self.method = 'GET'
        try:
            res = self.run(endpoint=f'api/{self.api_version}/health', headers={'accept': 'application/json'}, extra_options={'check_response': False})
            if res.status_code == 200:
                return (True, 'Connection successfully tested')
            else:
                return (False, res.text)
        except Exception as e:
            return (False, str(e))
        finally:
            self.method = 'POST'