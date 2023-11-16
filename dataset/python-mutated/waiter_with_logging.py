from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any
import jmespath
from botocore.exceptions import WaiterError
from airflow.exceptions import AirflowException
if TYPE_CHECKING:
    from botocore.waiter import Waiter

def wait(waiter: Waiter, waiter_delay: int, waiter_max_attempts: int, args: dict[str, Any], failure_message: str, status_message: str, status_args: list[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Use a boto waiter to poll an AWS service for the specified state.\n\n    Although this function uses boto waiters to poll the state of the\n    service, it logs the response of the service after every attempt,\n    which is not currently supported by boto waiters.\n\n    :param waiter: The boto waiter to use.\n    :param waiter_delay: The amount of time in seconds to wait between attempts.\n    :param waiter_max_attempts: The maximum number of attempts to be made.\n    :param args: The arguments to pass to the waiter.\n    :param failure_message: The message to log if a failure state is reached.\n    :param status_message: The message logged when printing the status of the service.\n    :param status_args: A list containing the JMESPath queries to retrieve status information from\n        the waiter response.\n        e.g.\n        response = {"Cluster": {"state": "CREATING"}}\n        status_args = ["Cluster.state"]\n\n        response = {\n        "Clusters": [{"state": "CREATING", "details": "User initiated."},]\n        }\n        status_args = ["Clusters[0].state", "Clusters[0].details"]\n    '
    log = logging.getLogger(__name__)
    for attempt in range(waiter_max_attempts):
        if attempt:
            time.sleep(waiter_delay)
        try:
            waiter.wait(**args, WaiterConfig={'MaxAttempts': 1})
        except WaiterError as error:
            if 'terminal failure' in str(error):
                log.error('%s: %s', failure_message, _LazyStatusFormatter(status_args, error.last_response))
                raise AirflowException(f'{failure_message}: {error}')
            log.info('%s: %s', status_message, _LazyStatusFormatter(status_args, error.last_response))
        else:
            break
    else:
        raise AirflowException('Waiter error: max attempts reached')

async def async_wait(waiter: Waiter, waiter_delay: int, waiter_max_attempts: int, args: dict[str, Any], failure_message: str, status_message: str, status_args: list[str]):
    """
    Use an async boto waiter to poll an AWS service for the specified state.

    Although this function uses boto waiters to poll the state of the
    service, it logs the response of the service after every attempt,
    which is not currently supported by boto waiters.

    :param waiter: The boto waiter to use.
    :param waiter_delay: The amount of time in seconds to wait between attempts.
    :param waiter_max_attempts: The maximum number of attempts to be made.
    :param args: The arguments to pass to the waiter.
    :param failure_message: The message to log if a failure state is reached.
    :param status_message: The message logged when printing the status of the service.
    :param status_args: A list containing the JMESPath queries to retrieve status information from
        the waiter response.
        e.g.
        response = {"Cluster": {"state": "CREATING"}}
        status_args = ["Cluster.state"]

        response = {
        "Clusters": [{"state": "CREATING", "details": "User initiated."},]
        }
        status_args = ["Clusters[0].state", "Clusters[0].details"]
    """
    log = logging.getLogger(__name__)
    for attempt in range(waiter_max_attempts):
        if attempt:
            await asyncio.sleep(waiter_delay)
        try:
            await waiter.wait(**args, WaiterConfig={'MaxAttempts': 1})
        except WaiterError as error:
            if 'terminal failure' in str(error):
                log.error('%s: %s', failure_message, _LazyStatusFormatter(status_args, error.last_response))
                raise AirflowException(f'{failure_message}: {error}')
            log.info('%s: %s', status_message, _LazyStatusFormatter(status_args, error.last_response))
        else:
            break
    else:
        raise AirflowException('Waiter error: max attempts reached')

class _LazyStatusFormatter:
    """
    Contains the info necessary to extract the status from a response; only computes the value when necessary.

    Used to avoid computations if the logs are disabled at the given level.
    """

    def __init__(self, jmespath_queries: list[str], response: dict[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        self.jmespath_queries = jmespath_queries
        self.response = response

    def __str__(self):
        if False:
            print('Hello World!')
        'Loop through the args list and generate a string containing values from the waiter response.'
        values = []
        for query in self.jmespath_queries:
            value = jmespath.search(query, self.response)
            if value is not None and value != '':
                values.append(str(value))
        return ' - '.join(values)