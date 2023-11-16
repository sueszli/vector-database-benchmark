from __future__ import annotations
import logging
import time
from typing import Callable
from airflow.exceptions import AirflowException
log = logging.getLogger(__name__)

def waiter(get_state_callable: Callable, get_state_args: dict, parse_response: list, desired_state: set, failure_states: set, object_type: str, action: str, countdown: int | float | None=25 * 60, check_interval_seconds: int=60) -> None:
    if False:
        while True:
            i = 10
    '\n    Call get_state_callable until it reaches the desired_state or the failure_states.\n\n    PLEASE NOTE:  While not yet deprecated, we are moving away from this method\n                  and encourage using the custom boto waiters as explained in\n                  https://github.com/apache/airflow/tree/main/airflow/providers/amazon/aws/waiters\n\n    :param get_state_callable: A callable to run until it returns True\n    :param get_state_args: Arguments to pass to get_state_callable\n    :param parse_response: Dictionary keys to extract state from response of get_state_callable\n    :param desired_state: Wait until the getter returns this value\n    :param failure_states: A set of states which indicate failure and should throw an\n        exception if any are reached before the desired_state\n    :param object_type: Used for the reporting string. What are you waiting for? (application, job, etc)\n    :param action: Used for the reporting string. What action are you waiting for? (created, deleted, etc)\n    :param countdown: Number of seconds the waiter should wait for the desired state before timing out.\n        Defaults to 25 * 60 seconds. None = infinite.\n    :param check_interval_seconds: Number of seconds waiter should wait before attempting\n        to retry get_state_callable. Defaults to 60 seconds.\n    '
    while True:
        state = get_state(get_state_callable(**get_state_args), parse_response)
        if state in desired_state:
            break
        if state in failure_states:
            raise AirflowException(f'{object_type.title()} reached failure state {state}.')
        if countdown is None:
            countdown = float('inf')
        if countdown > check_interval_seconds:
            countdown -= check_interval_seconds
            log.info('Waiting for %s to be %s.', object_type.lower(), action.lower())
            time.sleep(check_interval_seconds)
        else:
            message = f'{object_type.title()} still not {action.lower()} after the allocated time limit.'
            log.error(message)
            raise RuntimeError(message)

def get_state(response, keys) -> str:
    if False:
        for i in range(10):
            print('nop')
    value = response
    for key in keys:
        if value is not None:
            value = value.get(key, None)
    return value