"""
An Action Logger module.

Singleton pattern has been applied into this module so that registered
callbacks can be used all through the same python process.
"""
from __future__ import annotations
import json
import logging
from typing import Callable

def register_pre_exec_callback(action_logger):
    if False:
        return 10
    'Register more action_logger function callback for pre-execution.\n\n    This function callback is expected to be called with keyword args.\n    For more about the arguments that is being passed to the callback,\n    refer to airflow.utils.cli.action_logging().\n\n    :param action_logger: An action logger function\n    :return: None\n    '
    logging.debug('Adding %s to pre execution callback', action_logger)
    __pre_exec_callbacks.append(action_logger)

def register_post_exec_callback(action_logger):
    if False:
        i = 10
        return i + 15
    'Register more action_logger function callback for post-execution.\n\n    This function callback is expected to be called with keyword args.\n    For more about the arguments that is being passed to the callback,\n    refer to airflow.utils.cli.action_logging().\n\n    :param action_logger: An action logger function\n    :return: None\n    '
    logging.debug('Adding %s to post execution callback', action_logger)
    __post_exec_callbacks.append(action_logger)

def on_pre_execution(**kwargs):
    if False:
        i = 10
        return i + 15
    "Call callbacks before execution.\n\n    Note that any exception from callback will be logged but won't be propagated.\n\n    :param kwargs:\n    :return: None\n    "
    logging.debug('Calling callbacks: %s', __pre_exec_callbacks)
    for callback in __pre_exec_callbacks:
        try:
            callback(**kwargs)
        except Exception:
            logging.exception('Failed on pre-execution callback using %s', callback)

def on_post_execution(**kwargs):
    if False:
        print('Hello World!')
    "Call callbacks after execution.\n\n    As it's being called after execution, it can capture status of execution,\n    duration, etc. Note that any exception from callback will be logged but\n    won't be propagated.\n\n    :param kwargs:\n    :return: None\n    "
    logging.debug('Calling callbacks: %s', __post_exec_callbacks)
    for callback in __post_exec_callbacks:
        try:
            callback(**kwargs)
        except Exception:
            logging.exception('Failed on post-execution callback using %s', callback)

def default_action_log(sub_command, user, task_id, dag_id, execution_date, host_name, full_command, **_):
    if False:
        return 10
    '\n    Behave similar to ``action_logging``; default action logger callback.\n\n    The difference is this function uses the global ORM session, and pushes a\n    ``Log`` row into the database instead of actually logging.\n    '
    from sqlalchemy.exc import OperationalError, ProgrammingError
    from airflow.models.log import Log
    from airflow.utils import timezone
    from airflow.utils.session import create_session
    try:
        with create_session() as session:
            extra = json.dumps({'host_name': host_name, 'full_command': full_command})
            session.bulk_insert_mappings(Log, [{'event': f'cli_{sub_command}', 'task_instance': None, 'owner': user, 'extra': extra, 'task_id': task_id, 'dag_id': dag_id, 'execution_date': execution_date, 'dttm': timezone.utcnow()}])
    except (OperationalError, ProgrammingError) as e:
        expected = ['"log" does not exist', 'no such table', "log' doesn't exist", "Invalid object name 'log'"]
        error_is_ok = e.args and any((x in e.args[0] for x in expected))
        if not error_is_ok:
            logging.warning('Failed to log action %s', e)
    except Exception as e:
        logging.warning('Failed to log action %s', e)
__pre_exec_callbacks: list[Callable] = []
__post_exec_callbacks: list[Callable] = []
register_pre_exec_callback(default_action_log)