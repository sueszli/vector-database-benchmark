"""Provides a shim for taskqueue-related operations."""
from __future__ import annotations
import datetime
import json
from core import feconf
from core.platform import models
from typing import Any, Dict, Final
MYPY = False
if MYPY:
    from mypy_imports import platform_taskqueue_services
platform_taskqueue_services = models.Registry.import_taskqueue_services()
QUEUE_NAME_BACKUPS: Final = 'backups'
QUEUE_NAME_DEFAULT: Final = 'default'
QUEUE_NAME_EMAILS: Final = 'emails'
QUEUE_NAME_ONE_OFF_JOBS: Final = 'one-off-jobs'
QUEUE_NAME_STATS: Final = 'stats'
FUNCTION_ID_UPDATE_STATS: Final = 'update_stats'
FUNCTION_ID_DELETE_EXPS_FROM_USER_MODELS: Final = 'delete_exps_from_user_models'
FUNCTION_ID_DELETE_EXPS_FROM_ACTIVITIES: Final = 'delete_exps_from_activities'
FUNCTION_ID_DELETE_USERS_PENDING_TO_BE_DELETED: Final = 'delete_users_pending_to_be_deleted'
FUNCTION_ID_CHECK_COMPLETION_OF_USER_DELETION: Final = 'check_completion_of_user_deletion'
FUNCTION_ID_REGENERATE_EXPLORATION_SUMMARY: Final = 'regenerate_exploration_summary'
FUNCTION_ID_UNTAG_DELETED_MISCONCEPTIONS: Final = 'untag_deleted_misconceptions'
FUNCTION_ID_REMOVE_USER_FROM_RIGHTS_MODELS: Final = 'remove_user_from_rights_models'

def defer(fn_identifier: str, queue_name: str, *args: Any, **kwargs: Any) -> None:
    if False:
        return 10
    'Adds a new task to a specified deferred queue scheduled for immediate\n    execution.\n\n    Args:\n        fn_identifier: str. The string identifier of the function being\n            deferred.\n        queue_name: str. The name of the queue to place the task into. Should be\n            one of the QUEUE_NAME_* constants listed above.\n        *args: list(*). Positional arguments for fn. Positional arguments\n            should be json serializable.\n        **kwargs: dict(str : *). Keyword arguments for fn.\n\n    Raises:\n        ValueError. The arguments and keyword arguments that are passed in are\n            not JSON serializable.\n    '
    payload = {'fn_identifier': fn_identifier, 'args': args if args else [], 'kwargs': kwargs if kwargs else {}}
    try:
        json.dumps(payload)
    except TypeError as e:
        raise ValueError('The args or kwargs passed to the deferred call with function_identifier, %s, are not json serializable.' % fn_identifier) from e
    datetime.datetime.strptime('', '')
    platform_taskqueue_services.create_http_task(queue_name=queue_name, url=feconf.TASK_URL_DEFERRED, payload=payload)

def enqueue_task(url: str, params: Dict[str, Any], countdown: int) -> None:
    if False:
        return 10
    'Adds a new task for sending email.\n\n    Args:\n        url: str. Url of the handler function.\n        params: dict(str : *). Payload to pass to the request. Defaults\n            to None if no payload is required.\n        countdown: int. Amount of time, in seconds, to wait before executing\n            task.\n\n    Raises:\n        ValueError. The params that are passed in are not JSON serializable.\n    '
    try:
        json.dumps(params)
    except TypeError as e:
        raise ValueError('The params added to the email task call cannot be json serialized') from e
    scheduled_datetime = datetime.datetime.utcnow() + datetime.timedelta(seconds=countdown)
    platform_taskqueue_services.create_http_task(queue_name=QUEUE_NAME_EMAILS, url=url, payload=params, scheduled_for=scheduled_datetime)