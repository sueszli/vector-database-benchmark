"""Provides a taskqueue API for the platform layer in DEV_MODE."""
from __future__ import annotations
import os
from core import feconf
from core.platform.taskqueue import cloud_tasks_emulator
import requests
from typing import TYPE_CHECKING, Any, Dict, Optional
if TYPE_CHECKING:
    import datetime
GOOGLE_APP_ENGINE_PORT = os.environ['PORT'] if 'PORT' in os.environ else '8181'

def _task_handler(url: str, payload: Dict[str, Any], queue_name: str, task_name: Optional[str]=None) -> None:
    if False:
        print('Hello World!')
    'Makes a POST request to the task URL.\n\n    Args:\n        url: str. URL of the handler function.\n        payload: dict(str : *). Payload to pass to the request. Defaults\n            to None if no payload is required.\n        queue_name: str. The name of the queue to add the task to.\n        task_name: str|None. Optional. The name of the task.\n    '
    headers: Dict[str, str] = {}
    headers['X-Appengine-QueueName'] = queue_name
    headers['X-Appengine-TaskName'] = task_name or 'task_without_name'
    headers['X-Appengine-TaskRetryCount'] = '0'
    headers['X-Appengine-TaskExecutionCount'] = '0'
    headers['X-Appengine-TaskETA'] = '0'
    headers['X-AppEngine-Fake-Is-Admin'] = '1'
    headers['method'] = 'POST'
    complete_url = 'http://localhost:%s%s' % (GOOGLE_APP_ENGINE_PORT, url)
    requests.post(complete_url, json=payload, headers=headers, timeout=feconf.DEFAULT_TASKQUEUE_TIMEOUT_SECONDS)
CLIENT = cloud_tasks_emulator.Emulator(task_handler=_task_handler)

def create_http_task(queue_name: str, url: str, payload: Optional[Dict[str, Any]]=None, scheduled_for: Optional[datetime.datetime]=None, task_name: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Creates a Task in the corresponding queue that will be executed when\n    the 'scheduled_for' countdown expires using the cloud tasks emulator.\n\n    Args:\n        queue_name: str. The name of the queue to add the task to.\n        url: str. URL of the handler function.\n        payload: dict(str : *). Payload to pass to the request. Defaults\n            to None if no payload is required.\n        scheduled_for: datetime|None. The naive datetime object for the\n            time to execute the task. Pass in None for immediate execution.\n        task_name: str|None. Optional. The name of the task.\n    "
    CLIENT.create_task(queue_name, url, payload=payload, scheduled_for=scheduled_for, task_name=task_name)