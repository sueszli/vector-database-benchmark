"""Provides functionality for Google Cloud Tasks-related operations."""
from __future__ import annotations
import datetime
import json
import logging
from core import feconf
from core.constants import constants
from google import auth
from google.api_core import retry
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from typing import Any, Dict, Optional
CLIENT = tasks_v2.CloudTasksClient(credentials=auth.credentials.AnonymousCredentials() if constants.EMULATOR_MODE else auth.default()[0])

def create_http_task(queue_name: str, url: str, payload: Optional[Dict[str, Any]]=None, scheduled_for: Optional[datetime.datetime]=None, task_name: Optional[str]=None) -> tasks_v2.types.Task:
    if False:
        i = 10
        return i + 15
    'Creates an http task with the correct http headers/payload and sends\n    that task to the Cloud Tasks API. An http task is an asynchronous task that\n    consists of a post request to a specified url with the specified payload.\n    The post request will be made by the Cloud Tasks Cloud Service when the\n    `scheduled_for` time is reached.\n\n    Args:\n        queue_name: str. The name of the queue to add the http task to.\n        url: str. URL of the handler function.\n        payload: dict(str : *). Payload to pass to the request. Defaults\n            to None if no payload is required.\n        scheduled_for: datetime|None. The naive datetime object for the\n            time to execute the task. Pass in None for immediate execution.\n        task_name: str|None. Optional. The name of the task.\n\n    Returns:\n        Response. Response object that is returned by the Cloud Tasks API.\n    '
    parent = CLIENT.queue_path(feconf.OPPIA_PROJECT_ID, feconf.GOOGLE_APP_ENGINE_REGION, queue_name)
    task: Dict[str, Any] = {'app_engine_http_request': {'http_method': tasks_v2.types.HttpMethod.POST, 'relative_uri': url}}
    if payload is not None:
        if isinstance(payload, dict):
            payload_text = json.dumps(payload)
            task['app_engine_http_request']['headers'] = {'Content-type': 'application/json'}
        converted_payload = payload_text.encode('utf-8')
        task['app_engine_http_request']['body'] = converted_payload
    if scheduled_for is not None:
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(scheduled_for)
        task['schedule_time'] = timestamp
    if task_name is not None:
        task['name'] = task_name
    response = CLIENT.create_task(parent=parent, task=task, retry=retry.Retry())
    logging.info('Created task %s' % response.name)
    return response