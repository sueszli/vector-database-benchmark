"""Tests for methods in the cloud_taskqueue_services."""
from __future__ import annotations
import datetime
import json
from core.domain import taskqueue_services
from core.platform.taskqueue import cloud_taskqueue_services
from core.tests import test_utils
from google.api_core import retry as retry_lib
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from typing import Any, Dict, Optional

class CloudTaskqueueServicesUnitTests(test_utils.TestBase):
    """Tests for cloud_taskqueue_services."""

    class Response:
        """Mock for the response object that is returned from a Cloud
        Tasks query.
        """

        def __init__(self, name: str) -> None:
            if False:
                i = 10
                return i + 15
            self.name = name

    def test_http_task_scheduled_immediately_sends_correct_request(self) -> None:
        if False:
            print('Hello World!')
        queue_name = 'queue'
        dummy_url = '/task/dummy_handler'
        payload = {'fn_identifier': taskqueue_services.FUNCTION_ID_DELETE_EXPS_FROM_USER_MODELS, 'args': [['1', '2', '3']], 'kwargs': {}}
        task_name = 'task1'

        def mock_create_task(parent: str, task: Dict[str, Any], retry: Optional[retry_lib.Retry]=None) -> CloudTaskqueueServicesUnitTests.Response:
            if False:
                print('Hello World!')
            self.assertIsInstance(retry, retry_lib.Retry)
            self.assertEqual(parent, u'projects/dev-project-id/locations/us-central1/queues/queue')
            self.assertEqual(task, {'app_engine_http_request': {'http_method': tasks_v2.types.HttpMethod.POST, 'relative_uri': dummy_url, 'headers': {'Content-type': 'application/json'}, 'body': json.dumps(payload).encode()}, 'name': task_name})
            return self.Response(task_name)
        with self.swap(cloud_taskqueue_services.CLIENT, 'create_task', mock_create_task):
            cloud_taskqueue_services.create_http_task(queue_name, dummy_url, payload=payload, task_name=task_name)

    def test_http_task_scheduled_for_later_sends_correct_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        queue_name = 'queue'
        dummy_url = '/task/dummy_handler'
        payload = {'fn_identifier': taskqueue_services.FUNCTION_ID_DELETE_EXPS_FROM_USER_MODELS, 'args': [['1', '2', '3']], 'kwargs': {}}
        datetime_to_execute_task = datetime.datetime.utcnow() + datetime.timedelta(seconds=20)
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(datetime_to_execute_task)
        task_name = 'task1'

        def mock_create_task(parent: str, task: Dict[str, Any], retry: Optional[retry_lib.Retry]=None) -> CloudTaskqueueServicesUnitTests.Response:
            if False:
                return 10
            self.assertIsInstance(retry, retry_lib.Retry)
            self.assertEqual(parent, u'projects/dev-project-id/locations/us-central1/queues/queue')
            self.assertEqual(task, {'app_engine_http_request': {'http_method': tasks_v2.types.HttpMethod.POST, 'relative_uri': dummy_url, 'headers': {'Content-type': 'application/json'}, 'body': json.dumps(payload).encode()}, 'schedule_time': timestamp, 'name': task_name})
            return self.Response(task_name)
        with self.swap(cloud_taskqueue_services.CLIENT, 'create_task', mock_create_task):
            cloud_taskqueue_services.create_http_task(queue_name, dummy_url, payload=payload, scheduled_for=datetime_to_execute_task, task_name=task_name)