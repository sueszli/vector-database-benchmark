"""Tests for methods in the dev_mode_taskqueue_services."""
from __future__ import annotations
import datetime
from core import feconf
from core.domain import taskqueue_services
from core.platform.taskqueue import dev_mode_taskqueue_services
from core.tests import test_utils
import requests
from typing import Any, Dict, Optional

class DevModeTaskqueueServicesUnitTests(test_utils.TestBase):
    """Tests for dev_mode_taskqueue_services."""

    def test_creating_dev_mode_task_will_create_the_correct_post_request(self) -> None:
        if False:
            while True:
                i = 10
        correct_queue_name = 'dummy_queue'
        dummy_url = '/dummy_handler'
        correct_payload = {'fn_identifier': taskqueue_services.FUNCTION_ID_DELETE_EXPS_FROM_USER_MODELS, 'args': [['1', '2', '3']], 'kwargs': {}}
        correct_task_name = 'task1'

        def mock_create_task(queue_name: str, url: str, payload: Dict[str, Any], scheduled_for: Optional[datetime.datetime]=None, task_name: Optional[str]=None) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(queue_name, correct_queue_name)
            self.assertEqual(url, dummy_url)
            self.assertEqual(payload, correct_payload)
            self.assertEqual(task_name, correct_task_name)
        swap_create_task = self.swap(dev_mode_taskqueue_services.CLIENT, 'create_task', mock_create_task)
        with swap_create_task:
            dev_mode_taskqueue_services.create_http_task(correct_queue_name, dummy_url, correct_payload, task_name=correct_task_name)

    def test_task_handler_will_create_the_correct_post_request(self) -> None:
        if False:
            i = 10
            return i + 15
        queue_name = 'dummy_queue'
        dummy_url = '/dummy_handler'
        correct_port = dev_mode_taskqueue_services.GOOGLE_APP_ENGINE_PORT
        correct_payload = {'fn_identifier': taskqueue_services.FUNCTION_ID_DELETE_EXPS_FROM_USER_MODELS, 'args': [['1', '2', '3']], 'kwargs': {}}
        task_name = 'task1'
        correct_headers = {'X-Appengine-QueueName': queue_name, 'X-Appengine-TaskName': task_name, 'X-Appengine-TaskRetryCount': '0', 'X-Appengine-TaskExecutionCount': '0', 'X-Appengine-TaskETA': '0', 'X-AppEngine-Fake-Is-Admin': '1', 'method': 'POST'}

        def mock_post(url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: int) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(url, 'http://localhost:%s%s' % (correct_port, dummy_url))
            self.assertEqual(json, correct_payload)
            self.assertEqual(headers, correct_headers)
            self.assertEqual(timeout, feconf.DEFAULT_TASKQUEUE_TIMEOUT_SECONDS)
        swap_post = self.swap(requests, 'post', mock_post)
        with swap_post:
            dev_mode_taskqueue_services._task_handler(dummy_url, correct_payload, queue_name, task_name=task_name)