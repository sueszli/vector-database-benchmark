"""Tests for the domain taskqueue services."""
from __future__ import annotations
import datetime
from core import feconf
from core import utils
from core.domain import taskqueue_services
from core.platform import models
from core.tests import test_utils
from typing import Dict, Optional, Set
MYPY = False
if MYPY:
    from mypy_imports import platform_taskqueue_services
platform_taskqueue_services = models.Registry.import_taskqueue_services()

class TaskqueueDomainServicesUnitTests(test_utils.TestBase):
    """Tests for domain taskqueue services."""

    def test_exception_raised_when_deferred_payload_is_not_serializable(self) -> None:
        if False:
            return 10

        class NonSerializableArgs:
            """Object that is not JSON serializable."""

            def __init__(self) -> None:
                if False:
                    while True:
                        i = 10
                self.x = 1
                self.y = 2
        arg1 = NonSerializableArgs()
        serialization_exception = self.assertRaisesRegex(ValueError, 'The args or kwargs passed to the deferred call with function_identifier, %s, are not json serializable.' % taskqueue_services.FUNCTION_ID_UPDATE_STATS)
        with serialization_exception:
            taskqueue_services.defer(taskqueue_services.FUNCTION_ID_UPDATE_STATS, taskqueue_services.QUEUE_NAME_DEFAULT, arg1)

    def test_exception_raised_when_email_task_params_is_not_serializable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        params: Dict[str, Set[str]] = {'param1': set()}
        serialization_exception = self.assertRaisesRegex(ValueError, 'The params added to the email task call cannot be json serialized')
        with serialization_exception:
            taskqueue_services.enqueue_task(feconf.TASK_URL_FEEDBACK_MESSAGE_EMAILS, params, 0)

    def test_defer_makes_the_correct_request(self) -> None:
        if False:
            while True:
                i = 10
        correct_fn_identifier = '/task/deferredtaskshandler'
        correct_args = (1, 2, 3)
        correct_kwargs = {'a': 'b', 'c': 'd'}
        expected_queue_name = taskqueue_services.QUEUE_NAME_EMAILS
        expected_url = feconf.TASK_URL_DEFERRED
        expected_payload = {'fn_identifier': correct_fn_identifier, 'args': correct_args, 'kwargs': correct_kwargs}
        create_http_task_swap = self.swap_with_checks(platform_taskqueue_services, 'create_http_task', lambda queue_name, url, payload=None, scheduled_for=None: None, expected_kwargs=[{'queue_name': expected_queue_name, 'url': expected_url, 'payload': expected_payload}])
        with create_http_task_swap:
            taskqueue_services.defer(correct_fn_identifier, taskqueue_services.QUEUE_NAME_EMAILS, *correct_args, **correct_kwargs)

    def test_enqueue_task_makes_the_correct_request(self) -> None:
        if False:
            while True:
                i = 10
        correct_payload = {'user_id': '1'}
        correct_url = feconf.TASK_URL_FEEDBACK_MESSAGE_EMAILS
        correct_queue_name = taskqueue_services.QUEUE_NAME_EMAILS

        def mock_create_http_task(queue_name: str, url: str, payload: Optional[Dict[str, str]]=None, scheduled_for: Optional[datetime.datetime]=None, task_name: Optional[str]=None) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(queue_name, correct_queue_name)
            self.assertEqual(url, correct_url)
            self.assertEqual(payload, correct_payload)
            self.assertIsNotNone(scheduled_for)
            self.assertIsNone(task_name)
        swap_create_http_task = self.swap(platform_taskqueue_services, 'create_http_task', mock_create_http_task)
        with swap_create_http_task:
            taskqueue_services.enqueue_task(correct_url, correct_payload, 0)

    def test_that_queue_names_are_in_sync_with_queue_yaml_file(self) -> None:
        if False:
            i = 10
            return i + 15
        'Checks that all of the queues that are instantiated in the queue.yaml\n        file has a corresponding QUEUE_NAME_* constant instantiated in\n        taskqueue_services.\n        '
        queue_name_dict = {}
        with utils.open_file('queue.yaml', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'name' in line:
                    queue_name = line.split(':')[1]
                    queue_name_dict[queue_name.strip()] = False
        attributes = dir(taskqueue_services)
        for attribute in attributes:
            if attribute.startswith('QUEUE_NAME_'):
                queue_name_dict[getattr(taskqueue_services, attribute)] = True
        for (queue_name, in_taskqueue_services) in queue_name_dict.items():
            self.assertTrue(in_taskqueue_services)